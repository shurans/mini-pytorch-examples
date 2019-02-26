'''Train unet for surface normals
'''

import os
import glob
import io

from tensorboardX import SummaryWriter
from termcolor import colored
import yaml
from attrdict import AttrDict
import imageio
import numpy as np

from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

import models.unet_normals as unet
import dataloader
from loss_functions import loss_fn_cosine, loss_fn_radians

print('Inference of Surface Normal Estimation model. Loading checkpoint...')

###################### Config #############################
CONFIG_FILE_PATH = 'config/config.yaml'
with open(CONFIG_FILE_PATH) as fd:
    config_yaml = yaml.safe_load(fd)
config = AttrDict(config_yaml)

# Read config file stored in the model checkpoint to re-use it's params
if not os.path.isfile(config.eval.pathWeightsFile):
    raise ValueError('Invalid path to the given weights file in config. The file "{}" does not exist'.format(
        config.eval.pathWeightsFile))

CHECKPOINT = torch.load(config.eval.pathWeightsFile, map_location='cpu')
if 'model_state_dict' in CHECKPOINT:
    print(colored('Loaded data from checkpoint {}'.format(config.eval.pathWeightsFile), 'green'))

    prev_loss = CHECKPOINT['epoch_loss']
    config_checkpoint_dict = CHECKPOINT['config']
    config_checkpoint = AttrDict(config_checkpoint_dict)

    print('    Last Epoch Loss:', prev_loss)
    print('    Config from Checkpoint:\n', config_checkpoint_dict, '\n\n')
else:
    raise ValueError('The checkpoint file does not have model_state_dict in it.\
                     Please use the newer checkpoint files!')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Check for results store dir
if not os.path.isdir(config.eval.resultsDirSynthetic):
    print(colored('The dir to store results "{}" does not exist. Creating dir'.format(
        config.eval.resultsDirSynthetic), 'red'))
    os.makedirs(config.eval.resultsDirSynthetic)
if not os.path.isdir(config.eval.resultsDirReal):
    print(colored('The dir to store results "{}" does not exist. Creating dir'.format(
        config.eval.resultsDirReal), 'red'))
    os.makedirs(config.eval.resultsDirReal)

###################### DataLoader #############################
# Make new dataloaders for each synthetic dataset
# CHECK DATALOADER FOR EXAMPLE OF AUGMENTATIONS
db_test_list_synthetic = []
for dataset in config.eval.datasets_synthetic:
    dataset = dataloader.SurfaceNormalsDataset(
        input_dir=dataset.images,
        label_dir=dataset.labels,
        transform=None,
        input_only=None
    )
    db_test_list_synthetic.append(dataset)

# Make new dataloaders for each real dataset
db_test_list_real = []
for dataset in config.eval.datasets_real:
    dataset = dataloader.SurfaceNormalsRealImagesDataset(
        input_dir=dataset.images,
        imgHeight=config_checkpoint.train.imgHeight,
        imgWidth=config_checkpoint.train.imgWidth
    )
    db_test_list_real.append(dataset)


db_test_synthetic = torch.utils.data.ConcatDataset(db_test_list_synthetic)
db_test_real = torch.utils.data.ConcatDataset(db_test_list_real)

testLoader_synthetic = DataLoader(db_test_synthetic, batch_size=config.eval.batchSize,
                                  shuffle=False, num_workers=config.eval.numWorkers, drop_last=True)
testLoader_real = DataLoader(db_test_real, batch_size=config.eval.batchSize,
                             shuffle=False, num_workers=config.eval.numWorkers, drop_last=True)


###################### ModelBuilder #############################
model = unet.Unet(num_classes=config_checkpoint.train.numClasses)

model.load_state_dict(CHECKPOINT['model_state_dict'])

# Enable Multi-GPU training
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    model = nn.DataParallel(model)

model = model.to(device)
model.eval()

### Select Loss Func ###
if config_checkpoint.train.lossFunc == 'cosine':
    criterion = loss_fn_cosine
elif config_checkpoint.train.lossFunc == 'radians':
    criterion = loss_fn_radians
else:
    raise ValueError('Invalid lossFunc from config file. Can only be "cosine" or "radians".\
                     Value passed is: {}'.format(config_checkpoint.train.lossFunc))


### Run Validation and Test Set ###
print('\nInference - Surface Normal Estimation')
print('-' * 50 + '\n')
print('Running inference on Test sets at:\n    {}\n    {}\n'.format(config.eval.resultsDirReal,
                                                                    config.eval.resultsDirSynthetic))
print('Results will be saved to:\n    {}\n    {}\n'.format(config.eval.resultsDirReal,
                                                           config.eval.resultsDirSynthetic))

dataloaders_dict = {'real': testLoader_real, 'synthetic': testLoader_synthetic}

for key in dataloaders_dict:
    print(key + ':')
    print('=' * 30)

    testLoader = dataloaders_dict[key]
    running_loss = 0.0

    for ii, sample_batched in enumerate(testLoader):

        if key == 'real':
            inputs = sample_batched
            labels = torch.zeros(inputs.shape, dtype=torch.float)
        else:
            inputs, labels = sample_batched

        # Forward pass of the mini-batch
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            normal_vectors = model(inputs)

        normal_vectors_norm = nn.functional.normalize(normal_vectors, p=2, dim=1)
        loss = criterion(normal_vectors_norm, labels, reduction='elementwise_mean')

        running_loss += loss.item()

        if config_checkpoint.train.lossFunc == 'cosine':
            print('Image {:09d} Loss: {:.4f} (cosine loss)'.format(ii, loss.item()))
        else:
            print('Image {:09d} Loss: {:.4f} radians'.format(ii, loss.item()))

        # Save output image to results
        img_tensor = inputs[:3].detach().cpu()
        output_tensor = normal_vectors_norm[:3].detach().cpu()
        label_tensor = labels[:3].detach().cpu()

        images = []
        for img, output, label in zip(img_tensor, output_tensor, label_tensor):
            images.append(img)
            images.append(output)
            images.append(label)

        grid_image = make_grid(images, 3, normalize=True, scale_each=True)
        numpy_grid = grid_image * 255  # Scale from range [0.0, 1.0] to [0, 255]
        numpy_grid = numpy_grid.numpy().transpose(1, 2, 0).astype(np.uint8)

        if key == 'real':
            result_path = os.path.join(config.eval.resultsDirReal, 'result-%09d.jpg' % (ii))
        else:
            result_path = os.path.join(config.eval.resultsDirSynthetic, 'result-%09d.jpg' % (ii))

        imageio.imwrite(result_path, numpy_grid)

    epoch_loss = running_loss / (len(testLoader))
    print('Test Mean Loss: {:.4f}\n\n'.format(epoch_loss))
