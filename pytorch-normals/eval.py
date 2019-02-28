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
import h5py

from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

import models.unet_normals as unet
import dataloader
from loss_functions import loss_fn_cosine, loss_fn_radians

print('Inference of Surface Normal Estimation model. Loading checkpoint...')

###################### Load Config File #############################
CONFIG_FILE_PATH = 'config/config.yaml'
with open(CONFIG_FILE_PATH) as fd:
    config_yaml = yaml.safe_load(fd)
config = AttrDict(config_yaml)

###################### Load Checkpoint and its data #############################
if not os.path.isfile(config.eval.pathWeightsFile):
    raise ValueError('Invalid path to the given weights file in config. The file "{}" does not exist'.format(
        config.eval.pathWeightsFile))

# Read config file stored in the model checkpoint to re-use it's params
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

# Check for results store dir
DIR_RESULTS_REAL = os.path.join(config.eval.resultsDirReal, config.eval.resultsHdf5SubDir)
DIR_RESULTS_SYNTHETIC = os.path.join(config.eval.resultsDirSynthetic, config.eval.resultsHdf5SubDir)
if not os.path.isdir(DIR_RESULTS_REAL):
    print(colored('The dir to store results "{}" does not exist. Creating dir'.format(DIR_RESULTS_REAL), 'red'))
    os.makedirs(DIR_RESULTS_REAL)
if not os.path.isdir(DIR_RESULTS_SYNTHETIC):
    print(colored('The dir to store results "{}" does not exist. Creating dir'.format(DIR_RESULTS_SYNTHETIC), 'red'))
    os.makedirs(DIR_RESULTS_SYNTHETIC)

###################### DataLoader #############################
# Make new dataloaders for each synthetic dataset
db_test_list_synthetic = []
for dataset in config.eval.datasetsSynthetic:
    dataset = dataloader.SurfaceNormalsDataset(
        input_dir=dataset.images,
        label_dir=dataset.labels,
        transform=None,
        input_only=None
    )
    db_test_list_synthetic.append(dataset)

# Make new dataloaders for each real dataset
db_test_list_real = []
for dataset in config.eval.datasetsReal:
    dataset = dataloader.SurfaceNormalsRealImagesDataset(
        input_dir=dataset.images,
        imgHeight=config_checkpoint.train.imgHeight,
        imgWidth=config_checkpoint.train.imgWidth
    )
    db_test_list_real.append(dataset)


db_test_synthetic = torch.utils.data.ConcatDataset(db_test_list_synthetic)
db_test_real = torch.utils.data.ConcatDataset(db_test_list_real)

testLoader_synthetic = DataLoader(db_test_synthetic, batch_size=config.eval.batchSize,
                                  shuffle=False, num_workers=config.eval.numWorkers, drop_last=False)
testLoader_real = DataLoader(db_test_real, batch_size=config.eval.batchSize,
                             shuffle=False, num_workers=config.eval.numWorkers, drop_last=False)


###################### ModelBuilder #############################
model = unet.Unet(num_classes=config_checkpoint.train.numClasses)

model.load_state_dict(CHECKPOINT['model_state_dict'])

# Enable Multi-GPU training
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    model = nn.DataParallel(model)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
print('Running inference on Test sets at:\n    {}\n    {}\n'.format(config.eval.datasetsReal,
                                                                    config.eval.datasetsSynthetic))
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
            print('Batch {:09d} Loss: {:.4f} (cosine loss)'.format(ii, loss.item()))
        else:
            print('Batch {:09d} Loss: {:.4f} radians'.format(ii, loss.item()))

        # Save output images, one at a time, to results
        img_tensor = inputs.detach().cpu()
        output_tensor = normal_vectors_norm.detach().cpu()
        label_tensor = labels.detach().cpu()

        # Extract each tensor within batch
        for iii, sample_batched in enumerate(zip(img_tensor, output_tensor, label_tensor)):
            img, output, label = sample_batched

            grid_image = make_grid([img, output, label], 3, normalize=True, scale_each=True)
            numpy_grid = grid_image * 255  # Scale from range [0.0, 1.0] to [0, 255]
            numpy_grid = numpy_grid.numpy().transpose(1, 2, 0).astype(np.uint8)

            if key == 'real':
                result_path = os.path.join(config.eval.resultsDirReal, '{:09d}-normals.jpg'
                                           .format(ii * config.eval.batchSize + iii))
                result_hdf5_path = os.path.join(config.eval.resultsDirReal,
                                                config.eval.resultsHdf5SubDir, '{:09d}-normals.h5'
                                                .format(ii * config.eval.batchSize + iii))
            else:
                result_path = os.path.join(config.eval.resultsDirSynthetic, '{:09d}-normals.jpg'
                                           .format(ii * config.eval.batchSize + iii))
                result_hdf5_path = os.path.join(config.eval.resultsDirSynthetic,
                                                config.eval.resultsHdf5SubDir, '{:09d}-normals.h5'
                                                .format(ii * config.eval.batchSize + iii))

            # Save grid image with input, prediction and label
            imageio.imwrite(result_path, numpy_grid)

            # Write Predicted Surface Normal as hdf5 file for depth2depth
            # NOTE: The hdf5 expected shape is (3, height, width), float32
            with h5py.File(result_hdf5_path, "w") as f:
                dset2 = f.create_dataset('/result', data=output.numpy())

    epoch_loss = running_loss / (len(testLoader))
    print('Test Mean Loss: {:.4f}\n\n'.format(epoch_loss))
