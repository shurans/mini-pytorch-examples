'''Train unet for surface normals
'''

import os
import glob
import io

from termcolor import colored
import yaml
from attrdict import AttrDict
import imageio
import numpy as np
import h5py
from PIL import Image
import cv2
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import torch
import torch.nn as nn
import imgaug as ia
from imgaug import augmenters as iaa
from tqdm import tqdm

from models import unet
import dataloader
from utils import utils

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
DIR_RESULTS_REAL = os.path.join(config.eval.resultsDirReal, config.eval.resultsWeightsSubDir,
                                config.eval.resultsWeightsVizSubDir)
DIR_RESULTS_SYNTHETIC = os.path.join(config.eval.resultsDirSynthetic, config.eval.resultsWeightsSubDir,
                                     config.eval.resultsWeightsVizSubDir)
if not os.path.isdir(DIR_RESULTS_REAL):
    print(colored('The dir to store results "{}" does not exist. Creating dir'.format(DIR_RESULTS_REAL), 'red'))
    os.makedirs(DIR_RESULTS_REAL)
if not os.path.isdir(DIR_RESULTS_SYNTHETIC):
    print(colored('The dir to store results "{}" does not exist. Creating dir'.format(DIR_RESULTS_SYNTHETIC), 'red'))
    os.makedirs(DIR_RESULTS_SYNTHETIC)

###################### DataLoader #############################
augs_test = iaa.Sequential([
    iaa.Resize({"height": config.train.imgHeight, "width": config.train.imgWidth}, interpolation='nearest'),
])

# Make new dataloaders for each synthetic dataset
db_test_list_synthetic = []
for dataset in config.eval.datasetsSynthetic:
    if dataset.images:
        db = dataloader.SurfaceNormalsDataset(
            input_dir=dataset.images,
            label_dir=dataset.labels,
            transform=augs_test,
            input_only=None
        )
        db_test_list_synthetic.append(db)

# Make new dataloaders for each real dataset
db_test_list_real = []
for dataset in config.eval.datasetsReal:
    if dataset.images:
        db = dataloader.SurfaceNormalsRealImagesDataset(
            input_dir=dataset.images,
            transform=augs_test
        )
        db_test_list_real.append(db)

if db_test_list_synthetic:
    db_test_synthetic = torch.utils.data.ConcatDataset(db_test_list_synthetic)
    testLoader_synthetic = DataLoader(db_test_synthetic, batch_size=config.eval.batchSize,
                                      shuffle=False, num_workers=config.eval.numWorkers, drop_last=False)

if db_test_list_real:
    db_test_real = torch.utils.data.ConcatDataset(db_test_list_real)
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
criterion = nn.CrossEntropyLoss(size_average=False, reduce=True)


### Run Validation and Test Set ###
print('\nInference - Outline Prediction')
print('-' * 50 + '\n')
print('Running inference on Test sets at:\n    {}\n    {}\n'.format(config.eval.datasetsReal,
                                                                    config.eval.datasetsSynthetic))
print('Results will be saved to:\n    {}\n    {}\n'.format(config.eval.resultsDirReal,
                                                           config.eval.resultsDirSynthetic))

dataloaders_dict = {}
if db_test_list_real:
    dataloaders_dict.update({'real': testLoader_real})
if db_test_list_synthetic:
    dataloaders_dict.update({'synthetic': testLoader_synthetic})

for key in dataloaders_dict:
    print('\n' + key + ':')
    print('=' * 30)

    testLoader = dataloaders_dict[key]
    running_loss = 0.0
    total_iou = 0.0

    for ii, sample_batched in enumerate(tqdm(testLoader)):

        inputs, labels = sample_batched

        # Forward pass of the mini-batch
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)

        predictions = torch.max(outputs, 1)[1]
        loss = criterion(outputs, labels.long().squeeze(1))

        running_loss += loss.item()

        _total_iou, per_class_iou, num_images_per_class = utils.get_iou(predictions, labels.long().squeeze(1),
                                                                        n_classes=config.train.numClasses)
        total_iou += _total_iou

        # print('Batch {:09d} Loss: {:.4f}'.format(ii, loss.item()))

        # Save output images, one at a time, to results
        img_tensor = inputs.detach().cpu()
        output_tensor = outputs.detach().cpu()
        label_tensor = labels.detach().cpu()

        # Extract each tensor within batch and save results
        for iii, sample_batched in enumerate(zip(img_tensor, output_tensor, label_tensor)):
            img, output, label = sample_batched

            if key == 'real':
                RESULTS_DIR = config.eval.resultsDirReal
            else:
                RESULTS_DIR = config.eval.resultsDirSynthetic

            result_path = os.path.join(RESULTS_DIR, '{:09d}-outlines-result.png'
                                       .format(ii * config.eval.batchSize + iii))
            result_weights_path = os.path.join(RESULTS_DIR,
                                               config.eval.resultsWeightsSubDir, '{:09d}-occlusion-weight.png'
                                               .format(ii * config.eval.batchSize + iii))
            result_weights_viz_path = os.path.join(RESULTS_DIR,
                                                   config.eval.resultsWeightsSubDir,
                                                   config.eval.resultsWeightsVizSubDir,
                                                   '{:09d}-occlusion-weight.png'
                                                   .format(ii * config.eval.batchSize + iii))

            # Save Result - grid image with input, prediction and label
            output_prediction = torch.unsqueeze(torch.max(output, 0)[1].float(), 0)
            output_prediction_rgb = utils.label_to_rgb(output_prediction)
            label_rgb = utils.label_to_rgb(label)

            images = torch.cat((img, output_prediction_rgb, label_rgb), dim=2)
            grid_image = make_grid(images, 1, normalize=True, scale_each=True)
            numpy_grid = grid_image * 255  # Scale from range [0.0, 1.0] to [0, 255]
            numpy_grid = numpy_grid.numpy().transpose(1, 2, 0).astype(np.uint8)
            imageio.imwrite(result_path, numpy_grid)

            # Save the Occlusion Weights file used by depth2depth
            # calculating occlusion weights
            output_softmax = nn.Softmax(dim=1)(output).numpy()
            weight = (1 - output_softmax[1, :, :])
            x = np.power(weight, 3)
            x = np.multiply(x, 1000)
            final_weight = x.astype(np.uint16)
            # Increase the min and max values by small amount epsilon so that absolute min/max values
            # don't cause problems in the depth2depth optimization code.
            eps = 1
            final_weight[final_weight == 0] += eps
            final_weight[final_weight == 1000] -= eps
            # Save the weights file
            array_buffer = final_weight.tobytes()
            img = Image.new("I", final_weight.T.shape)
            img.frombytes(array_buffer, 'raw', 'I;16')
            img.save(result_weights_path)

            # Save the weights' visualization
            final_weight_color = (weight * 255).astype(np.uint8)
            final_weight_color = np.expand_dims(final_weight_color, axis=2)
            final_weight_color = cv2.applyColorMap(final_weight_color, cv2.COLORMAP_OCEAN)
            final_weight_color = cv2.cvtColor(final_weight_color, cv2.COLOR_BGR2RGB)
            imageio.imwrite(result_weights_viz_path, final_weight_color)

    epoch_loss = running_loss / (len(testLoader))
    print('\nTest Mean Loss: {:.4f}'.format(epoch_loss))
    miou = total_iou / ((len(testLoader)) * config.eval.batchSize)
    print('Test Mean IoU: {:.4f}'.format(miou))
