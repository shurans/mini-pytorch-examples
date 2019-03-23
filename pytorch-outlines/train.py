'''Train unet for surface normals
'''

import os
import glob
import io
import shutil
from multiprocessing import Process

from tensorboardX import SummaryWriter
from termcolor import colored
import oyaml
from attrdict import AttrDict
import numpy as np
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import imgaug as ia
from imgaug import augmenters as iaa

from models import unet
import dataloader
from loss_functions import cross_entropy2d


def label_to_rgb(label):
    '''Output RGB visualizations of the outlines' labels

    The labels of outlines have 3 classes: Background, Depth Outlines, Surface Normal Outlines which are mapped to
    Red, Green and Blue respectively.

    Args:
        label (torch.Tensor): Shape: (no. of images, 1, height, width). Each pixel contains an int with value of class.

    Returns:
        torch.Tensor: Shape (no. of images, 3, height, width): RGB representation of the labels
    '''
    rgbArray = torch.zeros((label.shape[0], 3, label.shape[2], label.shape[3]), dtype=torch.float)
    rgbArray[:, 0, :, :][label[:, 0, :, :] == 0] = 1
    rgbArray[:, 1, :, :][label[:, 0, :, :] == 1] = 1
    rgbArray[:, 2, :, :][label[:, 0, :, :] == 2] = 1

    return rgbArray


def create_grid_image(inputs, outputs, labels, max_num_images_to_save=3):
    '''Make a grid of images for display purposes
    Size of grid is (3, N, 3), where each coloum belongs to input, output, label resp

    Args:
        inputs (Tensor): Batch Tensor of shape (B x C x H x W)
        outputs (Tensor): Batch Tensor of shape (B x C x H x W)
        labels (Tensor): Batch Tensor of shape (B x C x H x W)
        max_num_images_to_save (int, optional): Defaults to 3. Out of the given tensors, chooses a
            max number of imaged to put in grid

    Returns:
        numpy.ndarray: A numpy array with of input images arranged in a grid
    '''

    img_tensor = inputs[:max_num_images_to_save]
    output_tensor = torch.unsqueeze(torch.max(outputs[:max_num_images_to_save], 1)[1].float(), 1)
    output_tensor_rgb = label_to_rgb(output_tensor)
    label_tensor = labels[:max_num_images_to_save]
    label_tensor_rgb = label_to_rgb(label_tensor)

    images = torch.cat((img_tensor, output_tensor_rgb, label_tensor_rgb), dim=3)
    grid_image = make_grid(images, 1, normalize=True, scale_each=True)

    return grid_image


###################### Load Config File #############################
CONFIG_FILE_PATH = 'config/config.yaml'
with open(CONFIG_FILE_PATH) as fd:
    config_yaml = oyaml.load(fd)  # Returns an ordered dict. Used for printing

config = AttrDict(config_yaml)
print(colored('Config being used for training:\n{}\n\n'.format(oyaml.dump(config_yaml)), 'green'))

###################### Logs (TensorBoard)  #############################
# Create a new directory to save logs
runs = sorted(glob.glob(os.path.join(config.train.logsDir, 'exp-*')))
prev_run_id = int(runs[-1].split('-')[-1]) if runs else 0
MODEL_LOG_DIR = os.path.join(config.train.logsDir, 'exp-{:03d}'.format(prev_run_id + 1))
CHECKPOINT_DIR = os.path.join(MODEL_LOG_DIR, 'checkpoints')
os.makedirs(CHECKPOINT_DIR)
print('Saving logs to folder: ' + colored('"{}"'.format(MODEL_LOG_DIR), 'blue'))

# Save a copy of config file in the logs
shutil.copy(CONFIG_FILE_PATH, os.path.join(MODEL_LOG_DIR, 'config.yaml'))

# Create a tensorboard object and Write config to tensorboard
writer = SummaryWriter(MODEL_LOG_DIR, comment='create-graph')

string_out = io.StringIO()
oyaml.dump(config_yaml, string_out, default_flow_style=False)
config_str = string_out.getvalue().split('\n')
string = ''
for line in config_str:
    string = string + '    ' + line + '\n\r'
writer.add_text('Config', string, global_step=None)

###################### DataLoader #############################
# Train Dataset - Create a dataset object for each dataset in our list, Concatenate datasets, select subset for training
augs_train = iaa.Sequential([
    # Geometric Augs
    iaa.Resize({"height": config.train.imgHeight, "width": config.train.imgWidth}, interpolation='nearest'),
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5),
])

db_train_list = []
for dataset in config.train.datasets:
    db = dataloader.SurfaceNormalsDataset(input_dir=dataset.images, label_dir=dataset.labels,
                                          transform=None, input_only=None)
    db_train_list.append(db)

db_train = torch.utils.data.ConcatDataset(db_train_list)
train_size = int(config.train.percentageDataForTraining * len(db_train))
db_train = torch.utils.data.Subset(db_train, range(train_size))

# Validation Dataset
print('config imsize', int(config.train.imgHeight), int(config.train.imgWidth))
augs_test = iaa.Sequential([
    iaa.Resize({"height": config.train.imgHeight, "width": config.train.imgWidth}, interpolation='nearest'),
])

db_val_list = []
for dataset in config.eval.datasetsSynthetic:
    if dataset.images:
        db = dataloader.SurfaceNormalsDataset(input_dir=dataset.images, label_dir=dataset.labels,
                                              transform=None, input_only=None)
        db_val_list.append(db)

if db_val_list:
    db_val = torch.utils.data.ConcatDataset(db_val_list)
    train_size = int(config.train.percentageDataForValidation * len(db_val))
    db_val = torch.utils.data.Subset(db_val, range(train_size))

# Test Dataset
db_test_list = []
for dataset in config.eval.datasetsReal:
    if dataset.images:
        db = dataloader.SurfaceNormalsRealImagesDataset(input_dir=dataset.images, transform=augs_test)
        db_test_list.append(db)
if db_test_list:
    db_test = torch.utils.data.ConcatDataset(db_test_list)


# Create dataloaders
assert (config.train.batchSize < len(db_train)), 'batchSize cannot be more than the number of images in \
                                                  training dataset'
assert (config.train.validationBatchSize < len(db_train)), 'validationBatchSize cannot be more than the number of \
                                                           images in validation dataset'
trainLoader = DataLoader(db_train, batch_size=config.train.batchSize,
                         shuffle=True, num_workers=config.train.numWorkers, drop_last=True, pin_memory=True)
if db_val_list:
    validationLoader = DataLoader(db_val, batch_size=config.train.validationBatchSize, shuffle=False,
                                  num_workers=config.train.numWorkers, drop_last=True, pin_memory=True)
if db_test_list:
    testLoader = DataLoader(db_test, batch_size=config.train.validationBatchSize, shuffle=False,
                            num_workers=config.train.numWorkers, drop_last=True, pin_memory=True)

###################### ModelBuilder #############################
if config.train.model == 'unet':
    model = unet.Unet(num_classes=config.train.numClasses)
else:
    raise ValueError('Invalid model "{}" in config file. Must be one of ["unet"]'.format(config.train.model))

if config.train.continueTraining:
    print('Transfer Learning enabled. Model State to be loaded from a prev checkpoint...')
    if not os.path.isfile(config.train.pathPrevCheckpoint):
        raise ValueError('Invalid path to the given weights file for transfer learning.\
                The file {} does not exist'.format(config.train.pathPrevCheckpoint))

    CHECKPOINT = torch.load(config.train.pathPrevCheckpoint, map_location='cpu')

    if 'model_state_dict' in CHECKPOINT:
        # Newer weights file with various dicts
        print(colored('Continuing training from checkpoint...Loaded data from checkpoint:', 'green'))
        print('Config Used to train Checkpoint:\n', oyaml.dump(CHECKPOINT['config']), '\n')
        print('From Checkpoint: Last Epoch Loss:', CHECKPOINT['epoch_loss'], '\n\n')

        model.load_state_dict(CHECKPOINT['model_state_dict'])
    else:
        # Old checkpoint containing only model's state_dict()
        model.load_state_dict(CHECKPOINT)

# Enable Multi-GPU training
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    model = nn.DataParallel(model)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

## Loss function ##
criterion = nn.CrossEntropyLoss(size_average=False, reduce=True)

###################### Setup Optimizer #############################
optimizer = torch.optim.Adam(model.parameters(), lr=config.train.optimAdam.learningRate,
                             weight_decay=config.train.optimAdam.weightDecay)

if not config.train.lrScheduler:
    pass
elif not config.train.lrScheduler and config.train.lrScheduler == 'StepLR':
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=config.train.lrSchedulerStep.step_size,
                                                   gamma=config.train.lrSchedulerStep.gamma)
elif not config.train.lrScheduler and config.train.lrScheduler == 'ReduceLROnPlateau':
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                              factor=config.train.lrSchedulerPlateau.factor,
                                                              patience=config.train.lrSchedulerPlateau.patience,
                                                              verbose=True)
else:
    raise ValueError('Invalid Scheduler from config file: "{}". Valid values are ["", "StepLR", "ReduceLROnPlateau"'
                     .format(config.train.lrScheduler))

# Continue Training from prev checkpoint if required
if config.train.continueTraining and config.train.initOptimizerFromCheckpoint:
    if 'optimizer_state_dict' in CHECKPOINT:
        optimizer.load_state_dict(CHECKPOINT['optimizer_state_dict'])
    else:
        print(colored('Could not load optimizer state from checkpoint, it does not contain "optimizer_state_dict" ',
                      'red'))


###################### Train Model #############################
# Set total iter_num (number of batches seen by model, used for logging)

total_iter_num = 0
START_EPOCH = 0
END_EPOCH = config.train.numEpochs
if (config.train.continueTraining and config.train.loadEpochNumberFromCheckpoint):
    if 'model_state_dict' in CHECKPOINT:
        # TODO: remove this second check for 'model_state_dict' soon. Kept for ensuring backcompatibility
        total_iter_num = CHECKPOINT['total_iter_num'] + 1
        START_EPOCH = CHECKPOINT['epoch'] + 1
        END_EPOCH = CHECKPOINT['epoch'] + config.train.numEpochs
    else:
        print(colored('Could not load epoch and total iter nums from checkpoint, they do not exist in checkpoint',
                      'red'))


for epoch in range(START_EPOCH, END_EPOCH):
    print('\n\nEpoch {}/{}'.format(epoch, END_EPOCH - 1))
    print('-' * 30)

    # Log the current Epoch Number
    writer.add_scalar('Epoch Number', epoch, total_iter_num)

    ###################### Training Cycle #############################
    print('Train:')
    print('=' * 10)

    model.train()

    running_loss = 0.0
    for iter_num, batch in enumerate(trainLoader):
        total_iter_num += 1

        # Get data
        inputs, labels = batch
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Forward + Backward Prop
        optimizer.zero_grad()
        torch.set_grad_enabled(True)
        outputs = model.forward(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # statistics
        running_loss += loss.item()
        writer.add_scalar('Train BatchWise Loss', loss.item(), total_iter_num)

        # Print loss every 20 Batches
        if (iter_num % 20) == 0:
            print('Epoch{} Batch{} BatchLoss: {:.4f} '.format(epoch, iter_num, loss.item()))

    # Log Epoch Loss
    epoch_loss = running_loss / (len(trainLoader))
    writer.add_scalar('Train Epoch Loss', epoch_loss, total_iter_num)
    print('\nTrain Epoch Loss: {:.4f}\n'.format(epoch_loss))

    # Update Learning Rate Scheduler
    if config.train.lrScheduler == 'StepLR':
        lr_scheduler.step()
    elif config.train.lrScheduler == 'ReduceLROnPlateau':
        lr_scheduler.step(epoch_loss)

    # Log Current Learning Rate
    # TODO: NOTE: The lr of adam is not directly accessible. Adam creates a loss for every parameter in model.
    #    The value read here will only reflect the initial lr value.
    current_learning_rate = optimizer.param_groups[0]['lr']
    writer.add_scalar('Learning Rate', current_learning_rate, total_iter_num)

    # Log 3 images every N epochs
    if (epoch % config.train.saveImageInterval) == 0:
        grid_image = create_grid_image(inputs.detach().cpu(), outputs.detach().cpu(),
                                       labels.detach().cpu(), max_num_images_to_save=3)
        writer.add_image('Train', grid_image, total_iter_num)

    # Save the model checkpoint every N epochs
    if (epoch % config.train.saveModelInterval) == 0:
        filename = os.path.join(CHECKPOINT_DIR, 'checkpoint-epoch-{:04d}.pth'.format(epoch))
        if torch.cuda.device_count() > 1:
            model_params = model.module.state_dict()  # Saving nn.DataParallel model
        else:
            model_params = model.state_dict()

        torch.save({
            'model_state_dict': model_params,
            'optimizer_state_dict': optimizer.state_dict(),

            'epoch': epoch,
            'total_iter_num': total_iter_num,

            'epoch_loss': epoch_loss,
            'config': config_yaml
        }, filename)

    ###################### Validation Cycle #############################
    print('Validation:')
    print('=' * 10)

    model.eval()

    running_loss = 0.0
    for iter_num, sample_batched in enumerate(validationLoader):
        inputs, labels = sample_batched

        # Forward pass of the mini-batch
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)

        loss = criterion(outputs, labels)

        running_loss += loss.item()

        # Pring loss every 20 Batches
        if (iter_num % 20) == 0:
            print('Epoch{} Batch{} BatchLoss: {:.4f} '.format(epoch, iter_num, loss.item()))

    # Log Epoch Loss
    epoch_loss = running_loss / (len(validationLoader))
    writer.add_scalar('Validation Epoch Loss', epoch_loss, total_iter_num)
    print('\nValidation Epoch Loss: {:.4f}\n\n'.format(epoch_loss))

    # Log 10 images every N epochs
    if (epoch % config.train.saveImageInterval) == 0:
        grid_image = create_grid_image(inputs.detach().cpu(), outputs.detach().cpu(),
                                       labels.detach().cpu(), max_num_images_to_save=10)
        writer.add_image('Validation', grid_image, total_iter_num)

    ###################### Test Cycle #############################
    if db_test_list:
        print('Testing:')
        print('=' * 10)

        model.eval()

        running_loss = 0.0
        for iter_num, sample_batched in enumerate(testLoader):
            inputs, labels = sample_batched

            # Forward pass of the mini-batch
            inputs = inputs.to(device)

            with torch.no_grad():
                outputs = model(inputs)

            # Pring loss every 1 Batche
            if (iter_num % 1) == 0:
                print('Test Epoch{} Batch{} '.format(epoch, iter_num))

        # Log 30 images every N epochs
        if (epoch % config.train.saveImageInterval) == 0:
            grid_image = create_grid_image(inputs.detach().cpu(), outputs.detach().cpu(),
                                           labels.detach().cpu(), max_num_images_to_save=30)
            writer.add_image('Testing', grid_image, total_iter_num)

writer.close()
