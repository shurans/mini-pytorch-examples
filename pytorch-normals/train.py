'''Train unet for surface normals
'''

import os
import glob
import io
import shutil

from tqdm import tqdm
from tensorboardX import SummaryWriter
from termcolor import colored
import oyaml
from attrdict import AttrDict
import numpy as np
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import imgaug as ia
from imgaug import augmenters as iaa

from models import unet_normals as unet
from models import deeplab_xception, deeplab_resnet
import dataloader
from loss_functions import loss_fn_cosine, loss_fn_radians, cross_entropy2d
from utils import utils


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

    # Blur and Noise
    # iaa.Sometimes(0.2, iaa.GaussianBlur(sigma=(0.25, 1.20), name="gaus-blur")),
    # iaa.Sometimes(0.2, iaa.AdditiveLaplaceNoise(scale=(0, 0.075 * 255), per_channel=True, name="gaus-noise")),
])
input_only = ["gaus-blur", "gaus-noise"]

db_train_list = []
for dataset in config.train.datasetsTrain:
    db = dataloader.SurfaceNormalsDataset(input_dir=dataset.images, label_dir=dataset.labels,
                                          transform=augs_train, input_only=input_only)
    train_size = int(config.train.percentageDataForTraining * len(db))
    db = torch.utils.data.Subset(db, range(train_size))
    db_train_list.append(db)

db_train = torch.utils.data.ConcatDataset(db_train_list)


# Validation Dataset
augs_test = iaa.Sequential([
    iaa.Resize({"height": config.train.imgHeight, "width": config.train.imgWidth}, interpolation='nearest'),
])

db_val_list = []
for dataset in config.train.datasetsVal:
    if dataset.images:
        db = dataloader.SurfaceNormalsDataset(input_dir=dataset.images, label_dir=dataset.labels,
                                              transform=augs_test, input_only=None)
        train_size = int(config.train.percentageDataForValidation * len(db))
        db = torch.utils.data.Subset(db, range(train_size))
        db_val_list.append(db)

if db_val_list:
    db_val = torch.utils.data.ConcatDataset(db_val_list)

# Test Dataset - Real
db_test_list = []
for dataset in config.train.datasetsTestReal:
    if dataset.images:
        db = dataloader.SurfaceNormalsDataset(input_dir=dataset.images, label_dir=dataset.labels,
                                              transform=augs_test, input_only=None)
        db_test_list.append(db)
if db_test_list:
    db_test = torch.utils.data.ConcatDataset(db_test_list)

# Test Dataset - Synthetic
db_test_synthetic_list = []
print('config.train.datasetsTestSynthetic', config.train.datasetsTestSynthetic)
for dataset in config.train.datasetsTestSynthetic:
    if dataset.images:
        db = dataloader.SurfaceNormalsDataset(input_dir=dataset.images, label_dir=dataset.labels,
                                              transform=augs_test, input_only=None)
        db_test_synthetic_list.append(db)
if db_test_synthetic_list:
    db_test_synthetic = torch.utils.data.ConcatDataset(db_test_synthetic_list)


# Create dataloaders
# NOTE: Calculation of statistics like epoch_loss depend on the param drop_last being True. They calculate total num
#       of images as num of batches * batchSize, which is true only when drop_last=True.
assert (config.train.batchSize <= len(db_train)), \
    ('batchSize ({}) cannot be more than ' +
     'the number of images in training dataset ({})').format(config.train.batchSize, len(db_train))

trainLoader = DataLoader(db_train, batch_size=config.train.batchSize,
                         shuffle=True, num_workers=config.train.numWorkers, drop_last=True, pin_memory=True)
if db_val_list:
    assert (config.train.validationBatchSize <= len(db_val)), \
        ('validationBatchSize ({}) cannot be more than the ' +
         'number of images in validation dataset: {}').format(config.train.validationBatchSize, len(db_val))

    validationLoader = DataLoader(db_val, batch_size=config.train.validationBatchSize, shuffle=True,
                                  num_workers=config.train.numWorkers, drop_last=False)
if db_test_list:
    assert (config.train.testBatchSize <= len(db_test)), \
        ('testBatchSize ({}) cannot be more than the ' +
         'number of images in test dataset: {}').format(config.train.testBatchSize, len(db_test))

    testLoader = DataLoader(db_test, batch_size=config.train.testBatchSize, shuffle=False,
                            num_workers=config.train.numWorkers, drop_last=False)
if db_test_synthetic_list:
    assert (config.train.testBatchSize <= len(db_test_synthetic)), \
        ('testBatchSize ({}) cannot be more than the ' +
         'number of images in test dataset: {}').format(config.train.testBatchSize, len(db_test_synthetic_list))

    testSyntheticLoader = DataLoader(db_test_synthetic, batch_size=config.train.testBatchSize, shuffle=True,
                                     num_workers=config.train.numWorkers, drop_last=False)

###################### ModelBuilder #############################
if config.train.model == 'unet':
    model = unet.Unet(num_classes=config.train.numClasses)
elif config.train.model == 'deeplab_xception':
    model = deeplab_xception.DeepLabv3_plus(n_classes=config.train.numClasses, os=config.train.outputStride,
                                            nInputChannels=config.train.numInputChannels, pretrained=True)
elif config.train.model == 'deeplab_resnet':
    model = deeplab_resnet.DeepLabv3_plus(n_classes=config.train.numClasses, os=config.train.outputStride,
                                          nInputChannels=config.train.numInputChannels, pretrained=True)
else:
    raise ValueError('Invalid model "{}" in config file. Must be one of ["unet", "deeplab_xception", "deeplab_resnet"]'
                     .format(config.train.model))

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

###################### Setup Optimizer #############################
if config.train.model == 'unet':
    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.optimAdam.learningRate,
                                 weight_decay=config.train.optimAdam.weightDecay)
elif config.train.model == 'deeplab_xception' or config.train.model == 'deeplab_resnet':
    optimizer = torch.optim.SGD(model.parameters(), lr=float(config.train.optimSgd.learningRate),
                                momentum=config.train.optimSgd.momentum,
                                weight_decay=float(config.train.optimSgd.weight_decay))

if not config.train.lrScheduler:
    pass
elif config.train.lrScheduler == 'StepLR':
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=config.train.lrSchedulerStep.step_size,
                                                   gamma=config.train.lrSchedulerStep.gamma)
elif config.train.lrScheduler == 'ReduceLROnPlateau':
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                              factor=config.train.lrSchedulerPlateau.factor,
                                                              patience=config.train.lrSchedulerPlateau.patience,
                                                              verbose=True)
elif config.train.lrScheduler == 'lr_poly':
    print('Using Polynomial Learning Rate scheduler')
else:
    raise ValueError("Invalid Scheduler from config file: '{}'. Valid values are ['', 'StepLR', 'ReduceLROnPlateau']"
                     .format(config.train.lrScheduler))

# Continue Training from prev checkpoint if required
if config.train.continueTraining and config.train.initOptimizerFromCheckpoint:
    if 'optimizer_state_dict' in CHECKPOINT:
        optimizer.load_state_dict(CHECKPOINT['optimizer_state_dict'])
    else:
        print(colored('WARNING: Could not load optimizer state from checkpoint as checkpoint does not contain ' +
                      '"optimizer_state_dict". Continuing without loading optimizer state. ', 'red'))

### Select Loss Func ###
if config.train.lossFunc == 'cosine':
    criterion = loss_fn_cosine
elif config.train.lossFunc == 'radians':
    criterion = loss_fn_radians
else:
    raise ValueError("Invalid lossFunc from config file. Can only be ['cosine', 'radians']. " +
                     "Value passed is: {}".format(config.train.lossFunc))


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
        print(colored('Could not load epoch and total iter nums from checkpoint, they do not exist in checkpoint.\
                       Starting from epoch num 0', 'red'))

for epoch in range(START_EPOCH, END_EPOCH):
    print('\n\nEpoch {}/{}'.format(epoch, END_EPOCH - 1))
    print('-' * 30)

    # Log the current Epoch Number
    writer.add_scalar('data/Epoch Number', epoch, total_iter_num)

    ###################### Training Cycle #############################
    print('Train:')
    print('=' * 10)

    # Update Learning Rate Scheduler
    if config.train.lrScheduler == 'StepLR':
        lr_scheduler.step()
    elif config.train.lrScheduler == 'ReduceLROnPlateau':
        lr_scheduler.step(epoch_loss)
    elif config.train.lrScheduler == 'lr_poly':
        if epoch % config.train.epochSize == config.train.epochSize - 1:
            lr_ = utils.lr_poly(config.train.optimSgd.learningRate, epoch - START_EPOCH, END_EPOCH - START_EPOCH, 0.9)
            # optimizer = optim.SGD(net.parameters(), lr=lr_, momentum=p['momentum'], weight_decay=p['wd'])
            optimizer = torch.optim.SGD(model.parameters(), lr=config.train.optimSgd.learningRate,
                                        momentum=config.train.optimSgd.momentum,
                                        weight_decay=config.train.optimSgd.weight_decay)

    model.train()

    running_loss = 0.0
    for iter_num, batch in enumerate(tqdm(trainLoader)):
        total_iter_num += 1

        # Get data
        inputs, labels = batch
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Forward + Backward Prop
        optimizer.zero_grad()
        torch.set_grad_enabled(True)
        normal_vectors = model(inputs)
        normal_vectors_norm = nn.functional.normalize(normal_vectors, p=2, dim=1)

        if config.train.model == 'unet':
            loss = criterion(normal_vectors_norm, labels, reduction='sum')
        elif config.train.model == 'deeplab_xception' or config.train.model == 'deeplab_resnet':
            loss = criterion(normal_vectors_norm, labels, reduction='sum')
            loss /= config.train.batchSize
        loss.backward()
        optimizer.step()

        # statistics
        running_loss += loss.item()
        writer.add_scalar('data/Train BatchWise Loss', loss.item(), total_iter_num)

        # # Print loss every 20 Batches
        # if (iter_num % 20) == 0:
        #     if config.train.lossFunc == 'cosine':
        #         print('Epoch{} Batch{} BatchLoss: {:.4f} (cosine loss)'.format(epoch, iter_num, loss.item()))
        #     else:
        #         print('Epoch{} Batch{} BatchLoss: {:.4f} radians'.format(epoch, iter_num, loss.item()))

    # Log Epoch Loss
    epoch_loss = running_loss / (len(trainLoader))
    writer.add_scalar('data/Train Epoch Loss', epoch_loss, total_iter_num)
    print('Train Epoch Loss: {:.4f}'.format(epoch_loss))

    # Log Current Learning Rate
    # TODO: NOTE: The lr of adam is not directly accessible. Adam creates a loss for every parameter in model.
    #    The value read here will only reflect the initial lr value.
    current_learning_rate = optimizer.param_groups[0]['lr']
    writer.add_scalar('Learning Rate', current_learning_rate, total_iter_num)

    # Log 3 images every N epochs
    if (epoch % config.train.saveImageInterval) == 0:
        grid_image = utils.create_grid_image(inputs.detach().cpu(), normal_vectors_norm.detach().cpu(),
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
    print('\nValidation:')
    print('=' * 10)

    model.eval()

    running_loss = 0.0
    for iter_num, sample_batched in enumerate(tqdm(validationLoader)):
        inputs, labels = sample_batched

        # Forward pass of the mini-batch
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            normal_vectors = model(inputs)

        normal_vectors_norm = nn.functional.normalize(normal_vectors, p=2, dim=1)
        if config.train.model == 'unet':
            loss = criterion(normal_vectors_norm, labels, reduction='sum')
        elif config.train.model == 'deeplab_xception' or config.train.model == 'deeplab_resnet':
            loss = criterion(normal_vectors_norm, labels, reduction='sum')
            loss /= config.train.batchSize

        running_loss += loss.item()

        # # Pring loss every 20 Batches
        # if (iter_num % 20) == 0:
        #     if config.train.lossFunc == 'cosine':
        #         print('Epoch{} Batch{} BatchLoss: {:.4f} (cosine loss)'.format(epoch, iter_num, loss.item()))
        #     else:
        #         print('Epoch{} Batch{} BatchLoss: {:.4f} radians'.format(epoch, iter_num, loss.item()))

    # Log Epoch Loss
    epoch_loss = running_loss / (len(validationLoader))
    writer.add_scalar('data/Validation Epoch Loss', epoch_loss, total_iter_num)
    print('Validation Epoch Loss: {:.4f}'.format(epoch_loss))

    # Log 10 images every N epochs
    if (epoch % config.train.saveImageInterval) == 0:
        grid_image = utils.create_grid_image(inputs.detach().cpu(), normal_vectors_norm.detach().cpu(),
                                             labels.detach().cpu(), max_num_images_to_save=10)
        writer.add_image('Validation', grid_image, total_iter_num)

    ###################### Test Cycle - Real #############################
    if db_test_list:
        print('\nTesting:')
        print('=' * 10)

        model.eval()

        for iter_num, sample_batched in enumerate(tqdm(testLoader)):
            inputs, labels = sample_batched

            # Forward pass of the mini-batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                normal_vectors = model(inputs)

            normal_vectors_norm = nn.functional.normalize(normal_vectors, p=2, dim=1)

        # Log 30 images every N epochs
        if (epoch % config.train.saveImageInterval) == 0:
            grid_image = utils.create_grid_image(inputs.detach().cpu(), normal_vectors_norm.detach().cpu(),
                                                 labels.detach().cpu(), max_num_images_to_save=30)
            writer.add_image('Testing', grid_image, total_iter_num)

    ###################### Test Cycle - Synthetic #############################
    if db_test_synthetic_list:
        print('\nTest Synthetic:')
        print('=' * 10)

        model.eval()

        running_loss = 0.0
        for iter_num, sample_batched in enumerate(tqdm(testSyntheticLoader)):
            inputs, labels = sample_batched

            # Forward pass of the mini-batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                normal_vectors = model(inputs)

            normal_vectors_norm = nn.functional.normalize(normal_vectors, p=2, dim=1)
            if config.train.model == 'unet':
                loss = criterion(normal_vectors_norm, labels, reduction='sum')
            elif config.train.model == 'deeplab_xception' or config.train.model == 'deeplab_resnet':
                loss = criterion(normal_vectors_norm, labels, reduction='sum')
                loss /= config.train.batchSize

            running_loss += loss.item()

        # Log Epoch Loss
        epoch_loss = running_loss / (len(testSyntheticLoader))
        writer.add_scalar('data/Test Synthetic Epoch Loss', epoch_loss, total_iter_num)
        print('\Test Synthetic Epoch Loss: {:.4f}'.format(epoch_loss))

        # Log 30 images every N epochs
        if (epoch % config.train.saveImageInterval) == 0:
            grid_image = utils.create_grid_image(inputs.detach().cpu(), normal_vectors_norm.detach().cpu(),
                                                 labels.detach().cpu(), max_num_images_to_save=10)
            writer.add_image('Test Synthetic', grid_image, total_iter_num)

writer.close()
