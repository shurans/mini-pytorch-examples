'''Train unet for surface normals
'''

import os
import glob
import io
import shutil

from tensorboardX import SummaryWriter
from termcolor import colored
import oyaml
from attrdict import AttrDict

from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

import models.unet_normals as unet
import dataloader
from loss_functions import loss_fn_cosine, loss_fn_radians


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

    img_tensor = inputs[:max_num_images_to_save].detach()
    output_tensor = outputs[:max_num_images_to_save].detach()
    label_tensor = labels[:max_num_images_to_save].detach()

    images = torch.cat((img_tensor, output_tensor, label_tensor), dim=3)
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
# Create a dataset object for each dataset in our list
db_trainval = []
for dataset in config.train.datasets:
    dataset = dataloader.SurfaceNormalsDataset(input_dir=dataset.images, label_dir=dataset.labels,
                                               transform=None, input_only=None)
    db_trainval.append(dataset)

# Join all the datasets into 1 large dataset
db_trainval = torch.utils.data.ConcatDataset(db_trainval)

# Split into training and validation datasets
train_size = int(config.train.percentageDataForTraining * len(db_trainval))
test_size = len(db_trainval) - train_size
db_train, db_validation = torch.utils.data.random_split(db_trainval, [train_size, test_size])

# Create dataloaders
assert (config.train.batchSize < len(db_train)), 'batchSize cannot be more than the number of images in \
                                                  training dataset'
assert (config.train.validationBatchSize < len(db_train)),'validationBatchSize cannot be more than the number of \
                                                           images in validation dataset'
trainLoader = DataLoader(db_train, batch_size=config.train.batchSize,
                         shuffle=True, num_workers=config.train.numWorkers, drop_last=True, pin_memory=True)
validationLoader = DataLoader(db_validation, batch_size=config.train.validationBatchSize, shuffle=False,
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

###################### Setup Optimizer #############################
optimizer = torch.optim.Adam(model.parameters(), lr=config.train.adamOptim.learningRate,
                             weight_decay=config.train.adamOptim.weightDecay)

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

### Select Loss Func ###
if config.train.lossFunc == 'cosine':
    criterion = loss_fn_cosine
elif config.train.lossFunc == 'radians':
    criterion = loss_fn_radians
else:
    raise ValueError('Invalid lossFunc from config file. Can only be "cosine" or "radians".\
                     Value passed is: {}'.format(config.train.lossFunc))


###################### Train Model #############################
# Set total iter_num (number of batches seen by model, used for logging)
if (config.train.continueTraining and config.train.loadEpochNumberFromCheckpoint):
    if 'model_state_dict' in CHECKPOINT:
        # TODO: remove this second check for 'model_state_dict' soon. Kept for ensuring backcompatibility
        total_iter_num = CHECKPOINT['total_iter_num'] + 1
        START_EPOCH = CHECKPOINT['epoch'] + 1
        END_EPOCH = CHECKPOINT['epoch'] + config.train.numEpochs
    else:
        print(colored('Could not load epoch and total iter nums from checkpoint, they do not exist in checkpoint',
                      'red'))
        total_iter_num = 0
        START_EPOCH = 0
        END_EPOCH = config.train.numEpochs

for epoch in range(START_EPOCH, END_EPOCH):
    print('Epoch {}/{}'.format(epoch, END_EPOCH - 1))
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
        normal_vectors = model(inputs)
        normal_vectors_norm = nn.functional.normalize(normal_vectors, p=2, dim=1)

        loss = criterion(normal_vectors_norm, labels, reduction='elementwise_mean')
        loss.backward()
        optimizer.step()

        # statistics
        running_loss += loss.item()
        writer.add_scalar('Train BatchWise Loss', loss.item(), total_iter_num)

        # Print loss every 20 Batches
        if (iter_num % 20) == 0:
            if config.train.lossFunc == 'cosine':
                print('Epoch{} Batch{} BatchLoss: {:.4f} (cosine loss)'.format(epoch, iter_num, loss.item()))
            else:
                print('Epoch{} Batch{} BatchLoss: {:.4f} radians'.format(epoch, iter_num, loss.item()))

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
        grid_image = create_grid_image(inputs, normal_vectors_norm, labels, max_num_images_to_save=3)
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
            normal_vectors = model(inputs)

        normal_vectors_norm = nn.functional.normalize(normal_vectors, p=2, dim=1)
        loss = criterion(normal_vectors_norm, labels, reduction='elementwise_mean')

        running_loss += loss.item()

        # Pring loss every 20 Batches
        if (iter_num % 20) == 0:
            if config.train.lossFunc == 'cosine':
                print('Epoch{} Batch{} BatchLoss: {:.4f} (cosine loss)'.format(epoch, iter_num, loss.item()))
            else:
                print('Epoch{} Batch{} BatchLoss: {:.4f} radians'.format(epoch, iter_num, loss.item()))

    # Log Epoch Loss
    epoch_loss = running_loss / (len(validationLoader))
    writer.add_scalar('Validation Epoch Loss', epoch_loss, total_iter_num)
    print('\nValidation Epoch Loss: {:.4f}\n\n'.format(epoch_loss))

    # Log 10 images every N epochs
    if (epoch % config.train.saveImageInterval) == 0:
        grid_image = create_grid_image(inputs, normal_vectors_norm, labels, max_num_images_to_save=10)
        writer.add_image('Validation', grid_image, total_iter_num)

writer.close()
