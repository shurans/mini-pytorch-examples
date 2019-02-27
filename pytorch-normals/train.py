'''Train unet for surface normals
'''

import os
import glob
import io

from tensorboardX import SummaryWriter
from termcolor import colored
import yaml
from attrdict import AttrDict

from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

import models.unet_normals as unet
import dataloader
from loss_functions import loss_fn_cosine, loss_fn_radians


###################### Config #############################
CONFIG_FILE_PATH = 'config/config.yaml'
with open(CONFIG_FILE_PATH) as fd:
    config_yaml = yaml.safe_load(fd)
config = AttrDict(config_yaml)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

###################### Logs  #############################
# Create a new directory to save logs
runs = sorted(glob.glob(os.path.join(config.train.logsDir, 'exp-*')))
prev_run_id = int(runs[-1].split('-')[-1]) if runs else 0
MODEL_LOG_DIR = os.path.join(config.train.logsDir, 'exp-{:03d}'.format(prev_run_id + 1))
CHECKPOINT_DIR = os.path.join(MODEL_LOG_DIR, 'checkpoints')
os.makedirs(CHECKPOINT_DIR)
print('Saving logs to folder "{}"'.format(MODEL_LOG_DIR))

CONFIG_SAVE_PATH = os.path.join(MODEL_LOG_DIR, 'config.yaml')
with open(CONFIG_SAVE_PATH, "x") as fd:
    yaml.dump(config_yaml, fd, default_flow_style=False)


writer = SummaryWriter(MODEL_LOG_DIR, comment='create-graph')

# Write config to tensorboard
string_out = io.StringIO()
yaml.dump(config_yaml, string_out, default_flow_style=False)
config_str = string_out.getvalue()
config_str = config_str.split('\n')
string = ''
for i in config_str:
    string = string + '    ' + i + '\n\r'

writer.add_text('Config', string, global_step=None)

###################### DataLoader #############################

# Create a dataset object for each dataset in our list
db_trainval = []
for dataset in config.train.datasets:
    dataset = dataloader.SurfaceNormalsDataset(
        input_dir=dataset.images,
        label_dir=dataset.labels,
        transform=None,
        input_only=None
    )
    db_trainval.append(dataset)

# Join all the datasets into 1 large dataset
db_trainval = torch.utils.data.ConcatDataset(db_trainval)

# Split into training and validation datasets
# What percentage of dataset to be used for training
train_size = int(config.train.percentageDataForTraining * len(db_trainval))
test_size = len(db_trainval) - train_size
db_train, db_validation = torch.utils.data.random_split(db_trainval, [train_size, test_size])


trainLoader = DataLoader(db_train, batch_size=config.train.batchSize,
                         shuffle=True, num_workers=config.train.numWorkers, drop_last=True,
                         pin_memory=True)
validationLoader = DataLoader(
    db_validation, batch_size=config.train.validationBatchSize, shuffle=False,
    num_workers=config.train.numWorkers, drop_last=False, pin_memory=True)


###################### ModelBuilder #############################
if config.train.model == 'unet':
    model = unet.Unet(num_classes=config.train.numClasses)
else:
    raise ValueError('Invalid model "{}" in config file. Must be one of ["unet"]'.format(config.train.model))

if config.train.transferLearning:
    if not os.path.isfile(config.train.pathWeightsPrevRun):
        raise ValueError('Invalid path to the given weights file for transfer learning.\
                The file {} does not exist'.format(config.train.pathWeightsPrevRun))

    CHECKPOINT = torch.load(config.train.pathWeightsPrevRun, map_location='cpu')

    if 'model_state_dict' in CHECKPOINT:
        # Newer weights file with various dicts
        print(colored('Continuing training from checkpoint...Loaded data from checkpoint.', 'green'))
        print('    Last Epoch Loss:', CHECKPOINT['epoch_loss'])
        print('    Config:\n', CHECKPOINT['config'], '\n\n')

        model.load_state_dict(CHECKPOINT['model_state_dict'])
    else:
        # Old checkpoint containing only model's state_dict()
        model.load_state_dict(CHECKPOINT)


# Enable Multi-GPU training
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    model = nn.DataParallel(model)

model = model.to(device)
model.train()

###################### Setup Optimization #############################
optimizer = torch.optim.Adam(model.parameters(), lr=config.train.adamOptim.learningRate,
                             weight_decay=config.train.adamOptim.weightDecay)
step_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=config.train.lrSchedulerStep.step_size,
                                                    gamma=config.train.lrSchedulerStep.gamma)
plateau_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=config.train.lrSchedulerPlateau.factor,
    patience=config.train.lrSchedulerPlateau.patience, verbose=True)

# Continue Training from prev checkpoint if required
if config.train.transferLearning:
    # TODO: remove backward compatibility. Check if optim works properly with this method.
    if 'model_state_dict' in CHECKPOINT:
        optimizer.load_state_dict(CHECKPOINT['optimizer_state_dict'])
        prev_loss = CHECKPOINT['epoch_loss']


### Select Loss Func ###
if config.train.lossFunc == 'cosine':
    criterion = loss_fn_cosine
elif config.train.lossFunc == 'radians':
    criterion = loss_fn_radians
else:
    raise ValueError('Invalid lossFunc from config file. Can only be "cosine" or "radians".\
                     Value passed is: {}'.format(config.train.lossFunc))


###################### Train Model #############################
# Calculate total iter_num
if config.train.transferLearning and config.train.continueTraining and 'model_state_dict' in CHECKPOINT:
    # TODO: remove this second check soon. Kept for ensuring backcompatibility
    total_iter_num = CHECKPOINT['total_iter_num'] + 1
    START_EPOCH = CHECKPOINT['epoch'] + 1
    END_EPOCH = CHECKPOINT['epoch'] + config.train.numEpochs
else:
    total_iter_num = 0
    START_EPOCH = 0
    END_EPOCH = config.train.numEpochs

for epoch in range(START_EPOCH, END_EPOCH):
    print('Epoch {}/{}'.format(epoch, END_EPOCH - 1))
    print('-' * 30)
    print('=' * 10)
    print('Train:')

    # Each epoch has a training and validation phase
    running_loss = 0.0

    # Iterate over data.
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

        # Save image to tensorboard every N epochs
        if (epoch % config.train.saveImageInterval) == 0:
            img_tensor = inputs[:3].detach().cpu()
            output_tensor = normal_vectors_norm[:3].detach().cpu()
            label_tensor = labels[:3].detach().cpu()

            images = []
            for img, output, label in zip(img_tensor, output_tensor, label_tensor):
                images.append(img)
                images.append(output)
                images.append(label)

            grid_image = make_grid(images, 3, normalize=True, scale_each=True)
            writer.add_image('Train', grid_image, epoch)

        # Print loss every N Batches
        if (iter_num % 2) == 0:
            if config.train.lossFunc == 'cosine':
                print('Epoch{} Batch{} BatchLoss: {:.4f} (cosine loss)'.format(epoch, iter_num, loss.item()))
            else:
                print('Epoch{} Batch{} BatchLoss: {:.4f} radians'.format(epoch, iter_num, loss.item()))

    epoch_loss = running_loss / (len(trainLoader))
    writer.add_scalar('Train Epoch Loss', epoch_loss, epoch)
    print('\nTrain Epoch Loss: {:.4f}'.format(epoch_loss))

    # step_lr_scheduler.step() # This is for the Step LR Scheduler
    # plateau_lr_scheduler.step(epoch_loss) # This is for the Reduce LR on Plateau Scheduler
    current_learning_rate = optimizer.param_groups[0]['lr']
    writer.add_scalar('learning_rate', current_learning_rate, epoch)

    # Save the model checkpoint every N epochs
    if (epoch % config.train.saveModelInterval) == 0:
        filename = 'checkpoint-epoch-{:04d}.pth'.format(epoch)

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
        }, os.path.join(CHECKPOINT_DIR, filename))

    ###################### Run Validation and Test Set  #############################
    nTestInterval = config.train.testInterval
    if nTestInterval > 0 and epoch % nTestInterval == (nTestInterval - 1):
        model.eval()
        dataloaders_dict = {'Validation': validationLoader}
        for key in dataloaders_dict:
            print('\n' + '=' * 10)
            print(key + ':')

            # TODO: rename the dataloader variable, conflics with module name. optionally, change module name.
            dataloader = dataloaders_dict[key]
            running_loss = 0.0

            for ii, sample_batched in enumerate(dataloader):
                inputs, labels = sample_batched

                # Forward pass of the mini-batch
                inputs = inputs.to(device)
                labels = labels.to(device)

                with torch.no_grad():
                    normal_vectors = model(inputs)

                normal_vectors_norm = nn.functional.normalize(normal_vectors, p=2, dim=1)
                loss = criterion(normal_vectors_norm, labels, reduction='elementwise_mean')

                running_loss += loss.item()

            # Save output image to tensorboard
            img_tensor = inputs[:3].detach().cpu()
            output_tensor = normal_vectors_norm[:3].detach().cpu()
            label_tensor = labels[:3].detach().cpu()

            images = []
            for img, output, label in zip(img_tensor, output_tensor, label_tensor):
                images.append(img)
                images.append(output)
                images.append(label)

            grid_image = make_grid(images, 3, normalize=True, scale_each=True)
            writer.add_image(key, grid_image, epoch)

            epoch_loss = running_loss / (len(dataloader))

            writer.add_scalar(key + ' Epoch Loss', epoch_loss, epoch)
            print(key + ' Epoch Loss: {:.4f}\n\n'.format(epoch_loss))


writer.close()
