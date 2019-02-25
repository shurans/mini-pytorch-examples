'''Train unet for surface normals
'''

import os
import sys
# sys.path.append('./models/')

from tensorboardX import SummaryWriter
import numpy as np

from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

# from data_loader import Dataset, Options
import models.unet_normals as unet
import dataloader


class OPT():
    def __init__(self):
        self.dataroot = './data/'
        self.file_list = './data/datalist'
        self.batchSize = 24
        self.shuffle = True
        self.phase = 'train'
        self.num_epochs = 500
        self.imsize = (288, 512)
        self.num_classes = int(3)
        self.gpu = '1'
        self.logs_path = 'logs/exp109'
        self.use_pretrained = False


opt = OPT()





###################### Options #############################
phase = opt.phase
# device = torch.device("cuda:"+ opt.gpu if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

###################### TensorBoardX #############################
# if os.path.exists(opt.logs_path):
#     raise Exception('The folder \"{}\" already exists! Define a new log path or delete old contents.'.format(opt.logs_path))

writer = SummaryWriter(opt.logs_path, comment='create-graph')

###################### DataLoader #############################
# Make new dataloader for each object's dataset
db_trainval1 = dataloader.SurfaceNormalsDataset(
    input_dir='data/datasets/milk-bottles/resized-files/preprocessed-rgb-imgs',
    label_dir='data/datasets/milk-bottles/resized-files/preprocessed-camera-normals',
    transform=None,
    input_only=None
)

# Join all the datasets into 1 large dataset
db_trainval = torch.utils.data.ConcatDataset([db_trainval1])

# Split into training and validation datasets
# What percentage of dataset to be used for training
percentage_as_training_set = 0.9
train_size = int(percentage_as_training_set * len(db_trainval))
test_size = len(db_trainval) - train_size
db_train, db_validation = torch.utils.data.random_split(
    db_trainval, [train_size, test_size])

trainLoader = DataLoader(db_train, batch_size=opt.batchSize,
                         shuffle=True, num_workers=32, drop_last=True)
validationLoader = DataLoader(
    db_validation, batch_size=opt.batchSize, shuffle=False, num_workers=32, drop_last=True)


###################### ModelBuilder #############################
model = unet.Unet(num_classes=opt.num_classes)

# Enable Multi-GPU training
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    model = nn.DataParallel(model)

# Load weights from checkpoint
if (opt.use_pretrained == True):
    checkpoint_path = 'logs/exp7/checkpoints/checkpoint.pth'
    model.load_state_dict(torch.load(checkpoint_path))

model = model.to(device)
model.train()

###################### Setup Optimization #############################
optimizer = torch.optim.Adam(
    model.parameters(), lr=0.0001, weight_decay=0.0001)
step_lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=7, gamma=0.1)
plateau_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.8, patience=25, verbose=True)


###################### Loss fuction - Cosine Loss #############################
def loss_fn_cosine(input_vec, target_vec, reduction='elementwise_mean'):
    '''A cosine loss function for use with surface normals estimation.
    Calculates the cosine loss between 2 vectors. Both should be of the same size.

    Arguments:
        input_vec {tensor} -- The 1st vectors with whom cosine loss is to be calculated
                              The dimensions of the matrices are expected to be (batchSize, 3, height, width).
        target_vec {tensor } -- The 2nd vectors with whom cosine loss is to be calculated
                                The dimensions of the matrices are expected to be (batchSize, 3, height, width).

    Keyword Arguments:
        reduction {str} -- Can have values 'elementwise_mean' and 'none'.
                           If 'elemtwise_mean' is passed, the mean of all elements is returned
                           if 'none' is passed, a matrix of all cosine losses is returned, same size as input.
                           (default: {'elementwise_mean'})

    Raises:
        Exception -- Exception is an invalid reduction is passed

    Returns:
        tensor -- A single mean value of cosine loss or a matrix of elementwise cosine loss.
    '''

    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    loss_cos = 1.0 - cos(input_vec, target_vec)
    if (reduction == 'elementwise_mean'):
        loss_cos = torch.mean(loss_cos)
    elif (reduction == 'none'):
        loss_cos = loss_cos
    else:
        raise Exception(
            'Invalid value for reduction  parameter passed. Please use \'elementwise_mean\' or \'none\''.format())

    return loss_cos


###################### Loss fuction - Avg Angle Calc #############################
def loss_fn_radians(input_vec, target_vec, reduction='elementwise_mean'):
    '''Loss func for estimation of surface normals. Calculated the angle between 2 vectors
    by taking the inverse cos of cosine loss.

    Arguments:
        input_vec {tensor} -- First vector with whole loss is to be calculated. Expected size (batchSize, 3, height, width)
        target_vec {tensor} -- Second vector with whom the loss is to be calculated. Expected size (batchSize, 3, height, width)

    Keyword Arguments:
        reduction {str} -- Can have values 'elementwise_mean' and 'none'.
                           If 'elemtwise_mean' is passed, the mean of all elements is returned
                           if 'none' is passed, a matrix of all cosine losses is returned, same size as input.
                           (default: {'elementwise_mean'})

    Raises:
        Exception -- If any unknown value passed as reduction argument.

    Returns:
        tensor -- Loss from 2 input vectors. Size depends on value of reduction arg.
    '''

    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    loss_cos = cos(input_vec, target_vec)
    if (reduction == 'elementwise_mean'):
        loss_rad = torch.acos(torch.mean(loss_cos))
    elif (reduction == 'none'):
        loss_rad = torch.acos(loss_cos)
    else:
        raise Exception(
            'Invalid value for reduction  parameter passed. Please use \'elementwise_mean\' or \'none\''.format())

    return loss_rad


### Select Loss Func ###
loss_fn = loss_fn_cosine


###################### Train Model #############################
# Calculate total iter_num
total_iter_num = 0

for epoch in range(0, opt.num_epochs):
    print('Epoch {}/{}'.format(epoch, opt.num_epochs - 1))
    print('-' * 30)

    # Each epoch has a training and validation phase
    running_loss = 0.0

    # Iterate over data.
    for i, batch in enumerate(trainLoader):
        total_iter_num += 1

        # Get data
        inputs, labels = batch
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Forward + Backward Prop
        optimizer.zero_grad()
        torch.set_grad_enabled(True)
        normal_vectors = model(inputs)
        normal_vectors_norm = nn.functional.normalize(
            normal_vectors, p=2, dim=1)

        loss = loss_fn(normal_vectors_norm, labels, reduction='elementwise_mean')
        loss.backward()
        optimizer.step()

        # statistics
        running_loss += loss.item()
        writer.add_scalar('loss', loss.item(), total_iter_num)

        # TODO:
        # Print image every N epochs
        nTestInterval = 1
        if (epoch % nTestInterval) == 0:
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

        if (i % 2 == 0):
            print('Epoch{} Batch{} Loss: {:.4f} (rad)'.format(
                epoch, i, loss.item()))

    epoch_loss = running_loss / (len(trainLoader))
    writer.add_scalar('epoch_loss', epoch_loss, epoch)
    print('{} Loss: {:.4f}'.format(phase, epoch_loss))

    # step_lr_scheduler.step() # This is for the Step LR Scheduler
    # plateau_lr_scheduler.step(epoch_loss) # This is for the Reduce LR on Plateau Scheduler
    learn_rate = optimizer.param_groups[0]['lr']
    writer.add_scalar('learning_rate', learn_rate, epoch)

    # Save the model checkpoint
    directory = opt.logs_path+'/checkpoints/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    if (epoch % 5 == 0):
        filename = opt.logs_path + \
            '/checkpoints/checkpoint-epoch_{}.pth'.format(epoch)
        torch.save(model.state_dict(), filename)


# Save final Checkpoint
filename = opt.logs_path + '/checkpoints/checkpoint.pth'
torch.save(model.state_dict(), filename)
