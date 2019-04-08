import socket
import timeit
from datetime import datetime
import os
import glob
from collections import OrderedDict
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import argparse
from imgaug import augmenters as iaa
import imgaug as ia


# PyTorch includes
import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import torch.nn as nn


# Tensorboard include
from tensorboardX import SummaryWriter

# Custom includes
import dataloader
from models import deeplab_xception, deeplab_resnet
from utils.lr_finder import LRFinder
from loss_functions import loss_fn_cosine, loss_fn_radians, cross_entropy2d

p = OrderedDict()  # Parameters to include in report
p['trainBatchSize'] = 96  # Training batch size
testBatchSize = 1  # Testing batch size
useTest = True  # See evolution of the test set when training
nTestInterval = 1  # Run on test set every nTestInterval epochs
snapshot = 2  # Store a model every snapshot epochs

p['nAveGrad'] = 1  # Average the gradient of several iterations
p['lr'] = 1e-10  # Learning rate
p['wd'] = 5e-4  # Weight decay
p['momentum'] = 0.9  # Momentum
p['epoch_size'] = 2  # How many epochs to change learning rate

p['Model'] = 'deeplab'  # Choose model: unet or deeplab
backbone = 'xception'  # For deeplab only: Use xception or resnet as feature extractor,
num_of_classes = 3
imsize = 128  # 256 or 512
output_stride = 16  # 8 or 16, 8 is better. Increases resolution of convolution layers.
numInputChannels = 3

# Network definition
if p['Model'] == 'deeplab':
    if backbone == 'xception':
        net = deeplab_xception.DeepLabv3_plus(nInputChannels=numInputChannels, n_classes=num_of_classes,
                                              os=output_stride, pretrained=True)
    elif backbone == 'resnet':
        net = deeplab_resnet.DeepLabv3_plus(nInputChannels=numInputChannels, n_classes=num_of_classes,
                                            os=output_stride, pretrained=True)
    else:
        raise NotImplementedError
    modelName = 'deeplabv3plus-' + backbone

    # Use the following optimizer
    optimizer = optim.SGD(net.parameters(), lr=p['lr'], momentum=p['momentum'], weight_decay=p['wd'])
    p['optimizer'] = str(optimizer)

    # Use the following loss function
    criterion = loss_fn_cosine
else:
    raise NotImplementedError

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Enable Multi-GPU training
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    net = nn.DataParallel(net)

augs_train = iaa.Sequential([
    iaa.Scale((imsize, imsize), 0),
])


db_train = dataloader.SurfaceNormalsDataset(
    input_dir='data/datasets/train/milk-bottles-train/resized-files/preprocessed-rgb-imgs',
    label_dir='data/datasets/train/milk-bottles-train/resized-files/preprocessed-camera-normals',
    transform=augs_train,
    input_only=None,
)


trainLoader = DataLoader(db_train, batch_size=p['trainBatchSize'], shuffle=True, num_workers=32, drop_last=True)


# %matplotlib inline

lr_finder = LRFinder(net, optimizer, criterion, device="cuda")
lr_finder.range_test(trainLoader, end_lr=1, num_iter=100)
lr_finder.plot()
plt.show()