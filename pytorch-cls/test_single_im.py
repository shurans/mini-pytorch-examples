###################################################
###############Written by Shuran Song##############
###################################################
import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import models
import matplotlib.pyplot as plt
import time
import os
from data_loader import Dataset,Options
from PIL import Image
from torchvision import transforms, utils

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.show()

def transformImage(im,imsize):
    transform_list = []
    transform_list.append(transforms.Resize([imsize,imsize]))
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    tf = transforms.Compose(transform_list)
    im = tf(im)
    return im

###################### Options #############################
opt = Options().parse()
phase = opt.phase
device = torch.device("cuda:"+ opt.gpu if torch.cuda.is_available() else "cpu")
print(device)

model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model = model.to(device)
model.eval()
###################### Load model #############################
load_path = '500_net_exp.pth'
state_dict = torch.load(load_path, map_location=str(device))
model.load_state_dict(state_dict)

###################### Load Image #############################
im_path = opt.test_im
im = Image.open(im_path).convert("RGB")
im = transformImage(im,opt.imsize)
inputs = im.unsqueeze(0)
imshow(inputs[0])

###################### Test Image #############################
inputs = inputs.to(device)
outputs = model(inputs)
_, preds = torch.max(outputs, 1)
print(outputs)
print(preds)
