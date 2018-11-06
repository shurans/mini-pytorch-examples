import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import models
import matplotlib.pyplot as plt
import time
import os
from data_loader import Dataset,Options

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.cpu().numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.show()

###################### Options #############################
opt = Options().parse()
opt.batchSize=1
phase = opt.phase
opt.shuffle = False
device = torch.device("cuda:"+ opt.gpu if torch.cuda.is_available() else "cpu")
print(device)

###################### DataLoader #############################
dataloader = Dataset(opt)

###################### ModelBuilder #############################
model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model = model.to(device)
model.eval()
###################### Load model #############################
load_path = '1000_net_exp.pth'
state_dict = torch.load(load_path, map_location=str(device))
model.load_state_dict(state_dict)


running_corrects = 0
for i in range(int(dataloader.size()/opt.batchSize)):
    inputs, labels =  dataloader.get_batch()
    inputs = inputs.to(device)
    labels = labels.to(device)
    # imshow(inputs[0])
    # forward
    outputs = model(inputs)
    _, preds = torch.max(outputs, 1)
    print(outputs)

    running_corrects += torch.sum(preds == labels.data)

epoch_acc = running_corrects.double() / dataloader.size()
print('Acc: {:.4f}'.format(epoch_acc))
