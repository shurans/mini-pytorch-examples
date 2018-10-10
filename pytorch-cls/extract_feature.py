import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import models
import matplotlib.pyplot as plt
import time
import os
from data_loader import Dataset,Options
import h5py

# python3 extract_feature.py --gpu=4  --file_list=../../illummaps/pytorch-learning/datalist/trainlist_NN.txt --dataroot=/n/fs/rgbd/data/matterport/v1/ 
# python3 extract_feature.py --file_list=../../illummaps/pytorch-learning/datalist/trainlist_NN_small.txt --dataroot=../../data/matterport/v1/ 

###################### Options #############################
opt = Options().parse()
opt.batchsize = 1
opt.shuffle = False  # no shuffle
opt.has_class_label = 0
phase = opt.phase
#opt.how_many = 10000
device = torch.device("cuda:"+ opt.gpu if torch.cuda.is_available() else "cpu")
print(device)

###################### DataLoader #############################
dataloader = Dataset(opt)


###################### ModelBuilder #############################
model = models.resnet18(pretrained=True)
model = model.to(device)
num_ftrs = model.fc.in_features
features = torch.zeros(dataloader.size(),1000)


cnt = 0
for i in range(int(dataloader.size()/opt.batchSize)):
    inputs,_ =  dataloader.get_batch()

    inputs = inputs.to(device)
    outputs = model(inputs)
    print(outputs.size())
    features[cnt:cnt+opt.batchSize,:] = outputs.cpu()
    cnt = cnt+opt.batchSize
    print('%d,%d/%d'%(i,cnt,dataloader.size()))

hf = h5py.File('feature.h5', 'w')
hf.create_dataset('features', data=features)
hf.close()