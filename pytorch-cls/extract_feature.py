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

# python3 extract_feature.py   --file_list=../../illummaps/pytorch-learning/datalist/trainlist_NN.txt --dataroot=/n/fs/rgbd/data/matterport/v1/ 
# python3 extract_feature.py --file_list=../../illummaps/pytorch-learning/datalist/trainlist_NN_small.txt --dataroot=../../data/matterport/v1/ 
# python3 extract_feature.py  --phase='test'  --file_list=../../illummaps/pytorch-learning/datalist/testlist_NN.txt --dataroot=/n/fs/rgbd/data/matterport/v1/ 
###################### Options #############################
opt = Options().parse()
opt.batchsize = 1
opt.shuffle = False  # no shuffle
opt.has_class_label = 0
phase = opt.phase
if opt.phase == 'test':
    opt.how_many = 400
else:
    opt.how_many = 5000

device = torch.device("cuda:"+ opt.gpu if torch.cuda.is_available() else "cpu")
print(device)

###################### DataLoader #############################
dataloader = Dataset(opt)


###################### ModelBuilder #############################
model = models.resnet18(pretrained=True)
model = model.to(device)
model.eval()
num_ftrs = model.fc.in_features
features = np.zeros((opt.how_many,1000))


cnt = 0
for i in range(int(dataloader.size()/opt.batchSize)):
    inputs,_ =  dataloader.get_batch()

    inputs = inputs.to(device)
    outputs = model(inputs)
    features[cnt:cnt+opt.batchSize,:] = outputs.detach().cpu().numpy() 
    

    cnt = cnt+opt.batchSize
    print('%d,%d/%d'%(i,cnt,dataloader.size()))
    if cnt+opt.batchSize > opt.how_many:
        break


if opt.phase == 'test':
    hf = h5py.File('feature_test.h5', 'w')
else:
    hf = h5py.File('feature_train.h5', 'w')

hf.create_dataset('features', data=features)
hf.close()
