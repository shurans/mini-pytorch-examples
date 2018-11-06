import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import models
import matplotlib.pyplot as plt
import time
import os
from data_loader import Dataset,Options

###################### Options #############################
opt = Options().parse()
phase = opt.phase
device = torch.device("cuda:"+ opt.gpu if torch.cuda.is_available() else "cpu")
print(device)

###################### DataLoader #############################
dataloader = Dataset(opt)

###################### ModelBuilder #############################
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)

model = model.to(device)
criterion = nn.CrossEntropyLoss()


###################### Setup Optimazation #############################
# Observe that all parameters are being optimized
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)


iter_count = 0
for epoch in range(opt.num_epochs):
    print('Epoch {}/{}'.format(epoch, opt.num_epochs - 1))
    print('-' * 10)

    # Each epoch has a training and validation phase
    exp_lr_scheduler.step()
    model.train()  # Set model to training mode

    running_loss = 0.0
    running_corrects = 0

    # Iterate over data.
    for i in range(int(dataloader.size()/opt.batchSize)):
        inputs, labels =  dataloader.get_batch()

        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        with torch.set_grad_enabled(True):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            # backward + optimize 
            loss.backward()
            optimizer.step()
            print(preds)

         # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        
        if iter_count%10==0:
            print('{} Loss: {:.4f}'.format(phase, loss.item()))
        
        if iter_count%100==0:
            save_filename = '%s_net_%s.pth' % (iter_count,opt.name)
            torch.save(model.state_dict(), save_filename)
        iter_count = iter_count+1

    epoch_loss = running_loss / dataloader.size()
    epoch_acc = running_corrects.double() / dataloader.size()
    print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
