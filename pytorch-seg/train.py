import sys
sys.path.append('./models/')
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from data_loader import Dataset,Options
import models.resnet_dilated as reset_dilated
from tensorboardX import SummaryWriter


# mkdir data
# wget http://www.doc.ic.ac.uk/~ahanda/nyu_test_rgb.tgz
# wget http://www.doc.ic.ac.uk/~ahanda/nyu_train_rgb.tgz
# wget https://github.com/ankurhanda/nyuv2-meta-data/raw/master/test_labels_13/nyuv2_test_class13.tgz
# wget https://github.com/ankurhanda/nyuv2-meta-data/raw/master/train_labels_13/nyuv2_train_class13.tgz
# python3 train.py

###################### Options #############################
opt = Options().parse()
phase = opt.phase
device = torch.device("cuda:"+ opt.gpu if torch.cuda.is_available() else "cpu")

###################### TensorBoardX #############################
writer = SummaryWriter(opt.logs_path, comment='create-graph')
graph_created = False

###################### DataLoader #############################
dataloader = Dataset(opt)
inputs, labels =  dataloader.get_batch()

###################### ModelBuilder #############################

model = reset_dilated.Resnet18_8s(num_classes=opt.num_classes)
model = model.to(device)
model.train()
#criterion = nn.CrossEntropyLoss(size_average=False).to(device)
criterion = nn.CrossEntropyLoss(reduction='sum').to(device)

###################### Setup Optimazation #############################
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

###################### Custom fuction #############################
def flatten_logits(logits, number_of_classes):
    """Flattens the logits batch except for the logits dimension"""
    logits_permuted = logits.permute(0, 2, 3, 1)
    print('\n logits_permuted dim:')
    print(logits_permuted.shape)

    logits_permuted_cont = logits_permuted.contiguous()
    print('\n logits_permuted_cont dim:')
    print(logits_permuted_cont.shape)

    logits_flatten = logits_permuted_cont.view(-1, number_of_classes)
    print('\n logits_flatten dim:')
    print(logits_flatten.shape)

    return logits_flatten

def flatten_annotations(annotations):
    return annotations.view(-1)

def get_valid_annotations_index(flatten_annotations, mask_out_value=255):
    return torch.squeeze( torch.nonzero((flatten_annotations != mask_out_value )), 1)

inputs, labels =  dataloader.get_batch()
inputs = inputs.to(device)


###################### Train Model #############################
# Calculate total iter_num
total_iter_num = 0

for epoch in range(opt.num_epochs):
    print('Epoch {}/{}'.format(epoch, opt.num_epochs - 1))
    print('-' * 10)

    # Each epoch has a training and validation phase
    exp_lr_scheduler.step()
    running_loss = 0.0

    # Iterate over data.
    for i in range(int(dataloader.size()/opt.batchSize)):
        total_iter_num += 1

        inputs, labels =  dataloader.get_batch()

        inputs = inputs.to(device)
        labels = labels.to(device)


        # We need to flatten annotations and logits to apply index of valid annotations.
        print('\nlabels dim:')
        print(labels.shape)

        anno_flatten = flatten_annotations(labels)
        print('\nanno_flatten dim:')
        print(anno_flatten.shape)

        index = get_valid_annotations_index(anno_flatten, mask_out_value=255)
        print('\nindex dim:')
        print(index.shape)

        anno_flatten_valid = torch.index_select(anno_flatten, 0, index)
        print('\nanno_flatten_valid dim:')
        print(anno_flatten_valid.shape)

        # fowrad + backward
        optimizer.zero_grad()
        torch.set_grad_enabled(True)
        logits = model(inputs)
        print('\nlogits dim:')
        print(logits.shape)

        logits_flatten = flatten_logits(logits, number_of_classes=opt.num_classes)
        logits_flatten_valid = torch.index_select(logits_flatten, 0, index)
        loss = criterion(logits_flatten_valid, anno_flatten_valid)

        ###################### Create Graph #############################
        if graph_created == False:
            graph_created = True
            writer.add_graph(model, inputs, verbose=False)
        #################################################################

        loss.backward()
        optimizer.step()

        # statistics
        loss_num = (loss.item() / logits_flatten_valid.size(0))
        running_loss += loss_num
        writer.add_scalar('running_loss', loss_num, total_iter_num)

        if (i%10 == 0) :
            print('{} Loss: {:.4f}'.format(phase, (loss.item() / logits_flatten_valid.size(0)) ))

        if (i%1000 == 0) :
            filename = 'checkpoints/checkpoint-{}-{}.pt'.format(epoch,i)
            #model.save_state_dict(filename)
            torch.save(model.state_dict(), filename)

    epoch_loss = running_loss / dataloader.size()
    print('{} Loss: {:.4f}'.format(phase, epoch_loss))
    filename = 'checkpoints/checkpoint-{}-{}.pt'.format(epoch,i)
    #model.save_state_dict(filename)
    torch.save(model.state_dict(), filename)
