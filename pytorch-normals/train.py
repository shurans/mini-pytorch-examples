'''Train unet for surface normals
'''

import os
import sys
sys.path.append('./models/')
import torch
import torch.nn as nn
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
import numpy as np
from data_loader import Dataset, Options
import models.unet_normals as unet
import utils.create_datalist


class OPT():
    def __init__(self):
        self.dataroot = './data/'
        self.file_list = './data/datalist'
        self.batchSize = 8
        self.shuffle = True
        self.phase = 'train'
        self.num_epochs = 500
        self.imsize = (288,512)
        self.num_classes = int(3)
        self.gpu = '1'
        self.logs_path = 'logs/exp100'
        self.use_pretrained = False

opt = OPT()



###################### Options #############################
phase = opt.phase
device = torch.device("cuda:"+ opt.gpu if torch.cuda.is_available() else "cpu")

###################### TensorBoardX #############################
# if os.path.exists(opt.logs_path):
#     raise Exception('The folder \"{}\" already exists! Define a new log path or delete old contents.'.format(opt.logs_path))
    
writer = SummaryWriter(opt.logs_path, comment='create-graph')
graph_created = False

###################### DataLoader #############################
dataloader = Dataset(opt)


###################### ModelBuilder #############################
model = unet.Unet(num_classes=opt.num_classes)

# Load weights from checkpoint
if (opt.use_pretrained == True):
    checkpoint_path = 'logs/exp7/checkpoints/checkpoint.pth'
    model.load_state_dict(torch.load(checkpoint_path))

model = model.to(device)
model.train()

###################### Setup Optimazation #############################
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001) #, weight_decay=0.0001
step_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
plateau_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, patience=25, verbose=True)


###################### Loss fuction - Cosine Loss #############################
'''
@input: The 2 vectors whose cosine loss is to be calculated
The dimensions of the matrices are expected to be (batchSize, 3, imsize, imsize). 

@return: 
elementwise_mean: will return the sum of all losses divided by num of elements
none: The loss will be calculated to be of size (batchSize, imsize, imsize) containing cosine loss of each pixel
'''
def loss_fn_cosine(input_vec, target_vec, reduction='elementwise_mean'):
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    loss_val = 1.0 - cos(input_vec, target_vec)
    if (reduction=='elementwise_mean'):
        return torch.mean(loss_val)
    elif (reduction=='none'):
        return loss_val
    else:
        raise Exception('Warning! The reduction is invalid. Please use \'elementwise_mean\' or \'none\''.format())

###################### Loss fuction - Avg Angle Calc #############################
'''
@input: The 2 vectors whose cosine loss is to be calculated
The dimensions of the matrices are expected to be (batchSize, 3, imsize, imsize). 

@return: 
elementwise_mean: will return the sum of all losses divided by num of elements
none: The loss will be calculated to be of size (batchSize, imsize, imsize) containing cosine loss of each pixel
'''
def loss_fn_radians(input_vec, target_vec, reduction='elementwise_mean'):
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    loss_cos = cos(input_vec, target_vec)    
    if (reduction=='elementwise_mean'):
        return torch.acos(torch.mean(loss_cos))
    elif (reduction=='none'):
        return torch.acos(loss_cos)
    else:
        raise Exception('Warning! The reduction is invalid. Please use \'elementwise_mean\' or \'none\''.format())


### Select Loss Func ###
loss_fn = loss_fn_cosine
        
def normal_to_rgb(normals_to_convert):
    camera_normal_rgb = (normals_to_convert + 1)/2
    # camera_normal_rgb *= 127.5
    # camera_normal_rgb = camera_normal_rgb.astype(np.uint8)
    return camera_normal_rgb


###################### Train Model #############################
# Calculate total iter_num
total_iter_num = 0

for epoch in range(0, opt.num_epochs):
    print('Epoch {}/{}'.format(epoch, opt.num_epochs - 1))
    print('-' * 30)

    # Each epoch has a training and validation phase
    running_loss = 0.0
    
    


    # Iterate over data.
    for i in range(int(dataloader.size()/opt.batchSize)):
        total_iter_num += 1
        
        # Get data
        inputs, labels =  dataloader.get_batch()
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        ## Create Graph ##
        if graph_created == False:
            graph_created = True
            writer.add_graph(model, inputs, verbose=False)
        
        # Forward + Backward Prop
        optimizer.zero_grad()
        torch.set_grad_enabled(True)
        normal_vectors = model(inputs)
        normal_vectors_norm = nn.functional.normalize(normal_vectors, p=2, dim=1)
        
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

            grid_image = make_grid(images, 3, normalize=True, scale_each=True )
            writer.add_image('Train', grid_image, epoch)
        
        if (i % 2 == 0):
            print('Epoch{} Batch{} Loss: {:.4f} (rad)'.format(epoch, i, loss.item()))

    epoch_loss = running_loss / (dataloader.size()/opt.batchSize)
    writer.add_scalar('epoch_loss', epoch_loss, epoch)
    print('{} Loss: {:.4f}'.format(phase, epoch_loss))
    
    #step_lr_scheduler.step() # This is for the Step LR Scheduler
    #plateau_lr_scheduler.step(epoch_loss) # This is for the Reduce LR on Plateau Scheduler
    learn_rate = optimizer.param_groups[0]['lr']
    writer.add_scalar('learning_rate', learn_rate, epoch)
    
    # Save the model checkpoint
    directory = opt.logs_path+'/checkpoints/'
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    if (epoch % 5 == 0):
        filename = opt.logs_path + '/checkpoints/checkpoint-epoch_{}.pth'.format(epoch)
        torch.save(model.state_dict(), filename)
        

# Save final Checkpoint
filename = opt.logs_path + '/checkpoints/checkpoint.pth'
torch.save(model.state_dict(), filename)


