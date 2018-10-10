import os
import torchvision
import csv
from torchvision import transforms, utils
import numpy as np
from PIL import Image
import torch
import matplotlib.pyplot as plt
import random
import argparse

class Dataset():
    def __init__(self, opt):
        self.dataroot = opt.dataroot
        self.file_list = opt.file_list
        self.batchSize = opt.batchSize
        self.doshuffle = opt.shuffle
        self.phase = opt.phase
        self.currIdx = 0
        self.imsize = opt.imsize
        self.has_class_label = opt.has_class_label
        self.read_file_list()
        if self.doshuffle:
            self.shuffle()

    def read_file_list(self):
        with open(self.file_list, 'r') as f:
            reader = csv.reader(f)
            self.datalist = list(reader)

    def __len__(self):
        return len(self.datalist)

    def size(self):
        return len(self.datalist)
    
    def shuffle(self):
        print("shuffling the dataset")
        random.shuffle(self.datalist)
        

    def transformImage(self, im):
        transform_list = []
        transform_list.append(transforms.Resize([self.imsize,self.imsize]))
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
        tf = transforms.Compose(transform_list)
        im = tf(im)
        return im

    def imshow(self, inp, title=None):
        """Imshow for Tensor."""
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        plt.imshow(inp)
        if title is not None:
            plt.title(title)
        plt.pause(0.001)  # pause a bit so that plots are updated

    def get_batch(self):
        # # Change this function to adapt to different tasks
        im_batch = torch.Tensor()
        class_batch = torch.zeros(self.batchSize, dtype=torch.long)
        for x in range(self.batchSize):
            self.currIdx = self.currIdx+1
            if  self.currIdx  >= len(self.datalist):
                self.currIdx = 0
                if self.doshuffle:
                   self.shuffle()

            im_path = self.dataroot + self.datalist[self.currIdx][0]
            if self.has_class_label:
                im_class = self.datalist[self.currIdx][1]
            else:
                im_class = 0

            im = Image.open(im_path).convert("RGB")
            im = self.transformImage(im)
            im = im.unsqueeze(0)
            

            im_batch = torch.cat((im_batch,im),0)
            class_batch[x] = im_class


        return im_batch,class_batch



    


class Options():
    def __init__(self):
        self.initialized = False
        parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)        
        parser.add_argument('--dataroot',  default='./hymenoptera_data/train/', help='path to images')
        parser.add_argument('--file_list', default='./datalist', help='path to images')
        parser.add_argument('--batchSize', type=int, default=4, help='input batch size')
        parser.add_argument('--shuffle', type=bool, default=True, help='input batch size')
        parser.add_argument('--phase', default='train', help='input batch size')
        parser.add_argument('--num_epochs', type=int, default=10, help='input batch size')
        parser.add_argument('--imsize', type=int, default=224, help='scale images to this size')
        parser.add_argument('--has_class_label', type=int, default=1, help='scale images to this size')
        parser.add_argument('--gpu', default='0', help='scale images to this size')
        self.parser = parser
    
    def parse(self):    
        opt = self.parser.parse_args()
        return opt




# if __name__== "__main__":
#     opt = Options().parse()
#     dataloader = Dataset(opt)

#     for i in range(1000):
#         outim, outlabel = dataloader.get_batch()
#         im_vis = torchvision.utils.make_grid(outim)
#         dataloader.imshow(im_vis)
