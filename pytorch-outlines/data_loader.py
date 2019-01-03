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
import OpenEXR
import Imath
from skimage.transform import resize
import imageio
import cv2


class Dataset():
    def __init__(self, opt):
        self.dataroot = opt.dataroot
        self.file_list = opt.file_list
        self.batchSize = opt.batchSize
        self.doshuffle = opt.shuffle
        self.phase = opt.phase
        self.currIdx = 0
        self.imsize = opt.imsize
        self.read_file_list()

        self.ignore_label = 255
        self.ordered_train_labels = np.append([self.ignore_label], np.asarray(range(13)))

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
        # transform_list.append(transforms.Resize([self.imsize,self.imsize]))
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
        tf = transforms.Compose(transform_list)
        im = tf(im)
        return im

    def transformLabel(self, label):
        transform_list = []
        transform_list.append(transforms.Resize([self.imsize,self.imsize]))
        transform_list.append(transforms.ToTensor())
        tf = transforms.Compose(transform_list)
        label = tf(label)
        return label


    def get_batch(self):
        # this function get image and segmentation mask
        im_batch = torch.Tensor()
        label_batch = torch.LongTensor()
        label_truth_batch = torch.Tensor()

        for x in range(self.batchSize):
            self.currIdx = self.currIdx + 1
            if self.currIdx >= len(self.datalist):
                self.currIdx = 0
                if self.doshuffle:
                   self.shuffle()

            im_path = self.dataroot + self.datalist[self.currIdx][0]
            label_path = self.dataroot + self.datalist[self.currIdx][1]
            # label_truth_batch = self.dataroot + self.datalist[self.currIdx][2]

            # Open pre-processed imgs
            im = np.load(im_path)
            im = torch.tensor(im, dtype=torch.float)
            im = im.unsqueeze(0)

            # Open outlines
            label = Image.open(label_path)
            label_np = np.asarray(label).copy()

            # https://stackoverflow.com/questions/8188726/how-do-i-do-this-array-lookup-replace-with-numpy
            label_np = self.ordered_train_labels[label_np].astype(np.uint8)

            label = Image.fromarray(label_np)
            label = self.transformLabel(label)
            label = label.unsqueeze(0).type('torch.LongTensor')

            # Open outlines ground truth
            # label_truth = Image.open(label_truth_path)
            # label_truth = np.asarray(label_truth)

            # Create batches out of data
            im_batch=torch.cat((im_batch, im), 0)
            label_batch = torch.cat((label_batch,label),0)
            # label_truth_batch = torch.cat((label_truth_batch,label_truth),0)

        return im_batch, label_batch
