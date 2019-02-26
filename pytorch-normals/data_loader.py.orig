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
import OpenEXR, Imath
from skimage.transform import resize


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
        self.ordered_train_labels = np.append( [self.ignore_label] , np.asarray( range(13) ))

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
        transform_list.append(transforms.Resize(self.imsize, interpolation = Image.BILINEAR))
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))
        tf = transforms.Compose(transform_list)
        im = tf(im)
        return im

    def transformLabel(self, label):
        transform_list = []
        transform_list.append(transforms.Resize(self.imsize, interpolation = Image.NEAREST))
        transform_list.append(transforms.ToTensor())
        tf = transforms.Compose(transform_list)
        label = tf(label)
        return label

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
        plt.pause(1)  # pause a bit so that plots are updated

    def exr_loader(self, EXR_PATH, ndim=3):
        """
        loads an .exr file as a numpy array
        :param path: path to the file
        :param ndim: number of channels that the image has,
                        if 1 the 'R' channel is taken
                        if 3 the 'R', 'G' and 'B' channels are taken
        :return: np.array containing the .exr image
        """

        exr_file = OpenEXR.InputFile(EXR_PATH)
        cm_dw = exr_file.header()['dataWindow']
        size = (cm_dw.max.x - cm_dw.min.x + 1, cm_dw.max.y - cm_dw.min.y + 1)

        pt = Imath.PixelType(Imath.PixelType.FLOAT)

        if ndim == 3:
            # read channels indivudally
            allchannels = []
            for c in ['R', 'G', 'B']:
                # transform data to numpy
                channel = np.frombuffer(exr_file.channel(c, pt), dtype=np.float32)
                channel.shape = (size[1], size[0])
                allchannels.append(channel)

            # create array and transpose dimensions to match tensor style
            exr_arr = np.array(allchannels).transpose((0, 1, 2))
            return exr_arr

        if ndim == 1:
            # transform data to numpy
            channel = np.frombuffer(exr_file.channel('R', pt), dtype=np.float32)
            channel.shape = (size[1], size[0])  # Numpy arrays are (row, col)
            exr_arr = np.array(channel)
            return exr_arr

    def normals_to_rgb(self, normals_to_convert):
        normals_to_convert[normals_to_convert < 0] = 0
        camera_normal_rgb = normals_to_convert
        camera_normal_rgb *= 255
        camera_normal_rgb = camera_normal_rgb.astype(np.uint8)
        return camera_normal_rgb


    def normals_to_rgb_with_negatives(self, normals_to_convert):
        camera_normal_rgb = normals_to_convert + 1
        camera_normal_rgb *= 127.5
        camera_normal_rgb = camera_normal_rgb.astype(np.uint8)
        return camera_normal_rgb

    def get_batch(self):
        # this function get image and segmentation mask
        im_batch = torch.Tensor()
        label_batch = torch.FloatTensor()

        for x in range(self.batchSize):
            self.currIdx = self.currIdx+1
            if  self.currIdx  >= len(self.datalist):
                self.currIdx = 0
                if self.doshuffle:
                   self.shuffle()

            im_path = self.dataroot + self.datalist[self.currIdx][0]
            label_path = self.dataroot + self.datalist[self.currIdx][1]

            # Open pre-processed imgs
            im = Image.open(im_path)
            im = self.transformImage(im)
            im = im.unsqueeze(0)

            # Open pre-processed surface normals
            normals = self.exr_loader(label_path, ndim=3)
            # normals = torch.from_numpy(normals).float().unsqueeze(0)

            # Resize the normals
            normals = normals.transpose(1, 2, 0)
            normals_resized = resize(normals, self.imsize, anti_aliasing=True, clip=True, mode='reflect')
            normals_resized = normals_resized.transpose(2, 0, 1)

            # Normalize the normals
            normals = torch.from_numpy(normals_resized)
            normals = nn.functional.normalize(normals, p=2, dim=0)
            normals = normals.unsqueeze(0)

            im_batch = torch.cat((im_batch, im),0)
            label_batch = torch.cat((label_batch, normals),0)

        return im_batch, label_batch






class Options():
    def __init__(self):
        parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('--dataroot',  default='./data/', help='path to images')
        parser.add_argument('--file_list', default='./data/datalist', help='list of file names in training data')
        parser.add_argument('--batchSize', type=int, default=4, help='input batch size')
        parser.add_argument('--shuffle', type=bool, default=True, help='should the images be shuffled')
        parser.add_argument('--phase', default='train', help='train/eval phase')
        parser.add_argument('--num_epochs', type=int, default=10, help='num of epochs to train')
        parser.add_argument('--imsize', type=int, default=224, help='scale images to this size')
        parser.add_argument('--num_classes', type=int, default=13, help='num of classes in output')
        parser.add_argument('--gpu', default='0', help='which GPU device to use')
        parser.add_argument('--logs_path', default='logs/exp1', help='path of logs to be saved for TensorBoardX')
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
