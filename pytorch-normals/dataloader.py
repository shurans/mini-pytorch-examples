#!/usr/bin/env python3

from __future__ import print_function, division
import os
import glob
from PIL import Image
import Imath
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from imgaug import augmenters as iaa
import imgaug as ia

from utils.utils import exr_loader, exr_saver


class SurfaceNormalsDataset(Dataset):
    """
    Dataset class for training model on estimation of surface normals.
    Uses imgaug for image augmentations
    """

    def __init__(self,
                 input_dir='data/datasets/milk-bottles/resized-files/preprocessed-rgb-imgs',
                 label_dir='data/datasets/milk-bottles/resized-files/preprocessed-camera-normals',
                 transform=None,
                 input_only=None,
                ):
        """
        Args:
            input_dir (str): Path to folder containing the input images.
            label_dir (str): Path to folder containing the labels.
            transform (imgaug transforms): Transforms to be applied to the imgs
            input_only (list, str): List of transforms that are to be applied only to the input img
        """
        super().__init__()

        self.images_dir = input_dir
        self.labels_dir = label_dir
        self.transform = transform
        self.input_only = input_only

        # Create list of filenames
        self._datalist_input = None  # Variable containing list of all input images filenames in dataset
        self._datalist_label = None  # Variable containing list of all ground truth filenames in dataset
        self._extension_input = '.png' # The file extension of input images
        self._extension_label = '.exr' # The file extension of labels
        self._create_lists_filenames(self.images_dir, self.labels_dir)


    def __len__(self):
        return len(self._datalist_input)

    def __getitem__(self, index):

        image_path = self._datalist_input[index]
        label_path = self._datalist_label[index]

        # Open input imgs
        _img = Image.open(image_path).convert('RGB')

        tf = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        _img = tf(_img)

        # Open labels
        _label = exr_loader(label_path, ndim=3)
        _label = torch.from_numpy(_label)

        # Apply image augmentations and convert to Tensor
        if self.transform is not None:
            raise ValueError('Transforms are not supported for now. Because Surface normals not stored as PIL image, cannot apply transforms!')

        # if self.transform:
        #     det_tf = self.transform.to_deterministic()
        #     _img = det_tf.augment_image(_img)
        #     _newlabel = det_tf.augment_image(_newlabel, hooks=ia.HooksImages(activator=self._activator_masks))
        # _img = np.ascontiguousarray(_img) # To prevent errors from negative stride, as caused by fliplr()
        # _img_tensor = transforms.ToTensor()(_img)
        # _newlabel_tensor = transforms.ToTensor()(_newlabel.astype(np.float)) # Without conversion of numpy to float, the numbers get normalized

        return _img, _label

    def _create_lists_filenames(self, images_dir, labels_dir):
        '''Create 2 lists of filenames of images and labels respectively. The indexes
        of both lists match
        '''
        assert os.path.isdir(images_dir), 'This directory does not exist: %s' % (images_dir)
        assert os.path.isdir(labels_dir), 'This directory does not exist: %s' % (labels_dir)

        imageSearchStr = os.path.join(images_dir, '*'+self._extension_input)
        labelSearchStr = os.path.join(labels_dir, '*'+self._extension_label)
        imagepaths = sorted(glob.glob(imageSearchStr))
        labelpaths = sorted(glob.glob(labelSearchStr))

        # Sanity Checks
        numImages = len(imagepaths)
        numLabels = len(labelpaths)
        if numImages == 0:
            raise ValueError('No images found in given directory. Searched for {}'.format(imageSearchStr))
        if numLabels == 0:
            raise ValueError('No labels found in given directory. Searched for {}'.format(imageSearchStr))
        if numImages != numImages:
            raise ValueError('The number of images and labels do not match. Please check data, found {} images and {} labels'.format(
                                numImages, numLabels))

        self._datalist_input = imagepaths
        self._datalist_label = labelpaths


    def _activator_masks(self, images, augmenter, parents, default):
        '''Used with imgaug to help only apply some augmentations to images and not labels
        Eg: Blur is applied to input only, not label. However, resize is applied to both.m hn
        '''
        if self.input_only and augmenter.name in self.input_only:
            return False
        else:
            return default






if __name__ == '__main__':
    import matplotlib.pyplot as plt

    from torch.utils.data import DataLoader
    from torchvision import transforms
    import torchvision

    # Example Augmentations using imgaug
    # imsize = 512
    # augs_train = iaa.Sequential([
    #     # Geometric Augs
    #     iaa.Scale((imsize, imsize), 0), # Resize image
    #     iaa.Fliplr(0.5),
    #     iaa.Flipud(0.5),
    #     iaa.Rot90((0, 4)),
    #     # Blur and Noise
    #     #iaa.Sometimes(0.2, iaa.GaussianBlur(sigma=(0, 1.5), name="gaus-blur")),
    #     #iaa.Sometimes(0.1, iaa.Grayscale(alpha=(0.0, 1.0), from_colorspace="RGB", name="grayscale")),
    #     iaa.Sometimes(0.2, iaa.AdditiveLaplaceNoise(scale=(0, 0.1*255), per_channel=True, name="gaus-noise")),
    #     # Color, Contrast, etc.
    #     #iaa.Sometimes(0.2, iaa.Multiply((0.75, 1.25), per_channel=0.1, name="brightness")),
    #     iaa.Sometimes(0.2, iaa.GammaContrast((0.7, 1.3), per_channel=0.1, name="contrast")),
    #     iaa.Sometimes(0.2, iaa.AddToHueAndSaturation((-20, 20), name="hue-sat")),
    #     #iaa.Sometimes(0.3, iaa.Add((-20, 20), per_channel=0.5, name="color-jitter")),
    # ])
    # augs_test = iaa.Sequential([
    #     # Geometric Augs
    #     iaa.Scale((imsize, imsize), 0),
    # ])



    augs = None # augs_train, augs_test, None
    input_only = None # ["gaus-blur", "grayscale", "gaus-noise", "brightness", "contrast", "hue-sat", "color-jitter"]

    db_test = SurfaceNormalsDataset(
        input_dir='data/datasets/milk-bottles/resized-files/preprocessed-rgb-imgs',
        label_dir='data/datasets/milk-bottles/resized-files/preprocessed-camera-normals',
        transform=augs,
        input_only=input_only
    )

    batch_size = 16
    testloader = DataLoader(db_test, batch_size=batch_size, shuffle=True, num_workers=32, drop_last=True)


    # Show 1 Shuffled Batch of Images
    for ii, batch in enumerate(testloader):
        # Get Batch
        img, label = batch
        print('image shape, type: ', img.shape, img.dtype)
        print('label shape, type: ', label.shape, label.dtype)

        # Show Batch
        sample = torch.cat((img, label), 2)
        im_vis = torchvision.utils.make_grid(sample, nrow=batch_size//4, padding=2, normalize=True, scale_each=True)
        plt.imshow(im_vis.numpy().transpose(1,2,0))
        plt.show()

        break
