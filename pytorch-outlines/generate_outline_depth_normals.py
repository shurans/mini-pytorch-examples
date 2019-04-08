import numpy as np
from matplotlib import pyplot as plt
import torch
from torchvision import transforms

#from skimage.transform import resize
from skimage.transform import resize

import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import OpenEXR, Imath
from scipy.misc import imsave
import imageio
from PIL import Image

import sys
sys.path.append('../pytorch-normals/')
import data_processing_script

class OPT():
    def __init__(self):
        self.dataroot = './data/'
        self.file_list = './data/datalist'
        self.batchSize = 32
        self.shuffle = True
        self.phase = 'train'
        self.num_epochs = 500
        self.imsize = 224
        self.num_classes = int(3)
        self.gpu = '0'
        self.logs_path = 'logs/exp9'
        self.use_pretrained = False

opt = OPT()

depth_path = 'data/source-files/depth-imgs/%09d-depth.exr'
path_save_depth_edges = 'data/edges-depth-imgs/%09d-depth-edges.png'
normals_path = './data/source-files/preprocessed-camera-normals/rgb-visualizations/%09d-cameraNormals.png'
path_save_normal_edges = './data/edges-normals-imgs/%09d-normals-edges.png'
path_save_combined_outline = './data/combined-edges/%09d-segmentation.png'
path_save_combined_outline_viz =  './data/combined-edges/viz/%09d-rgb.png'
depth_mask = './data/combined-edges/depth-mask/%09d-rgb.jpg'
allchannels = []
empty_channel = np.zeros((224,224), 'uint8')


def outline_from_depth(depth_img_orig):

    # Apply Laplacian filters for edge detection for depth images
    depth_img_blur = cv2.GaussianBlur(depth_img_orig,(5,5),0)
    edges_lap = cv2.Laplacian(depth_img_blur, cv2.CV_64F, ksize=7, borderType=0 )
    edges_lap = np.absolute(edges_lap).astype(np.uint8)

    # convert to binary and apply mask
    depth_edges = np.zeros(depth_img_orig.shape, dtype = np.uint8)  # edges_lap.copy()
    depth_edges[edges_lap>1] = 255
    depth_edges[edges_lap<=1] = 0

    # Make all depth values greater than 2.5m as 0 (for masking gradients near horizon)
    max_distance_to_object = 2.5
    depth_edges[ depth_img_orig > max_distance_to_object] = 0

    return depth_edges

def outline_from_normal(depth_img_orig_rgb):

    surface_normal_hsv = cv2.cvtColor(depth_img_orig_rgb, cv2.COLOR_BGR2HSV)
    surface_normal_hsv = surface_normal_hsv[:,:,1]
    surface_normal_hsv = cv2.normalize(surface_normal_hsv, None, alpha=0, beta=255,
                                       norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    surface_normal_gray = cv2.GaussianBlur(surface_normal_hsv,(5,5),0)

    # Apply Sobel Filter
    sobelx = cv2.Sobel(surface_normal_gray,cv2.CV_64F,1,0,ksize=1)
    sobely = cv2.Sobel(surface_normal_gray,cv2.CV_64F,0,1,ksize=1)
    sobelxy = sobelx + sobely
    sobelxy = np.uint8(np.absolute(sobelxy))

    # Convert to binary
    edges_sobel_binary = np.zeros((depth_img_orig_rgb.shape[0], depth_img_orig_rgb.shape[1]), dtype = np.uint8)
    edges_sobel_binary[sobelxy > 15] = 255
    edges_sobel_binary[sobelxy <= 15] = 0

    # print('normal nonzero:', np.sum((edges_sobel_binary > 0) & (edges_sobel_binary < 255)))
    return edges_sobel_binary

for i in range (0,100):
    # Load Depth Img convert to outlines and resize
    print('Loading img %d'%(i))
    depth_img_orig = data_processing_script.exr_loader(depth_path%(i), ndim=1)
    depth_edges = outline_from_depth(depth_img_orig)
    depth_edges_img = Image.fromarray(depth_edges, 'L').resize((224,224), resample=Image.NEAREST)

    depth_edges = np.asarray(depth_edges_img)

    # Load RGB image, convert to outlines and  resize
    depth_img_orig_rgb = imageio.imread(normals_path%(i))
    normals_edges = outline_from_normal(depth_img_orig_rgb)
    # edges = Image.fromarray(edges).resize((224,224))

    save_output = True
    if(save_output):
       depth_edges_img.save(path_save_depth_edges%(i))

    save_output = True
    if(save_output):
        imsave( path_save_normal_edges%(i), normals_edges )

    # Depth and Normal outlines should not overlap. Priority given to depth.
    depth_edges = depth_edges.astype(np.uint8)
    normals_edges[depth_edges == 255] = 0

    # modified edges and create mask
    output = np.zeros((224,224), 'uint8')
    output[ normals_edges==255 ] = 2
    output[ depth_edges==255 ] = 1

    # Remove gradient bars from the top and bottom of img
    num_of_rows_to_delete = 2
    output[:num_of_rows_to_delete, :] = 0
    output[-num_of_rows_to_delete:, :] = 0

    img = Image.fromarray(output, 'L')
    img.save(path_save_combined_outline%i)

    # visualization of outline
    rgbArray0 = np.zeros((224,224), 'uint8')
    rgbArray1 = np.zeros((224,224), 'uint8')
    rgbArray2 = np.zeros((224,224), 'uint8')
    rgbArray0[output == 0] = 255
    rgbArray1[output == 1] = 255
    rgbArray2[output == 2] = 255
    rgbArray = np.stack((rgbArray0, rgbArray1, rgbArray2), axis=2)
    img = Image.fromarray(rgbArray, 'RGB')
    img.save(path_save_combined_outline_viz%i)


# print(allchannels)
'''
    display_output = 1
    if(display_output):
        fig1 = plt.figure(figsize=(12,12))
        plt.imshow(depth_img_orig, cmap='gray')
        plt.show()
        fig1 = plt.figure(figsize=(12,12))
        plt.imshow(depth_img_blur, cmap='gray')
        plt.show()
        fig2 = plt.figure(figsize=(12,12))
        plt.imshow(edges_lap, cmap='gray')
        plt.show()
        fig3 = plt.figure(figsize=(12,12))
        plt.imshow(depth_edges, cmap='gray')
        plt.show()
        fig4 = plt.figure(figsize=(12,12))
        plt.imshow(edges, cmap='gray')
        plt.show()
'''
