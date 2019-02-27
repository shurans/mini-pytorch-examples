#!/usr/bin/env python3

import cv2
import numpy as np
import matplotlib.pyplot as plt
import OpenEXR
import imageio
from PIL import Image
from skimage.transform import resize


from utils import exr_loader, exr_saver

'''This script generates training data for the outlines model.
'''


path_depth = '/home/ganesh/Deep_Learning/google/test-edges/depth-imgs/%09d-depth.exr'
path_surface_normal_rgb = '/home/ganesh/Deep_Learning/google/test-edges/preprocessed-camera-normals/\
                           rgb-visualizations/%09d-cameraNormals.png'

path_edges_output_normals = '/home/ganesh/Deep_Learning/google/test-edges/preprocessed-camera-normals/\
                             edges-normals-imgs/%09d-normals-edges.jpg'
path_edges_output_depth = '/home/ganesh/Deep_Learning/google/test-edges/edges-depth-imgs/%09d-depth-edges.jpg'
path_edges_output_combined = '/home/ganesh/Deep_Learning/google/test-edges/output/%09d-rgb.jpg'

allchannels = []
empty_channel = np.zeros((224, 224), 'uint8')


def outline_from_depth(depth_img_orig):

    depth_img_resized = cv2.GaussianBlur(depth_img_orig, (5, 5), 0)
    # Make all depth values greater than 2.5m as 0 (for masking edge matrix)
    depth_img_mask = depth_img_orig.copy()
    depth_img_mask[depth_img_mask > 2.5] = 0
    depth_img_mask[depth_img_mask > 0] = 1

    # Apply Laplacian filters for edge detection for depth images
    edges_lap = cv2.Laplacian(depth_img_orig, cv2.CV_8U, ksize=7, borderType=0)
    edges_lap_binary = edges_lap.copy()
    # print(edges_lap_binary.max())
    edges_lap_binary[edges_lap_binary > 1] = 255
    edges_lap_binary[edges_lap_binary <= 1] = 0
    edges_lap_binary = edges_lap_binary * depth_img_mask

    return edges_lap_binary


def outline_from_normal(depth_img_orig_rgb):

    depth_img_rgb = cv2.cvtColor(depth_img_orig_rgb, cv2.COLOR_BGR2GRAY)
    depth_img_rgb = cv2.GaussianBlur(depth_img_rgb, (5, 5), 0)

    # Apply Canny filter to RGB image_files_rgb
    edges = cv2.Canny(depth_img_rgb, 50, 100)

    return edges


for i in range(0, 100):
    # Load Depth Img and Apply Blur
    print('Loading img %d' % (i))
    depth_img_orig = exr_loader(path_depth % (i), ndim=1)
    edges_lap_binary = outline_from_depth(depth_img_orig)
    edges_lap_binary = resize(edges_lap_binary, (224, 224), anti_aliasing=True, clip=True, mode='reflect')

    # Load RGB image and Apply GaussianBlur
    depth_img_orig_rgb = imageio.imread(path_surface_normal_rgb % (i))
    edges = outline_from_normal(depth_img_orig_rgb)

    save_output = True
    if(save_output):
        imageio.imwrite(path_edges_output_depth % (i), edges_lap_binary)

    save_output = True
    if(save_output):
        imageio.imwrite(path_edges_output_normals % (i), edges)

    edges = edges.astype(np.uint8)
    edges_lap_binary = edges_lap_binary.astype(np.uint8)

    rgbArray = np.zeros((224, 224, 3), 'uint8')
    rgbArray[..., 0] = empty_channel
    rgbArray[..., 1] = edges
    rgbArray[..., 2] = edges_lap_binary
    img = Image.fromarray(rgbArray)
    img.save(path_edges_output_combined % i)

    red = empty_channel
    green = edges
    blue = edges_lap_binary
    image_convert = np.array([red, green, blue])
    allchannels.append(image_convert)

print(allchannels)
