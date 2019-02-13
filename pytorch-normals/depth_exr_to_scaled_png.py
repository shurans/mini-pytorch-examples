import os
import warnings
from termcolor import colored
import fnmatch
import numpy as np
import OpenEXR
import Imath
import shutil
import glob
import concurrent.futures
import argparse

from PIL import Image
from pathlib import Path
from scipy.misc import imsave

from torch import nn
from sklearn import preprocessing
from skimage.transform import resize

'''
This script takes in a depth image in .exr format. 
Then it cuts out a hole in it and converts it as a scaled png image. Used as input to the depth2depth module.
'''

def exr_loader(EXR_PATH, ndim=3):

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


def main():
    '''It expects a dataset which is a directory containing all the depth.exr files.
    Script converts the exr files to png images
    '''
    parser = argparse.ArgumentParser(
        description='Dataset Directory path')
    parser.add_argument('-p', '--depth-path', required=True,
                        help='Path to directory containing depth images', metavar='path/to/dataset')
    parser.add_argument('-s', '--imsize', help='Path to dataset', type=int, default=224)
    args = parser.parse_args()

    # create a directory for depth scaled png images, if it doesn't exist
    depth_imgs = os.path.join(args.depth_path , 'depth_scaled_png')

    if not os.path.isdir(depth_imgs):
        os.makedirs(depth_imgs)
        print ("    Created dir:", depth_imgs)
    else:
        print("    Already Exists:", depth_imgs)

    # read the exr file as np array, scale it and store as png image
    scale_value = 4000
    print('Converting depth files from exr format to a scaled uin16 png format...')
    print('Will make half of img horizontally zero during conversion to test depth2depth executable')

    for root, dirs, files in os.walk(args.depth_path):
        for filename in sorted(fnmatch.filter(files, '*depth.exr')):
            name = filename[:-4] + '.png'
            np_image = exr_loader(os.path.join(args.depth_path, filename), ndim=1)
            np_image = np_image * scale_value
            np_image = np_image.astype(np.uint16)
            height, width = np_image.shape
            # h_start, h_stop = (height//8)*5, (height//8)*6
            # w_start, w_stop = (width//8)*5, (width//8)*6
            h_start, h_stop = (height//8)*5, (height//8)*6
            w_start, w_stop = (width//8)*5, (width//8)*6

            # Make half the image zero for testing depth2depth
            np_image[h_start:h_stop, w_start:w_stop] = 0

            # Convert to PIL
            array_buffer = np_image.tobytes()
            img = Image.new("I", np_image.T.shape)
            img.frombytes(array_buffer, 'raw', 'I;16')

            # Resize and save
            img = img.resize((args.imsize, args.imsize), Image.ANTIALIAS)
            img.save(os.path.join(depth_imgs, name))
    print('total ', len([name for name in os.listdir(depth_imgs) if os.path.isfile(os.path.join(depth_imgs, name))]), ' converted from exr to png')

if __name__ == "__main__":
    main()
