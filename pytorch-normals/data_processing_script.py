#!/usr/bin/env python3

import os
import warnings
from termcolor import colored
import fnmatch
import argparse
import numpy as np
import OpenEXR
import Imath
import json
import shutil
import glob
import csv
import concurrent.futures

from PIL import Image
from pathlib import Path
from scipy.misc import imsave

import torch
import torchvision
from torchvision import transforms, utils
from torch import nn
from sklearn import preprocessing
from skimage.transform import resize


SUBFOLDER_MAP = {
    'rgb-files':        {'postfix': '-rgb.jpg',
                         'folder-name': 'rgb-imgs'},
    'depth-files':      {'postfix': '-depth.exr',
                         'folder-name': 'depth-imgs'},
    'json-files':       {'postfix': '-masks.json',
                         'folder-name': 'json-files'},
    'world-normals':    {'postfix': '-normals.exr',
                         'folder-name': 'world-normals'},
    'variant-masks':    {'postfix': '-variantMasks.exr',
                         'folder-name': 'variant-masks'},
    'component-masks':  {'postfix': '-componentMasks.exr',
                         'folder-name': 'component-masks'},
    'camera-normals':   {'postfix': '-cameraNormals.npy',
                         'folder-name': 'camera-normals'},
    'camera-normals-rgb':  {'postfix': '-cameraNormals.png',
                            'folder-name': 'camera-normals/rgb-visualizations'},
}

NEW_DATASET_PATHS = {
    'root': 'data',
    'source-files': 'source-files',
    'training-data': 'train',
}

SUBFOLDER_MAP_TRAIN = {
    'preprocessed-rgb-imgs':  {'postfix': '-rgb.npy',
                               'folder-name': 'preprocessed-rgb-imgs'},
    'preprocessed-camera-normals':  {'postfix': '-cameraNormals.npy',
                                     'folder-name': 'preprocessed-camera-normals'},
    'preprocessed-camera-normals-viz':  {'postfix': '-cameraNormals.png',
                                         'folder-name': 'preprocessed-camera-normals/rgb-visualizations'},
    'preprocessed-rgb-imgs-viz':  {'postfix': '-rgb.png',
                                   'folder-name': 'preprocessed-rgb-imgs/rgb-visualizations'},
}


################################ RENAME AND MOVE ################################
def scene_prefixes(dataset_path):
    '''Returns a list of prefixes of all the json files present in dataset
    Eg, if our file is named 000000234-rgb.jpb, prefix is '000000234'

    Every set of images in dataset will contain 1 masks.json file, hence we can count just the json file.

    Args:
        dataset_path (str): Path to dataset containing all the new files.

    Returns:
        None
    '''
    dataset_prefixes = []
    for root, dirs, files in os.walk(dataset_path):
        # one mask json file per scene so we can get the prefixes from them
        json_filename = SUBFOLDER_MAP['json-files']['postfix']
        for filename in fnmatch.filter(files, '*' + json_filename):
            dataset_prefixes.append(filename[0:0 - len(json_filename)])
    dataset_prefixes.sort()
    return dataset_prefixes


def string_prefixes_to_sorted_ints(prefixes_list):
    unsorted = list(map(lambda x: int(x), prefixes_list))
    unsorted.sort()
    return unsorted


def move_and_rename_dataset(old_dataset_path, new_dataset_path, initial_value):
    '''All files are moved to new dir and renamed such that their prefix begins from the provided initial value.
    This helps in adding a dataset to previously existing dataset.

    Args:
        old_dataset_path (str): Path to dataset containing all the new files.
        new_dataset_path (str): Path to new dataset to which renamed files will be moved to.
        initial_value (int): Value from which new numbering will start.

    Returns:
        count_renamed (int): Number of files that were renamed.
    '''
    sorted_ints = string_prefixes_to_sorted_ints(scene_prefixes(old_dataset_path))

    count_renamed = 0
    for i in range(len(sorted_ints)):
        old_prefix_str = "{:09}".format(sorted_ints[i])
        new_prefix_str = "{:09}".format(initial_value + i)
        print("\tMoving files with prefix", old_prefix_str, "to", new_prefix_str)

        for root, dirs, files in os.walk(old_dataset_path):
            for filename in fnmatch.filter(files, (old_prefix_str + '*')):
                os.rename(os.path.join(old_dataset_path, filename), os.path.join(
                    new_dataset_path, filename.replace(old_prefix_str, new_prefix_str)))
                count_renamed += 1

    return count_renamed


def move_to_subfolders(dataset_path):
    '''Move each file type to it's own subfolder.
    It will create a folder for each file type. The file type is determined from it's postfix.
    The file types and their corresponding directory are defined in the SUBFOLDER_MAP dict

    Args:
        dataset_path (str): Path to dataset containing all the files.

    Returns:
        None
    '''
    for filetype in SUBFOLDER_MAP:
        subfolder_path = os.path.join(dataset_path, SUBFOLDER_MAP[filetype]['folder-name'])

        if not os.path.isdir(subfolder_path):
            os.makedirs(subfolder_path)
            print ("\tCreated dir:", subfolder_path)
        else:
            print("\tAlready Exists:", subfolder_path)

    for filetype in SUBFOLDER_MAP:
        file_postfix = SUBFOLDER_MAP[filetype]['postfix']
        subfolder = SUBFOLDER_MAP[filetype]['folder-name']

        count_files_moved = 0
        files = os.listdir(dataset_path)
        for filename in fnmatch.filter(files, '*' + file_postfix):
            shutil.move(os.path.join(dataset_path, filename), os.path.join(dataset_path, subfolder))
            count_files_moved += 1
        if count_files_moved > 0:
            color = 'green'
        else:
            color = 'red'
        print ("\tMoved", colored(count_files_moved, color), "files to dir:", subfolder)


################################ WORLD TO CAMERA SPACE ################################
##
# q: quaternion
# v: 3-element array
# @see adapted from blender's math_rotation.c
#
# \note:
# Assumes a unit quaternion?
#
# in fact not, but you may want to use a unit quat, read on...
#
# Shortcut for 'q v q*' when \a v is actually a quaternion.
# This removes the need for converting a vector to a quaternion,
# calculating q's conjugate and converting back to a vector.
# It also happens to be faster (17+,24* vs * 24+,32*).
# If \a q is not a unit quaternion, then \a v will be both rotated by
# the same amount as if q was a unit quaternion, and scaled by the square of
# the length of q.
#
# For people used to python mathutils, its like:
# def mul_qt_v3(q, v): (q * Quaternion((0.0, v[0], v[1], v[2])) * q.conjugated())[1:]
#
# \note: multiplying by 3x3 matrix is ~25% faster.
##
def _multiply_quaternion_vec3(q, v):
    t0 = -q[1] * v[0] - q[2] * v[1] - q[3] * v[2]
    t1 = q[0] * v[0] + q[2] * v[2] - q[3] * v[1]
    t2 = q[0] * v[1] + q[3] * v[0] - q[1] * v[2]
    i = [t1, t2, q[0] * v[2] + q[1] * v[1] - q[2] * v[0]]
    t1 = t0 * -q[1] + i[0] * q[0] - i[1] * q[3] + i[2] * q[2]
    t2 = t0 * -q[2] + i[1] * q[0] - i[2] * q[1] + i[0] * q[3]
    i[2] = t0 * -q[3] + i[2] * q[0] - i[0] * q[2] + i[1] * q[1]
    i[0] = t1
    i[1] = t2
    return i


def world_to_camera_normals(inverted_camera_quaternation, exr_x, exr_y, exr_z):
    camera_normal = np.empty([exr_x.shape[0], exr_x.shape[1], 3], dtype=np.float32)
    for i in range(exr_x.shape[0]):
        for j in range(exr_x.shape[1]):
            pixel_camera_normal = _multiply_quaternion_vec3(
                inverted_camera_quaternation,
                [exr_x[i][j], exr_y[i][j], exr_z[i][j]]
            )
            camera_normal[i][j][0] = pixel_camera_normal[0]
            camera_normal[i][j][1] = pixel_camera_normal[1]
            camera_normal[i][j][2] = pixel_camera_normal[2]
    return camera_normal


def normal_to_rgb(normals_to_convert):
    camera_normal_rgb = normals_to_convert + 1
    camera_normal_rgb *= 127.5
    camera_normal_rgb = camera_normal_rgb.astype(np.uint8)
    return camera_normal_rgb


# Return X, Y, Z normals as numpy arrays
def read_exr_normal_file(exr_path):
    exr_file = OpenEXR.InputFile(exr_path)
    # print("exr header:")
    # print(exr_file.header())
    cm_dw = exr_file.header()['dataWindow']
    exr_x = np.fromstring(
        exr_file.channel('R', Imath.PixelType(Imath.PixelType.HALF)),
        dtype=np.float16
    )
    exr_x.shape = (cm_dw.max.y - cm_dw.min.y + 1, cm_dw.max.x - cm_dw.min.x + 1)  # rows, cols
    exr_y = np.fromstring(
        exr_file.channel('G', Imath.PixelType(Imath.PixelType.HALF)),
        dtype=np.float16
    )
    exr_y.shape = (cm_dw.max.y - cm_dw.min.y + 1, cm_dw.max.x - cm_dw.min.x + 1)  # rows, cols

    exr_z = np.fromstring(
        exr_file.channel('B', Imath.PixelType(Imath.PixelType.HALF)),
        dtype=np.float16
    )
    exr_z.shape = (cm_dw.max.y - cm_dw.min.y + 1, cm_dw.max.x - cm_dw.min.x + 1)  # rows, cols
    return exr_x, exr_y, exr_z


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

def exr_saver(EXR_PATH, ndarr):
    '''Saves a numpy array as an EXR file with HALF precision (float16)
    Args:
        EXR_PATH (str): The path to which file will be saved
        ndarr (ndarray): A numpy array of shape (3 x height x width)
    
    Return:
        None
    '''
    # Convert each channel to strings
    Rs = ndarr[0,:,:].astype(np.float16).tostring()
    Gs = ndarr[1,:,:].astype(np.float16).tostring()
    Bs = ndarr[2,:,:].astype(np.float16).tostring()

    # Write the three color channels to the output file
    HEADER = OpenEXR.Header(ndarr.shape[2], ndarr.shape[1])
    half_chan = Imath.Channel(Imath.PixelType(Imath.PixelType.HALF))
    HEADER['channels'] = dict([(c, half_chan) for c in "RGB"])
    
    out = OpenEXR.OutputFile(EXR_PATH, HEADER)
    out.writePixels({'R' : Rs, 'G' : Gs, 'B' : Bs })
    out.close()


def preprocess_world_to_cam(world_normals, json_files):
    '''Will convert normals from World co-ords to Camera co-ords
    It will create a folder to store converted files. A quaternion for conversion of normal from world to camera
    co-ords is read from the json file and is multiplied with each normal in source file.

    Args:
        world_normals (str): Path to world co-ord normals file.
        json_files (str): Path to json file which stores quaternion.

    Returns:
        bool: False if file exists and it skipped it. True if it converted the file.
    '''
    #  Output paths and filenames
    camera_normal_dir_path = os.path.join(NEW_DATASET_PATHS['root'], NEW_DATASET_PATHS['source-files'],
                                          SUBFOLDER_MAP['camera-normals']['folder-name'])
    camera_normal_rgb_dir_path = os.path.join(NEW_DATASET_PATHS['root'], NEW_DATASET_PATHS['source-files'],
                                              SUBFOLDER_MAP['camera-normals-rgb']['folder-name'])

    prefix = os.path.basename(world_normals)[0:0 - len(SUBFOLDER_MAP['world-normals']['postfix'])]
    output_camera_normal_filename = (prefix + SUBFOLDER_MAP['camera-normals']['postfix'])
    camera_normal_rgb_filename = (prefix + SUBFOLDER_MAP['camera-normals-rgb']['postfix'])
    output_camera_normal_file = os.path.join(camera_normal_dir_path, output_camera_normal_filename)
    camera_normal_rgb_file = os.path.join(camera_normal_rgb_dir_path, camera_normal_rgb_filename)

    # If cam normal already exists, skip
    if Path(output_camera_normal_file).is_file():
        print('  Skipping {}, it already exists'.format(os.path.join(SUBFOLDER_MAP['camera-normals']['folder-name'], output_camera_normal_filename)))
        return False

    world_normal_file = os.path.join(SUBFOLDER_MAP['world-normals']['folder-name'], os.path.basename(world_normals))
    camera_normal_file = os.path.join(
        SUBFOLDER_MAP['camera-normals']['folder-name'], (prefix + SUBFOLDER_MAP['camera-normals']['postfix']))
    print("  Converting {} to {}".format(world_normal_file, camera_normal_file))

    # Read EXR File
    exr_x, exr_y, exr_z = read_exr_normal_file(world_normals)
    assert(exr_x.shape == exr_y.shape)
    assert(exr_y.shape == exr_z.shape)

    # Read Camera's Inverse Quaternion
    json_file = open(json_files)
    data = json.load(json_file)
    inverted_camera_quaternation = np.asarray(
        data['camera']['world_pose']['rotation']['inverted_quaternion'], dtype=np.float32)

    # Convert Normals to Camera Space
    camera_normal = world_to_camera_normals(inverted_camera_quaternation, exr_x, exr_y, exr_z)
    camera_normal2 = camera_normal.copy()

    # Output Converted EXR Files as numpy data
    exr_arr = np.array(camera_normal).transpose((2, 0, 1))
    np.save(output_camera_normal_file, exr_arr)

    # Output converted Normals as RGB images
    camera_normal_rgb = normal_to_rgb(camera_normal2)
    imsave(camera_normal_rgb_file, camera_normal_rgb)

    return True


################################ PREPROCESS FOR MODEL ################################
def transformImage(im, imsize):
    """Pytorch func to Resize and Normalize images.

    Args:
        im (numpy array): 3 channel Numpy array containing image to be resized.

    Returns:
        numpy array: Converted image as Pytorch tensor.

    """
    transform_list = []
    transform_list.append(transforms.Resize([imsize, imsize]))
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    tf = transforms.Compose(transform_list)
    im = tf(im)
    return im


def preprocess_normals(input):
    """Resize and Normalize Normals and save as Numpy array.

    Args:
        normals_path (str): The path to a numpy array (.npy) that contains the normal to be converted.

    Returns:
        bool: False if file exists and it skipped it. True if it converted the file.

    """
    normals_path, normsize = input

    prefix = os.path.basename(normals_path)[0:0 - len(SUBFOLDER_MAP['camera-normals']['postfix'])]

    preprocess_normals_dir = os.path.join(NEW_DATASET_PATHS['root'], NEW_DATASET_PATHS['training-data'],
                                                 SUBFOLDER_MAP_TRAIN['preprocessed-camera-normals']['folder-name'])
    preprocess_normal_viz_dir = os.path.join(NEW_DATASET_PATHS['root'], NEW_DATASET_PATHS['training-data'],
                                              SUBFOLDER_MAP_TRAIN['preprocessed-camera-normals-viz']['folder-name'])

    preprocess_normals_filename = prefix + SUBFOLDER_MAP_TRAIN['preprocessed-camera-normals']['postfix']
    preprocess_normal_viz_filename = prefix + SUBFOLDER_MAP_TRAIN['preprocessed-camera-normals-viz']['postfix']

    output_file = os.path.join(preprocess_normals_dir, preprocess_normals_filename)
    output_rgb_file = os.path.join(preprocess_normal_viz_dir, preprocess_normal_viz_filename)

    if Path(output_file).is_file():  # file exists
        print("    Skipping {}, it already exists".format(os.path.join(SUBFOLDER_MAP_TRAIN['preprocessed-camera-normals']['folder-name'],
                                                                       preprocess_normals_filename)))
        return False

    # print('    Converting {}'.format(normals_path))
    normals = np.load(normals_path)

    # Resize the normals
    normals = normals.transpose(1, 2, 0)
    normals_resized = resize(normals, (normsize, normsize), anti_aliasing=True, clip=True, mode='reflect')
    normals_resized = normals_resized.transpose(2, 0, 1)

    # Normalize the normals
    normals = torch.from_numpy(normals_resized)
    normal_vectors_norm = nn.functional.normalize(normals, p=2, dim=0)
    normals = normal_vectors_norm.numpy()

    # Save array
    np.save(output_file, normals)
    print('    saved', output_file)

    # Output converted Normals as RGB images
    camera_normal_rgb = normal_to_rgb(normals.transpose(1, 2, 0))
    imsave(output_rgb_file, camera_normal_rgb)

    return True


def preprocess_rgb(input):
    """Resize and Normalize RGB jpeg files and save as Numpy array.

    Args:
        im_path (str): The path to a jpg image. This need not be an absolute path.

    Returns:
        bool: False if file exists and it skipped it. True if it converted the file.

    """
    im_path, imsize = input

    prefix = os.path.basename(im_path)[0:0 - len(SUBFOLDER_MAP['rgb-files']['postfix'])]

    preprocess_rgb_dir = os.path.join(NEW_DATASET_PATHS['root'], NEW_DATASET_PATHS['training-data'],
                                             SUBFOLDER_MAP_TRAIN['preprocessed-rgb-imgs']['folder-name'])
    preprocess_rgb_viz_dir = os.path.join(NEW_DATASET_PATHS['root'], NEW_DATASET_PATHS['training-data'],
                                              SUBFOLDER_MAP_TRAIN['preprocessed-rgb-imgs-viz']['folder-name'])


    preprocess_rgb_filename = prefix + SUBFOLDER_MAP_TRAIN['preprocessed-rgb-imgs']['postfix']
    preprocess_rgb_viz_filename = prefix + SUBFOLDER_MAP_TRAIN['preprocessed-rgb-imgs-viz']['postfix']

    output_file = os.path.join(preprocess_rgb_dir, preprocess_rgb_filename)
    output_rgb_file = os.path.join(preprocess_rgb_viz_dir, preprocess_rgb_viz_filename)

    if Path(output_file).is_file():  # file exists
        print("    Skipping {}, it already exists".format(os.path.join(SUBFOLDER_MAP_TRAIN['preprocessed-rgb-imgs']['folder-name'],
                                                                       preprocess_rgb_filename)))
        return False

    # Open Image, transform and save
    im = Image.open(im_path).convert("RGB")
    im = transformImage(im, imsize)
    im = im.numpy()
    np.save(output_file, im)
    print('    saved', output_file)

    # Output converted RGB numpy arrays as RGB images
    imsave(output_rgb_file, im.transpose(1, 2, 0))

    return True


def main():
    '''Pre-Processes provided dataset for Google Brain Transparent Object Project.
    It expects a dataset which is a directory containing all the files. The expected naming of the files is set
    as postfix in the SUBFOLDER_MAP dict.

    Note: In a file named '000000020-rgb.jpg' its prefix is '000000020' and its postfix '-rgb.jpg'

    Requires Python > 3.2
    '''
    parser = argparse.ArgumentParser(
        description='Rearrange non-contiguous scenes in a dataset, move to separate folders.')
    parser.add_argument('--p', required=True,
                        help='Path to dataset', metavar='path/to/dataset')
    parser.add_argument('--n', default=0,
                        help='The initial value from which the numbering of renamed files must start')
    parser.add_argument('--imsize', default=224, help='The size to which input will be resized to')
    args = parser.parse_args()
    args.imsize = int(args.imsize)
    args.n = int(args.n)


    # Check if source dir is valid
    src_dir_path = os.path.join(NEW_DATASET_PATHS['root'], NEW_DATASET_PATHS['source-files'])
    if not os.path.isdir(src_dir_path):
        if not os.path.isdir(args.p):
            print(colored('ERROR: Did not find {}. Please pass correct path to dataset'.format(args.p), 'red'))
            exit()
        if not os.listdir(args.p):
            print(colored('ERROR: Empty dir {}. Please pass correct path to dataset'.format(args.p), 'red'))
            exit()
    else:
        if not os.path.isdir(args.p):
            print(colored("\nWARNING: Dir {} does not exist. However, found {}.".format(args.p, src_dir_path), 'red'))
            print(colored("Skipping the renaming of files from new dataset {} and proceeding to process file in {}".format(
                args.p, src_dir_path), 'red'))
        else:
            if not os.listdir(args.p):
                print(colored("\nWARNING: Dir {} is empty. However, found {}.".format(args.p, src_dir_path), 'red'))
                print(colored("Skipping the renaming of files from new dataset {} and proceeding to process file in {}".format(
                    args.p, src_dir_path), 'red'))

    # Create new dir to store processed dataset
    if not os.path.isdir(src_dir_path):
        os.makedirs(src_dir_path)
        print ("\nCreated dirs to store new dataset:", src_dir_path)
    else:
        print("\nDataset dir exists:", src_dir_path)

    print("Moving files to", src_dir_path, "and renaming them to start from prefix {:09}.".format(args.n))
    count_renamed = move_and_rename_dataset(args.p, src_dir_path, int(args.n))
    if(count_renamed > 0):
        color = 'green'
    else:
        color = 'red'
    print(colored("Renamed {} files".format(count_renamed), color))

    print("\nSeparating dataset into folders.")
    move_to_subfolders(src_dir_path)

    # Create a pool of processes. By default, one is created for each CPU in your machine.
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Get a list of files to process
        world_normals_path = os.path.join(src_dir_path, SUBFOLDER_MAP['world-normals']['folder-name'])
        json_files_path = os.path.join(src_dir_path, SUBFOLDER_MAP['json-files']['folder-name'])
        # TODO: convert this into SUBFOLDER_MAP vars
        world_normals = sorted(glob.glob(os.path.join(world_normals_path, "*" +
                                                      SUBFOLDER_MAP['world-normals']['postfix'])))
        json_files = sorted(glob.glob(os.path.join(json_files_path, "*" + SUBFOLDER_MAP['json-files']['postfix'])))

        # Process the list of files, but split the work across the process pool to use all CPUs!
        print("\n\nCheck your CPU usage...Converting World co-ord Normals to Camera co-ord Normals!!")
        num_converted, num_skipped = 0, 0
        for converted_file in executor.map(preprocess_world_to_cam, world_normals, json_files):
            if converted_file == True:
                num_converted += 1
            if converted_file == False:
                num_skipped += 1

        print(colored('\n  Converted {} world-normals'.format(num_converted), 'green'))
        print(colored('  Skipped {} world-normals'.format(num_skipped), 'red'))

    # Create dir to store training data
    print("\n\nPre-Processing data - this will be directly used as training data by model")
    train_dir_path = os.path.join(NEW_DATASET_PATHS['root'], NEW_DATASET_PATHS['training-data'])

    for filetype in SUBFOLDER_MAP_TRAIN:
        subfolder_path = os.path.join(train_dir_path, SUBFOLDER_MAP_TRAIN[filetype]['folder-name'])

        if not os.path.isdir(subfolder_path):
            os.makedirs(subfolder_path)
            print ("    Created dir:", subfolder_path)
        else:
            print("    Already Exists:", subfolder_path)

    print("\n")

    # Create a pool of processes. By default, one is created for each CPU in your machine.
    with concurrent.futures.ProcessPoolExecutor(1) as executor:
        # Get a list of files to process
        rgb_imgs_path = os.path.join(src_dir_path, SUBFOLDER_MAP['rgb-files']['folder-name'])
        camera_normals_path = os.path.join(src_dir_path, SUBFOLDER_MAP['camera-normals']['folder-name'])

        image_files_rgb = glob.glob(os.path.join(rgb_imgs_path, "*" + SUBFOLDER_MAP['rgb-files']['postfix']))
        image_files_normals = glob.glob(os.path.join(camera_normals_path, "*" +
                                                     SUBFOLDER_MAP['camera-normals']['postfix']))

        # Process the list of files, but split the work across the process pool to use all CPUs!
        # rgb files
        input = [(image, args.imsize) for image in sorted(image_files_rgb)]
        num_converted, num_skipped = 0, 0
        for converted_file in executor.map(preprocess_rgb, input):
            if converted_file == True:
                num_converted += 1
            if converted_file == False:
                num_skipped += 1
        print(colored('\n  Pre-processed {} rgb files'.format(num_converted), 'green'))
        print(colored('  Skipped {} rgb files\n'.format(num_skipped), 'red'))

        # surface normal files
        input = [(image, args.imsize) for image in sorted(image_files_normals)]
        num_converted, num_skipped = 0, 0
        for converted_file in executor.map(preprocess_normals, input):
            if converted_file == True:
                num_converted += 1
            if converted_file == False:
                num_skipped += 1
        print(colored('\n  Pre-processed {} camera-normal files'.format(num_converted), 'green'))
        print(colored('  Skipped {} camera-normal files\n'.format(num_skipped), 'red'))


if __name__ == "__main__":
    main()
