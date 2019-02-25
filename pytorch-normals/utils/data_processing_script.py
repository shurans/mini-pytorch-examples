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
import concurrent.futures
import time

from PIL import Image
from pathlib import Path
#from scipy.misc import imsave
from imageio import imsave

import torch
import torchvision
from torchvision import transforms, utils
from torch import nn
from skimage.transform import resize

from .utils import exr_loader, exr_saver

SUBFOLDER_MAP = {
    'rgb-files'         :{'postfix'      : '-rgb.jpg',
                          'folder-name'  : 'rgb-imgs'},

    'depth-files'       :{'postfix'      : '-depth.exr',
                          'folder-name'  : 'depth-imgs'},

    'json-files'        :{'postfix'      : '-masks.json',
                          'folder-name'  : 'json-files'},

    'world-normals'     :{'postfix'      : '-normals.exr',
                          'folder-name'  : 'world-normals'},

    'variant-masks'     :{'postfix'      : '-variantMasks.exr',
                          'folder-name'  : 'variant-masks'},

    'component-masks'   :{'postfix'      : '-componentMasks.exr',
                          'folder-name'  : 'component-masks'},

    'camera-normals'    :{'postfix'      : '-cameraNormals.exr',
                          'folder-name'  : 'camera-normals'},

    'camera-normals-rgb':{'postfix'      : '-cameraNormals.png',
                          'folder-name'  : 'camera-normals/rgb-visualizations'},
}

SUBFOLDER_MAP_TEST = {
    'rgb-files'         :{'postfix'      : '-rgb.jpg',
                          'folder-name'  : 'rgb-imgs'},

    'depth-files'       :{'postfix'      : '-depth.npy',
                          'folder-name'  : 'depth-imgs'},
}

NEW_DATASET_PATHS = {
    'root'          : '../data/dataset',
    'source-files'  : 'source-files',
    'training-data' : 'resized-files',
}

SUBFOLDER_MAP_RESIZED = {
    'preprocessed-camera-normals'       :{'postfix': '-cameraNormals.exr',
                                          'folder-name': 'preprocessed-camera-normals'},

    'preprocessed-camera-normals-viz'   :{'postfix': '-cameraNormals.png',
                                          'folder-name': 'preprocessed-camera-normals/rgb-visualizations'},

    'preprocessed-rgb-imgs'             :{'postfix': '-rgb.png',
                                          'folder-name': 'preprocessed-rgb-imgs'},
}

SUBFOLDER_MAP_RESIZED_TEST = {
    'preprocessed-rgb-imgs'             :{'postfix': '-rgb.png',
                                          'folder-name': 'preprocessed-rgb-imgs'},
}


################################ RENAME AND MOVE ################################
def scene_prefixes(dataset_path):
    '''Returns a list of prefixes of all the rgb files present in dataset
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
        rgb_filename = SUBFOLDER_MAP['rgb-files']['postfix']
        for filename in fnmatch.filter(files, '*' + rgb_filename):
            dataset_prefixes.append(filename[0:0 - len(rgb_filename)])
        break
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
    prefixes_str = scene_prefixes(old_dataset_path)
    sorted_ints = string_prefixes_to_sorted_ints(prefixes_str)

    count_renamed = 0
    for i in range(len(sorted_ints)):
        old_prefix_str = "{:09}".format(sorted_ints[i])
        new_prefix_str = "{:09}".format(initial_value + i)
        print("\tMoving files with prefix", old_prefix_str, "to", new_prefix_str)

        for root, dirs, files in os.walk(old_dataset_path):
            for filename in fnmatch.filter(files, (old_prefix_str + '*')):
                shutil.move(os.path.join(old_dataset_path, filename), os.path.join(
                    new_dataset_path, filename.replace(old_prefix_str, new_prefix_str)))
                count_renamed += 1
            break

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
    '''Converts a surface normals array into an RGB image.
    Surface normals are represented in a range of (-1,1),
    This is converted to a range of (0,255) to be written
    into an image.
    The surface normals are normally in camera co-ords, 
    with positive z axis coming out of the page. And the axes are
    mapped as (x,y,z) -> (R,G,B).
    '''
    camera_normal_rgb = normals_to_convert + 1
    camera_normal_rgb *= 127.5
    camera_normal_rgb = camera_normal_rgb.astype(np.uint8)
    return camera_normal_rgb





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
    exr_np = exr_loader(world_normals)
    exr_x, exr_y, exr_z = exr_np[0], exr_np[1], exr_np[2]
    assert(exr_x.shape == exr_y.shape)
    assert(exr_y.shape == exr_z.shape)

    # Read Camera's Inverse Quaternion
    json_file = open(json_files)
    data = json.load(json_file)
    inverted_camera_quaternation = np.asarray(
        data['camera']['world_pose']['rotation']['inverted_quaternion'], dtype=np.float32)

    # Convert Normals to Camera Space
    camera_normal = world_to_camera_normals(inverted_camera_quaternation, exr_x, exr_y, exr_z)
    # camera_normal2 = camera_normal.copy()

    # Output Converted EXR Files as numpy data
    # exr_arr = np.array(camera_normal).transpose((2, 0, 1))
    # np.save(output_camera_normal_file, exr_arr)

    # Output Converted EXR Files
    exr_arr = camera_normal.transpose((2, 0, 1))
    exr_saver(output_camera_normal_file, exr_arr, ndim=3)

    # Output converted Normals as RGB images
    camera_normal_rgb = normal_to_rgb(camera_normal)
    imsave(camera_normal_rgb_file, camera_normal_rgb)

    return True


################################ PREPROCESS FOR MODEL ################################
def preprocess_normals(input):
    """Resize and Normalize Normals and save as Numpy array.

    Args:
        normals_path (str): 
        (normals_path, (height, width)) (tuple): normals_path (str)     = The path to an exr file that contains the normal to be preprocessed.
                                                 (height, width)  (int) = The size to which image is to be resized.
    Returns:
        bool: False if file exists and it skipped it. True if it converted the file.

    """
    normals_path, imsize = input

    if len(imsize) != 2:
        raise ValueError('Pass imsize as a tuple of (height, width). Given imsize = {}'.format(imsize))

    prefix = os.path.basename(normals_path)[0:0 - len(SUBFOLDER_MAP['camera-normals']['postfix'])]

    preprocess_normals_dir = os.path.join(NEW_DATASET_PATHS['root'], NEW_DATASET_PATHS['training-data'],
                                                 SUBFOLDER_MAP_RESIZED['preprocessed-camera-normals']['folder-name'])
    preprocess_normal_viz_dir = os.path.join(NEW_DATASET_PATHS['root'], NEW_DATASET_PATHS['training-data'],
                                              SUBFOLDER_MAP_RESIZED['preprocessed-camera-normals-viz']['folder-name'])

    preprocess_normals_filename = prefix + SUBFOLDER_MAP_RESIZED['preprocessed-camera-normals']['postfix']
    preprocess_normal_viz_filename = prefix + SUBFOLDER_MAP_RESIZED['preprocessed-camera-normals-viz']['postfix']

    output_file = os.path.join(preprocess_normals_dir, preprocess_normals_filename)
    output_rgb_file = os.path.join(preprocess_normal_viz_dir, preprocess_normal_viz_filename)

    if Path(output_file).is_file():  # file exists
        print("    Skipping {}, it already exists".format(os.path.join(SUBFOLDER_MAP_RESIZED['preprocessed-camera-normals']['folder-name'],
                                                                       preprocess_normals_filename)))
        return False

    # print('    Converting {}'.format(normals_path))
    normals = exr_loader(normals_path, ndim=3)

    # Resize the normals
    normals = normals.transpose(1, 2, 0)
    normals_resized = resize(normals, imsize, anti_aliasing=True, clip=True, mode='reflect')
    normals_resized = normals_resized.transpose(2, 0, 1)

    # Normalize the normals
    normals = torch.from_numpy(normals_resized)
    normals = nn.functional.normalize(normals, p=2, dim=0)
    normals = normals.numpy()

    # # Save array as numpy file
    # np.save(output_file, normals)
    # print('    saved', output_file)

    # Save array as EXR file
    exr_saver(output_file, normals, ndim=3)
    print('    saved', output_file)

    # Output converted Normals as RGB images
    camera_normal_rgb = normal_to_rgb(normals.transpose(1, 2, 0))
    imsave(output_rgb_file, camera_normal_rgb)

    return True


def preprocess_rgb(input):
    """Resize and Normalize RGB jpeg files and save as Numpy array.

    Args:
        (im_path, (height, width)) (tuple): im_path (str)          = The path to a jpg image. This need not be an absolute path.
                                            (height, width)  (int) = The size to which image is to be resized.

    Returns:
        bool: False if file exists and it skipped it. True if it converted the file.

    """
    im_path, imsize = input

    if len(imsize) != 2:
        raise ValueError('Pass imsize as a tuple of (height, width). Given imsize = {}'.format(imsize))

    prefix = os.path.basename(im_path)[0:0 - len(SUBFOLDER_MAP['rgb-files']['postfix'])]

    preprocess_rgb_dir = os.path.join(NEW_DATASET_PATHS['root'], NEW_DATASET_PATHS['training-data'],
                                             SUBFOLDER_MAP_RESIZED['preprocessed-rgb-imgs']['folder-name'])
    preprocess_rgb_filename = prefix + SUBFOLDER_MAP_RESIZED['preprocessed-rgb-imgs']['postfix']
    
    output_file = os.path.join(preprocess_rgb_dir, preprocess_rgb_filename)
    

    if Path(output_file).is_file():  # file exists
        print("    Skipping {}, it already exists".format(os.path.join(SUBFOLDER_MAP_RESIZED['preprocessed-rgb-imgs']['folder-name'],
                                                                       preprocess_rgb_filename)))
        return False

    # Open Image, transform and save
    im = Image.open(im_path).convert("RGB")

    tf = transforms.Compose([
        transforms.Resize(imsize, interpolation = Image.BILINEAR),    #TODO: RESIZE INTO 16:9 ASPECT RATIO. ACCEPT 2 INPUTS FOR IMSIZE, OR TUPLE
        #transforms.ToTensor(), #saving back as image, this not needed.
        #transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    im = tf(im)

    im = np.array(im)
    
    
    # Output converted RGB numpy arrays as RGB images
    imsave(output_file, im)
    print('    saved', output_file)

    return True


def main():
    '''Pre-Processes provided dataset for Google Brain Transparent Object Project.
    It expects a dataset which is a directory containing all the files. The expected naming of the files is set
    as postfix in the SUBFOLDER_MAP dict.

    Note: In a file named '000000020-rgb.jpg' its prefix is '000000020' and its postfix '-rgb.jpg'

    Requires Python > 3.2
    '''
    parser = argparse.ArgumentParser(description='Rearrange non-contiguous scenes in a dataset, move to separate folders and process.')

    parser.add_argument('--p', required=True, help='Path to dataset', metavar='path/to/dataset')
    parser.add_argument('--root', default='../data', help='The root directory of new dataset. Files will moved to and created here.', metavar='path/to/result')
    parser.add_argument('--num_start', default=0, type=int,help='The initial value from which the numbering of renamed files must start')
    parser.add_argument('--height', default=288, type=int, help='The size to which input will be resized to')
    parser.add_argument('--width' , default=512, type=int, help='The size to which input will be resized to')
    parser.add_argument('--test_set', action='store_true', help='Whether we\'re processing a test set, which has only rgb images, and optionally depth images. If this flag is passed, only rgb/depth images are considered')
    args = parser.parse_args()
    
    global SUBFOLDER_MAP
    global SUBFOLDER_MAP_TEST
    global SUBFOLDER_MAP_RESIZED
    global SUBFOLDER_MAP_RESIZED_TEST
    imsize = (args.height, args.width)
    NEW_DATASET_PATHS['root'] = args.root
    if (args.test_set):
        SUBFOLDER_MAP = SUBFOLDER_MAP_TEST
        SUBFOLDER_MAP_RESIZED = SUBFOLDER_MAP_RESIZED_TEST


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
        if ((not os.path.isdir(args.p)) or (not os.listdir(args.p))):
            print(colored("\nWARNING: Source directory '{}' does not exist or is empty.\n  However, found dest dir '{}'.\n".format(args.p, src_dir_path), 'red'))
            print(colored("  Assuming files have already been renamed and moved from Source directory.\n  Proceeding to process files in Dest dir." , 'red'))
            time.sleep(2)
            

    

    
    ### STAGE 1: Move the data into subfolder ###
    # Create new dir to store processed dataset
    if not os.path.isdir(src_dir_path):
        os.makedirs(src_dir_path)
        print ("\nCreated dirs to store new dataset:", src_dir_path)
    else:
        print("\nDataset dir exists:", src_dir_path)

    print("Moving files to", src_dir_path, "and renaming them to start from prefix {:09}.".format(args.num_start))
    count_renamed = move_and_rename_dataset(args.p, src_dir_path, int(args.num_start))
    if(count_renamed > 0):
        color = 'green'
    else:
        color = 'red'
    print(colored("Renamed {} files".format(count_renamed), color))

    print("\nSeparating dataset into folders.")
    move_to_subfolders(src_dir_path)




    ### STAGE 2: Convert World Normals to Camera Normals - skip if test set ###
    if not (args.test_set):
        # Create a pool of processes. By default, one is created for each CPU in your machine.
        with concurrent.futures.ProcessPoolExecutor() as executor:
            # Get a list of files to process
            world_normals_dir = os.path.join(src_dir_path, SUBFOLDER_MAP['world-normals']['folder-name'])
            json_files_dir = os.path.join(src_dir_path, SUBFOLDER_MAP['json-files']['folder-name'])
            
            world_normals_files_list = sorted(glob.glob(os.path.join(world_normals_dir, "*" +
                                                        SUBFOLDER_MAP['world-normals']['postfix'])))
            json_files_list = sorted(glob.glob(os.path.join(json_files_dir, "*" + SUBFOLDER_MAP['json-files']['postfix'])))

            # Process the list of files, but split the work across the process pool to use all CPUs!
            print("\n\nConverting World co-ord Normals to Camera co-ord Normals...Check your CPU usage!!")
            num_converted, num_skipped = 0, 0
            for converted_file in executor.map(preprocess_world_to_cam, world_normals_files_list, json_files_list):
                if converted_file == True:
                    num_converted += 1
                if converted_file == False:
                    num_skipped += 1

            print(colored('\n  Converted {} world-normals'.format(num_converted), 'green'))
            print(colored('  Skipped {} world-normals'.format(num_skipped), 'red'))

    
    

    ### STAGE 3: Preprocess the data required for training ###
    # Create dir to store training data
    print("\n\nPre-Processing data - this will be directly used as training data by model")
    train_dir_path = os.path.join(NEW_DATASET_PATHS['root'], NEW_DATASET_PATHS['training-data'])

    for filetype in SUBFOLDER_MAP_RESIZED:
        subfolder_path = os.path.join(train_dir_path, SUBFOLDER_MAP_RESIZED[filetype]['folder-name'])

        if not os.path.isdir(subfolder_path):
            os.makedirs(subfolder_path)
            print ("    Created dir:", subfolder_path)
        else:
            print("    Already Exists:", subfolder_path)

    print("\n")

    # Create a pool of processes. By default, one is created for each CPU in your machine.
    with concurrent.futures.ProcessPoolExecutor(1) as executor:
        # Process the list of files, but split the work across the process pool to use all CPUs!

        # rgb files
        rgb_imgs_path = os.path.join(src_dir_path, SUBFOLDER_MAP['rgb-files']['folder-name'])
        image_files_rgb = glob.glob(os.path.join(rgb_imgs_path, "*" + SUBFOLDER_MAP['rgb-files']['postfix']))

        input = [(image, imsize) for image in sorted(image_files_rgb)]
        num_converted, num_skipped = 0, 0
        for converted_file in executor.map(preprocess_rgb, input):
            if converted_file == True:
                num_converted += 1
            if converted_file == False:
                num_skipped += 1
        print(colored('\n  Pre-processed {} rgb files'.format(num_converted), 'green'))
        print(colored('  Skipped {} rgb files\n'.format(num_skipped), 'red'))
        
        # surface normal files - skip if test set 
        if not (args.test_set):
            camera_normals_path = os.path.join(src_dir_path, SUBFOLDER_MAP['camera-normals']['folder-name'])
            image_files_normals = glob.glob(os.path.join(camera_normals_path, "*" +
                                                        SUBFOLDER_MAP['camera-normals']['postfix']))
            
            input = [(image, imsize) for image in sorted(image_files_normals)]
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
