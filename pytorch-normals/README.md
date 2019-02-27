
# Transparent Object Detection
This repo contains networks used to detect transparent objects with the help of the 
[Deep Depth Completion of a Single RGB-D Image](https://github.com/yindaz/DeepCompletionRelease) method.

## To Use

1. Install dependencies mentioned later in this doc. Then clone the repo by

```
$ git clone https://github.com/Shreeyak/mini-pytorch-examples.git
```

2) Download the synthetic datasets and pre-process them with the scripts in utils, or download the pre-processed datasets.

3) Add the datasets into pytorch-normals/data/datasets and pytorch-outlines/data/datasets and run the networks. Check out the notebooks for how to run training/eval.

## Preparing Data

### Using the prepared datasets

The pre-processed dataset will have the below folder structure. The actual, full resolution images are stored in `source-files` and are treated as source images to create training data for the model. 

```bash
milk-bottle
|- source-files
   |- ...
|- resized-files
   |- ...
```

 There are multiple objects for whom synthetic data has been rendered. A separate dataset is prepared for each object. A separate pytorch dataset is
 used within the code for each of these datasets, which are then concatenated into a large train-val dataset, which is
 subsequently split into train and val datasets.

### Preparing your own data

The synthetic data that is generated for training has 6 files for each image. All files are placed into a single folder, like so:

```bash
dataset
|- 0001-rgb.jpg
|- 0001-depth.exr
|- 0001-masks.json
|- 0001-variantMasks.exr
|- 0001-componentMasks.exr
|- 0001-normals.exr
|- 0002-rgb.jpg
|- 0002-depth.exr
.
.
.
```

The script `pytorch-normals/utils/data_processing_script.py` is used to process this dataset. First, each of the 6 types of files are
moved into their own folder, then the surface normals in world-coordinates are converted to camera-coordinates and finally, the
files are resized to make it faster to load into the network.
It can be used as so:

```bash
$ python3 data_processing_script.py  --p "path/to/raw/dataset"
  --root "path/to/store/new/dataset"
  --num_start 0     # The initial value from which the numbering of renamed files must start
  --height 288      # The size to which input will be resized to
  --width 512       # The size to which input will be resized to
```

In case of processing a test set, which contains only RGB images and optionally depth images in case of a realsense camera,
the command is used as below. In this case, the conversion of surface normals from world to camera co-ordinates is skipped.
Also, all files apart from rgb and depth are ignored:

```bash
$ python3 data_processing_script.py  --p "path/to/raw/dataset"
  --root "path/to/store/new/dataset"
  --num_start 0     # The initial value from which the numbering of renamed files must start
  --height 288      # The size to which input will be resized to
  --width 512       # The size to which input will be resized to
  --test_set
```

All the files in the dataset are expected to conform to a certain naming scheme and file format. This list can be seen and
modified in the dictionaries within the `data_processing_script.py` script.



## Setup Dependencies

Ubuntu 16.04 and Python3 is used for the code. Install the dependencies below. If using Conda to manage virtual environments,
conda install commands are provided. Otherwise, install with pip or from source.

### OpenEXR

OpenEXR is required to read in the files in .exr format. Install [OpenEXR](https://github.com/openexr/openexr) libs and
its [python bindings](https://github.com/jamesbowman/openexrpython). 

```bash
# Install OpenEXR
$ sudo apt install -y libopenexr-dev zlib1g-dev
$ sudo apt install openexr
$ pip install git+https://github.com/jamesbowman/openexrpython.git
```

Note: If using Conda, Do NOT install openexr libs from conda, it installs the openexr libs and not the OpenEXR python bindings, causing import issues.
These are the commands not to be run:

```bash
# DO NOT INSTALL THIS
# conda install -c conda-forge openexr
# conda install -c conda-forge ilmbase
``` 

### TensorboardX

[TensorboardX](https://github.com/lanpa/tensorboardX) is a version of tensorboard built for pytorch. It requires tensorboard to be installed

```bash
# Install tensorboardX
$ pip install tensorboardX
$ pip install tensorflow  #Alternate: conda install -c conda-forge tensorflow
```

### imgaug

This is a data augmentation library which we use with pytorch dataloaders to resize, crop, flip, rotate, etc.

```bash
# Requirements
$ pip install six numpy scipy Pillow matplotlib scikit-image imageio Shapely
#$ conda install six numpy scipy Pillow matplotlib scikit-image imageio Shapely

$ pip install opencv-python

# imgaug from github
$ pip install git+https://github.com/aleju/imgaug
```

### LibRealSense  

[LibRealSense](https://github.com/IntelRealSense/librealsense) is required to stream and capture images from Intel Realsense D415/D435 stereo cameras.  
Please check the [installation guide](https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md) to install from binaries, or compile from source.

```bash
# Register the server's public key:
$ sudo apt-key adv --keyserver keys.gnupg.net --recv-key C8B3A55A6F3EFCDE || sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-key C8B3A55A6F3EFCDE

# Ubuntu 16 LTS - Add the server to the list of repositories
$ sudo add-apt-repository "deb http://realsense-hw-public.s3.amazonaws.com/Debian/apt-repo xenial main" -u

# Install the libraries
$ sudo apt-get install librealsense2-dkms
$ sudo apt-get install librealsense2-dkms

# Install the developer and debug packages
$ sudo apt-get install librealsense2-dev
$ sudo apt-get install librealsense2-dbg
```

### AttrDict
[AttrDict](https://pypi.org/project/attrdict/) is a library to convert yaml files into python objects. The dict keys can be accessed as attributes.
This lib is used to parse the dictionary read in from yaml config file into object attributes, so they can be accessed by dot notation like `config.batchSize`.

```bash
# Install AttrDict
$ pip install attrdict # conda install attrdict -c conda-forge
```

### Others

```bash
# Misc Packages
$ pip install termcolor
```