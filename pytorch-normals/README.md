
# Transparent Object Detection
This repo contains networks used to detect transparent objects with the help of the 
[Deep Depth Completion of a Single RGB-D Image](https://github.com/yindaz/DeepCompletionRelease) method.

## To Use
Install dependencies mentioned below, then clone the repo by
```
$ git clone https://github.com/Shreeyak/mini-pytorch-examples.git
```

Download the synthetic datasets and pre-process them with the scripts in utils (#TODO).
Add the source files into pytorch-normals/data and pytorch-outlines/data and run the networks. Check out the notebooks 
for how to run training/eval.

## Setup
#### Dependencies
Python3 is used for the code. Install the dependencies below. If using Conda to manage virtual environments, 
conda install commands are provided. Otherwise, install with pip or from source.

- OpenEXR  
OpenEXR is required to read in the files in .exr format. Install [OpenEXR](https://github.com/openexr/openexr) libs and
its [python bindings](https://github.com/jamesbowman/openexrpython). 
```
$ sudo apt install -y libopenexr-dev zlib1g-dev
$ sudo apt install openexr
$ pip install git+https://github.com/jamesbowman/openexrpython.git
```

Note: Do NOT install openexr libs from conda, it installs the openexr libs and not the OpenEXR python bindings, causing import issues.
These are the commands not to be run:
```
# DO NOT INSTALL THIS
# conda install -c conda-forge openexr
# conda install -c conda-forge ilmbase
``` 

- TensorboardX
[TensorboardX](https://github.com/lanpa/tensorboardX) is a version of tensorboard built for pytorch. It requires tensorboard to be installed
```
$ pip install tensorboardX
$ pip install tensorflow  #Alternate: conda install -c conda-forge tensorflow
```

- Others
```
$ pip install termcolor
$ pip install imageio  #Alternate: conda install -c conda-forge imageio
```

#### Setup Jupyter Notebook
Set a password for notebook
```
$ jupyter notebook password
Enter password:
Verify password:
[NotebookPasswordApp] Wrote hashed password to /home/shrek/.jupyter/jupyter_notebook_config.json
```
Generate a config
```
$ jupyter notebook --generate-config
```
Set config params to allow remote access:
```
# Edit the file ~/.jupyter/jupyter_notebook_config.py
$ nano ~/.jupyter/jupyter_notebook_config.py

# Add the following lines:
c = get_config()
c.NotebookApp.allow_origin = '*'
c.NotebookApp.allow_remote_access = True
c.NotebookApp.ip = '*'
```