
### Clone
```
$ git clone https://github.com/Shreeyak/mini-pytorch-examples.git
```

### Dependencies
```
$ sudo apt install -y libopenexr-dev
$ sudo pip3 install OpenEXR
$ sudo pip3 install termcolor

$ sudo pip3 install tensorflow #(gpu version not needed, this is only for tensorboard)
$ sudo pip3 install tensorboardX

```

### Setup Jupyter Notebook
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