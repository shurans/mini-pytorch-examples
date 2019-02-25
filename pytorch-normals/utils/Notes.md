# Notes - Misc

## Configs of software

### Setup Jupyter Notebook

Set a password for notebook

```bash
# Set a password
$ jupyter notebook password
Enter password:
Verify password:
[NotebookPasswordApp] Wrote hashed password to /home/shrek/.jupyter/jupyter_notebook_config.json
```

Generate a config

```bash
# Create a config file
$ jupyter notebook --generate-config
```

Set config params to allow remote access:

```bash
# Edit the file ~/.jupyter/jupyter_notebook_config.py
$ nano ~/.jupyter/jupyter_notebook_config.py

# Add the following lines:
c = get_config()
c.NotebookApp.allow_origin = '*'
c.NotebookApp.allow_remote_access = True
c.NotebookApp.ip = '*'
```

### Select which GPU to train on
If multiple GPUs are present, we can select which GPUs are to be used by setting the environment variable
`CUDA_VISIBLE_DEVICES`.

Eg, to use GPU 0 and GPU 1:

```bash
export CUDA_VISIBLE_DEVICES=0,1
```