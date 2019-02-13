import os
import fnmatch
import argparse
import numpy as np
import OpenEXR
import Imath
from pathlib import Path
from scipy.misc import imsave

parser = argparse.ArgumentParser()
parser.add_argument('--p', required=True,
                    help='Path to dataset', metavar='path/to/dataset')
args = parser.parse_args()

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


for path,sub_dir,files in os.walk(args.p):
    for f in files:
        print(f)
        path = args.p+f[:-4]+'.exr'
        x = np.load(os.path.join(args.p,f))
        exr_saver(path, x)
