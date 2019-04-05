#!/bin/bash

# deleting depth rectified images
echo 'deleting depth rectified images'
rm -rf scoop-val/source-files/depth-imgs-rectified
rm -rf scoop-train/source-files/depth-imgs-rectified
rm -rf sphere-bath-bomb-val/source-files/depth-imgs-rectified
rm -rf sphere-bath-bomb-train/source-files/depth-imgs-rectified
rm -rf tree-bath-bomb-val/source-files/depth-imgs-rectified
rm -rf tree-bath-bomb-train/source-files/depth-imgs-rectified
rm -rf star-bath-bomb-val/source-files/depth-imgs-rectified
rm -rf star-bath-bomb-train/source-files/depth-imgs-rectified
rm -rf short-bottle-no-cap-val/source-files/depth-imgs-rectified
rm -rf short-bottle-no-cap-train/source-files/depth-imgs-rectified
rm -rf test-tube-with-cap-val/source-files/depth-imgs-rectified
rm -rf test-tube-with-cap-train/source-files/depth-imgs-rectified
rm -rf test-tube-no-cap-val/source-files/depth-imgs-rectified
rm -rf test-tube-no-cap-train/source-files/depth-imgs-rectified
rm -rf cup-with-waves-val/source-files/depth-imgs-rectified
rm -rf cup-with-waves-train/source-files/depth-imgs-rectified
rm -rf heart-bath-bomb-val/source-files/depth-imgs-rectified
rm -rf heart-bath-bomb-train/source-files/depth-imgs-rectified
rm -rf stemless-champagne-glass-val/source-files/depth-imgs-rectified
rm -rf stemless-champagne-glass-train/source-files/depth-imgs-rectified
rm -rf flower-bath-bomb-val/source-files/depth-imgs-rectified
rm -rf flower-bath-bomb-train/source-files/depth-imgs-rectified
rm -rf short-bottle-with-cap-val/source-files/depth-imgs-rectified
rm -rf short-bottle-with-cap-train/source-files/depth-imgs-rectified

# deleting outlines files
echo 'deleting outlines files'
rm -rf scoop-val/source-files/outlines
rm -rf scoop-train/source-files/outlines
rm -rf sphere-bath-bomb-val/source-files/outlines
rm -rf sphere-bath-bomb-train/source-files/outlines
rm -rf tree-bath-bomb-val/source-files/outlines
rm -rf tree-bath-bomb-train/source-files/outlines
rm -rf star-bath-bomb-val/source-files/outlines
rm -rf star-bath-bomb-train/source-files/outlines
rm -rf short-bottle-no-cap-val/source-files/outlines
rm -rf short-bottle-no-cap-train/source-files/outlines
rm -rf test-tube-with-cap-val/source-files/outlines
rm -rf test-tube-with-cap-train/source-files/outlines
rm -rf test-tube-no-cap-val/source-files/outlines
rm -rf test-tube-no-cap-train/source-files/outlines
rm -rf cup-with-waves-val/source-files/outlines
rm -rf cup-with-waves-train/source-files/outlines
rm -rf heart-bath-bomb-val/source-files/outlines
rm -rf heart-bath-bomb-train/source-files/outlines
rm -rf stemless-champagne-glass-val/source-files/outlines
rm -rf stemless-champagne-glass-train/source-files/outlines
rm -rf flower-bath-bomb-val/source-files/outlines
rm -rf flower-bath-bomb-train/source-files/outlines
rm -rf short-bottle-with-cap-val/source-files/outlines
rm -rf short-bottle-with-cap-train/source-files/outlines

# deleting preprocessed outlines files
echo 'deleting preprocessed outlines files'
rm -rf scoop-val/resized-files/preprocessed-outlines
rm -rf scoop-train/resized-files/preprocessed-outlines
rm -rf sphere-bath-bomb-val/resized-files/preprocessed-outlines
rm -rf sphere-bath-bomb-train/resized-files/preprocessed-outlines
rm -rf tree-bath-bomb-val/resized-files/preprocessed-outlines
rm -rf tree-bath-bomb-train/resized-files/preprocessed-outlines
rm -rf star-bath-bomb-val/resized-files/preprocessed-outlines
rm -rf star-bath-bomb-train/resized-files/preprocessed-outlines
rm -rf short-bottle-no-cap-val/resized-files/preprocessed-outlines
rm -rf short-bottle-no-cap-train/resized-files/preprocessed-outlines
rm -rf test-tube-with-cap-val/resized-files/preprocessed-outlines
rm -rf test-tube-with-cap-train/resized-files/preprocessed-outlines
rm -rf test-tube-no-cap-val/resized-files/preprocessed-outlines
rm -rf test-tube-no-cap-train/resized-files/preprocessed-outlines
rm -rf cup-with-waves-val/resized-files/preprocessed-outlines
rm -rf cup-with-waves-train/resized-files/preprocessed-outlines
rm -rf heart-bath-bomb-val/resized-files/preprocessed-outlines
rm -rf heart-bath-bomb-train/resized-files/preprocessed-outlines
rm -rf stemless-champagne-glass-val/resized-files/preprocessed-outlines
rm -rf stemless-champagne-glass-train/resized-files/preprocessed-outlines
rm -rf flower-bath-bomb-val/resized-files/preprocessed-outlines
rm -rf flower-bath-bomb-train/resized-files/preprocessed-outlines
rm -rf short-bottle-with-cap-val/resized-files/preprocessed-outlines
rm -rf short-bottle-with-cap-train/resized-files/preprocessed-outlines

# running data pre-processing script
echo 'running pre-processing script'

cd ../../utils

python data_processing_script.py --p ../data/datasets/empt --root ../data/datasets/sccop-val --num_start 0
python data_processing_script.py --p ../data/datasets/empt --root ../data/datasets/sccop-train --num_start 49
python data_processing_script.py --p ../data/datasets/empt --root ../data/datasets/sphere-bath-bomb-val --num_start 0
python data_processing_script.py --p ../data/datasets/empt --root ../data/datasets/sphere-bath-bomb-train --num_start 49
python data_processing_script.py --p ../data/datasets/empt --root ../data/datasets/tree-bath-bomb-val --num_start 0
python data_processing_script.py --p ../data/datasets/empt --root ../data/datasets/tree-bath-bomb-train --num_start 49
python data_processing_script.py --p ../data/datasets/empt --root ../data/datasets/star-bath-bomb-val --num_start 0
python data_processing_script.py --p ../data/datasets/empt --root ../data/datasets/star-bath-bomb-train --num_start 49
python data_processing_script.py --p ../data/datasets/empt --root ../data/datasets/test-tube-with-cap-val --num_start 0
python data_processing_script.py --p ../data/datasets/empt --root ../data/datasets/test-tube-with-cap-train --num_start 49
python data_processing_script.py --p ../data/datasets/empt --root ../data/datasets/test-tube-no-cap-val --num_start 0
python data_processing_script.py --p ../data/datasets/empt --root ../data/datasets/test-tube-no-cap-train --num_start 49
python data_processing_script.py --p ../data/datasets/empt --root ../data/datasets/cup-with-waves-val --num_start 0
python data_processing_script.py --p ../data/datasets/empt --root ../data/datasets/cup-with-waves-train --num_start 49
python data_processing_script.py --p ../data/datasets/empt --root ../data/datasets/heart-bath-bomb-val --num_start 0
python data_processing_script.py --p ../data/datasets/empt --root ../data/datasets/heart-bath-bomb-train --num_start 49
python data_processing_script.py --p ../data/datasets/empt --root ../data/datasets/stemless-champagne-glass-val --num_start 0
python data_processing_script.py --p ../data/datasets/empt --root ../data/datasets/stemless-champagne-glass-train --num_start 49
python data_processing_script.py --p ../data/datasets/empt --root ../data/datasets/flower-bath-bomb-val --num_start 0
python data_processing_script.py --p ../data/datasets/empt --root ../data/datasets/flower-bath-bomb-train --num_start 49
python data_processing_script.py --p ../data/datasets/empt --root ../data/datasets/short-bottle-with-cap-val --num_start 0
python data_processing_script.py --p ../data/datasets/empt --root ../data/datasets/short-bottle-with-cap-train --num_start 49

