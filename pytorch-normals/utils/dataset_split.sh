#!/bin/bash

# dataset path should not end in /
dataset_name=$1
dataset_val=${dataset_name}-val
echo 'actual dataset_path :'
echo $dataset_name
echo 'validation dataset path :'
echo $dataset_val


# Make directories to split dataset into train and valid
mkdir -p -v $dataset_val
mkdir -p -v "$dataset_val"/resized-files/preprocessed-camera-normals/rgb-visualizations
mkdir -p -v "$dataset_val"/resized-files/preprocessed-outlines/rgb-visualizations
mkdir -p -v "$dataset_val"/resized-files/preprocessed-rgb-imgs
mkdir -p -v "$dataset_val"/source-files/camera-normals/rgb-visualizations
mkdir -p -v "$dataset_val"/source-files/component-masks
mkdir -p -v "$dataset_val"/source-files/depth-imgs
mkdir -p -v "$dataset_val"/source-files/depth-imgs-rectified
mkdir -p -v "$dataset_val"/source-files/json-files
mkdir -p -v "$dataset_val"/source-files/outlines/rgb-visualizations
mkdir -p -v "$dataset_val"/source-files/rgb-imgs
mkdir -p -v "$dataset_val"/source-files/variant-masks
mkdir -p -v "$dataset_val"/source-files/world-normals

# To determine the percentage split
count=`ls -ltr ${dataset_name}/resized-files/preprocessed-rgb-imgs/* | wc -l`
count=$(($((count-1))/10))
echo $count

echo 'moving files to val dataset'
# splitting resized files
for i in $(eval echo "{000000000..$count}");
do
mv "$dataset_name"/resized-files/preprocessed-camera-normals/$i-cameraNormals.exr "$dataset_val"/resized-files/preprocessed-camera-normals/
mv "$dataset_name"/resized-files/preprocessed-camera-normals/rgb-visualizations/$i-cameraNormals.png "$dataset_val"/resized-files/preprocessed-camera-normals/rgb-visualizations/
mv "$dataset_name"/resized-files/preprocessed-outlines/$i-outlineSegmentation.png "$dataset_val"/resized-files/preprocessed-outlines/
mv "$dataset_name"/resized-files/preprocessed-outlines/rgb-visualizations/$i-outlineSegmentation.png "$dataset_val"/resized-files/preprocessed-outlines/rgb-visualizations/
mv "$dataset_name"/resized-files/preprocessed-rgb-imgs/$i-rgb.png "$dataset_val"/resized-files/preprocessed-rgb-imgs/

# splitting source files
mv "$dataset_name"/source-files/camera-normals/$i-cameraNormals.exr "$dataset_val"/source-files/camera-normals/
mv "$dataset_name"/source-files/camera-normals/rgb-visualizations/$i-cameraNormals.png "$dataset_val"/source-files/camera-normals/rgb-visualizations/
mv "$dataset_name"/source-files/component-masks/$i-componentMasks.exr "$dataset_val"/source-files/component-masks/
mv "$dataset_name"/source-files/depth-imgs/$i-depth.exr "$dataset_val"/source-files/depth-imgs/
mv "$dataset_name"/source-files/depth-imgs-rectified/$i-depth.exr "$dataset_val"/source-files/depth-imgs-rectified/
mv "$dataset_name"/source-files/json-files/$i-masks.json "$dataset_val"/source-files/json-files/
mv "$dataset_name"/source-files/outlines/$i-outlineSegmentation.png "$dataset_val"/source-files/outlines/
mv "$dataset_name"/source-files/outlines/rgb-visualizations/$i-outlineSegmentationRgb.png "$dataset_val"/source-files/outlines/rgb-visualizations/
mv "$dataset_name"/source-files/rgb-imgs/$i-rgb.jpg "$dataset_val"/source-files/rgb-imgs/
mv "$dataset_name"/source-files/variant-masks/$i-variantMasks.exr "$dataset_val"/source-files/variant-masks/
mv "$dataset_name"/source-files/world-normals/$i-normals.exr "$dataset_val"/source-files/world-normals/

done

# Re-name the original directory
echo 're-naming remaning dataset as train dataset'
mv "$dataset_name" "$dataset_name"-train