# Image Segmentation

The aim of this challenge is to solve a segmentation problem on the proposed dataset, which contains aerial images of different areas. Given an image, the goal is to segment buildings at pixel level, thus predicting for each pixel if it belongs to a building (class 1) or not (class 0).
 
 ## Data Details
 
* **Image size**:  256x256 pixels
* **Color space**: RGB
* **File Format**: tif
* **Number of classes**: 2
* **Classes**:
  * 'background' : 0
  * 'building' : 1 (corresponding to the value 255 in the stored masks)
* **Number of training images**: 7647
* **Number of tes images**: 1234

## Dataset Structure
The dataset contains two main folders: training and test. Folder training contains two subfolders: images and masks. Folder images contains RGB images. Folder masks contains the corresponding segmentation masks (ground truth). All images are in an additional subfolder img to allow the use of the ImageDataGenerator.flow_from_directory with the attribute class_mode set to None. Folder test contains only RGB images since no segmentation masks are provided. Both RGB images and masks are in tif format. Masks contains two values 0 (background) and 255 (buildings).