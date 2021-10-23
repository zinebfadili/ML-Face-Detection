# ML-Face-Detection

The goal of this project, a part of the Machine Learning Module, is to create a system for face detection using deep learning, or convolutional neural networks. In this project we prepared and found data for training and testing our model, we chose an architecture for our neural network. Furthermore, we implemented the bootstrapping technique to improve the accuracy of our model. Finally, we implemented the face detection system using the sliding window approach.

## Train and test dataset
Train and test datasets can be found on Google Drive (add link here)
To run the project, you need to create 4 directories in the project root:
1) `./test_images` that contains face and non-face images for testing
2) `./test_images_bootstrap` contains the non-faces images used for bootstrapping
3) `./train_images` contains train images from the original dataset
4) `./train_images_bootstrap` must contain the same train images as `./train_images` at the start of the run. The directory will then be updated with false positive images from bootstrapping

## Requirements
- Python 3
- Pytorch 1.10 (`pip3 install torch==1.10.0+cu102 torchvision==0.11.1+cu102 torchaudio===0.10.0+cu102 -f https://download.pytorch.org/whl/cu102/torch_stable.html`)
- PIL, opencv, numpy libraries

## Running the project
The model can be trained and tested by running the `test.py` script. The execution will save 2 seperate models:
- the model before running bootstrap `./model_without_bootstrap.pth`
- the model after running bootstrap loops `./model_with_bootstrap.pth`
Once the model is trained, run face detection on an image of your choice (format must be `.pgm`) using `scales.py`. The image will be exported in `.jpg` format with the rectangles containing the faces.