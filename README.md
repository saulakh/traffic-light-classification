## Traffic Light Classification
Self Driving Car Nanodegree - Capstone Project

### Overview

To supplement the [Self-Driving Car Capstone project](https://github.com/saulakh/self-driving-car-capstone), this repository contains the model I used to detect and classify traffic lights. These files will be modified for the capstone project, so this is a classifier that can run independently from the system integration project.

### Project Code

The `traffic_light_imgs` folder contains cropped and resized images of traffic lights for the training dataset. The `traffic_lights.csv` file contains image paths and corresponding labels for each image. The `traffic_light_model.py` file contains the keras model for traffic light classification, and saves the model as a .h5 file. The `traffic_light_classifier.py` file loads the saved model to predict new images.

The project can be run by doing the following from the project top directory:

1. python traffic_light_model.py
2. python traffic_light_classifier.py

### Traffic Light Dataset

I saved camera images from the capstone simulation and created a .csv file with image paths and traffic light labels, similar to Udacity’s training data from the behavioral cloning project. In the [tl_detector.py](https://github.com/saulakh/self-driving-car-capstone/blob/main/ros/src/tl_detector/tl_detector.py) file from the capstone project, I started saving images from the get_light_state function:

![image](https://user-images.githubusercontent.com/74683142/130552680-90c21a1d-b668-4a0f-a19e-bf2af1f993fd.png)

##### Object Detection Lab

Next, I downloaded the [COCO SSD model](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_11_06_2017.tar.gz)  to detect traffic lights from the images. Traffic lights were already one of the existing objects from the COCO model, so I worked through this [object detection lab](https://github.com/udacity/CarND-Object-Detection-Lab) from Udacity to get the bounding boxes for each traffic light.

![image](https://user-images.githubusercontent.com/74683142/130553105-64375e61-d499-407d-9f66-1dc40a6dc980.png)

Using the bounding boxes, I saved cropped images of the traffic lights and later resized the dimensions to (32,72,3).

![image](https://user-images.githubusercontent.com/74683142/130553487-19a89e4a-11d6-49e8-8596-dcf38179c730.png) ![image](https://user-images.githubusercontent.com/74683142/130553506-e9c08767-9c45-48ba-b839-d19f87470d0c.png) ![image](https://user-images.githubusercontent.com/74683142/130553522-3b800513-7109-4ecf-bb0b-b445e1123272.png)

##### Preprocessing

I experimented with different options for preprocessing images, to isolate the traffic lights from the rest of the image. With the HSV color space, the brightness values could have helped to classify traffic lights.

![image](https://user-images.githubusercontent.com/74683142/130553369-394d06e3-2de0-4bc1-b1fa-9f67991d8b4c.png) ![image](https://user-images.githubusercontent.com/74683142/130553387-b73ea36a-3579-40c3-9cde-f387b7f9ffdf.png)

I also tried OpenCV methods such as Hough circles, sobel gradients, color thresholding, and Canny edge detection:

![image](https://user-images.githubusercontent.com/74683142/130554198-a8dbc45e-df1d-49dc-9eca-97594de2427a.png) ![image](https://user-images.githubusercontent.com/74683142/130554214-20ada3c4-9288-47f8-acff-6ade031db2eb.png)

This did not help much to isolate the traffic lights, so I kept the RGB color space, with no additional preprocessing for the images.

##### Data Augmentation

The model was performing well with a small dataset, but I flipped the images to include additional data.

### Model Architecture

I had minimal machine learning experience before this project, so this was a good chance to experiment with models and datasets to learn more. I initially started with the NVIDIA architecture, using the full images before cropping out the traffic lights. To adapt the model for classification, I changed the size of the output layer to match the number of classes, added a softmax activation for the output layer, added one hot encoding for the traffic light labels, and switched to ‘categorical_crossentropy’ for the model loss.

This worked well enough as a starting point, with an accuracy of 87.3% on the test dataset. I also tested on new images and the model was able to predict all of them correctly. This was the loss plot using the NVIDIA architecture, after 10 epochs:

![image](https://user-images.githubusercontent.com/74683142/130557672-19f463b1-0a9d-455d-b599-973b930af817.png)

As I increased the variety of new images to test on, this model was not consistent enough. Some camera images had landscape in the background behind the traffic lights, and the model classified some of these images incorrectly:

![image](https://user-images.githubusercontent.com/74683142/130557494-6381e273-f5ca-4e66-926f-ee69f5319f9a.png) ![image](https://user-images.githubusercontent.com/74683142/130557512-ae3d93ec-5dac-4ff4-bd51-9707813fd24a.png)

I decided to use cropped traffic light images from the COCO object detection lab, and switch to transfer learning instead.

##### VGG-16

To switch to transfer learning, I looked at some of the options for pretrained models. VGG-16 was listed as one of the top choices for image classification, with 16 convolutional layers and a simpler architecture. For traffic light classification, the cropped images looked similar to each other and a simpler feature extraction seemed sufficient.

##### Additional Layers

I imported the VGG-16 model, changed the input shape to match the resized images, and added a dense layer for 3 classes with a softmax activation function.

```
model = Sequential()

# Add layers to pretrained model
model.add(VGG16(include_top = False, input_shape = (72,32,3)))
model.add(Flatten())
model.add(Dense(3, activation="softmax"))

model.compile(loss='categorical_crossentropy', optimizer='adam') # try different learning rates
history_object = model.fit(X_train, y_encoded, validation_split = 0.2, shuffle = True, epochs=10)
```

##### Classifier Results

Using transfer learning with the VGG-16 model, I had a 100% accuracy while testing on new images. I have not integrated the traffic light classifier with the capstone project yet, but this model works well on its own.

