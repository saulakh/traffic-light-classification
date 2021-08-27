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

I saved camera images from [tl_detector.py](https://github.com/saulakh/self-driving-car-capstone/blob/main/ros/src/tl_detector/tl_detector.py) in the capstone project and created a .csv file with image paths and traffic light labels, similar to Udacity’s training data from the behavioral cloning project. Here is a sample image:

![image](https://user-images.githubusercontent.com/74683142/130552680-90c21a1d-b668-4a0f-a19e-bf2af1f993fd.png)

I also included traffic light images from the internet, and had 246 total images in the dataset. Here are a few examples:

![image](https://user-images.githubusercontent.com/74683142/131002208-9342cd77-8ef5-4662-916d-d90d5cd6091e.png) ![image](https://user-images.githubusercontent.com/74683142/131002227-dc7cee67-dea8-43cc-9295-1a2364fd2e76.png) ![image](https://user-images.githubusercontent.com/74683142/131002233-99984eca-0f0b-4507-98eb-2f101d1b1eaf.png)

Next, I downloaded the [COCO SSD model](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_11_06_2017.tar.gz)  to detect traffic lights from the images. Traffic lights were already one of the existing objects from the COCO model, so I worked through this [object detection lab](https://github.com/udacity/CarND-Object-Detection-Lab) from Udacity to get the bounding boxes for each traffic light.

![image](https://user-images.githubusercontent.com/74683142/131134722-e4ace1b6-be2c-4891-9316-bf9732533e40.png) ![image](https://user-images.githubusercontent.com/74683142/131134752-82990bef-ee1c-4a78-a6e5-a7ca566e15a4.png)

Using the bounding boxes, I saved cropped images of the traffic lights and later resized the dimensions to (48,108,3).

![cropped4](https://user-images.githubusercontent.com/74683142/131135108-92669061-69ee-47eb-81b3-65c42d11023f.jpg) ![cropped2](https://user-images.githubusercontent.com/74683142/131134990-b5f11255-6bf5-4f5b-b2fe-48c380c7e4e1.jpg) ![image](https://user-images.githubusercontent.com/74683142/131134887-1889e4e5-d31e-4ba6-8d8b-459f3ce310f8.png) ![cropped3](https://user-images.githubusercontent.com/74683142/131135053-36f6d0d7-4cb5-44b7-9a3e-27187f05d864.jpg)

### Model Architecture

I had minimal machine learning experience before this project, so this was a good chance to experiment with models and datasets to learn more. I initially started with the NVIDIA architecture, using the full images before cropping out the traffic lights. To adapt the model for classification, I changed the size of the output layer to match the number of classes, added a softmax activation for the output layer, added one hot encoding for the traffic light labels, and switched to ‘categorical_crossentropy’ for the model loss.

This worked well enough as a starting point, with an accuracy of 87.3% on the test dataset. I also tested on 10 new images and the model was able to predict all of them correctly. This was the loss plot using the NVIDIA architecture, after 10 epochs:

![image](https://user-images.githubusercontent.com/74683142/130557672-19f463b1-0a9d-455d-b599-973b930af817.png)

As I increased the variety of new images to test on, this model was not consistent enough. Some camera images had landscape in the background behind the traffic lights, and the model classified some of these images incorrectly:

![image](https://user-images.githubusercontent.com/74683142/130557512-ae3d93ec-5dac-4ff4-bd51-9707813fd24a.png) ![image](https://user-images.githubusercontent.com/74683142/131140113-ff555471-63e2-4263-8d49-121c9371105a.png)

##### VGG-16

To switch to transfer learning, I looked at some of the options for pretrained models. VGG-16 was listed as one of the top choices for image classification, with 16 convolutional layers and a simpler architecture. For traffic light classification, the cropped images looked similar to each other and a simpler feature extraction seemed sufficient. I decided to use the coco object detection model to find traffic lights from the image, then use the keras model to predict the traffic light color.

##### Additional Layers

I imported the VGG-16 model, changed the input shape to match the resized images, and added a dense layer for 3 classes with a softmax activation function.

```
model = Sequential()

# Add layers to pretrained model
model.add(VGG16(include_top = False, input_shape = (108,48,3)))
model.add(Flatten())
model.add(Dense(3, activation="softmax"))

model.compile(loss='categorical_crossentropy', optimizer='adam') # try different learning rates
history_object = model.fit(X_train, y_encoded, validation_split = 0.2, shuffle = True, epochs=10)
```

Since the majority of the dataset included simulation images, the model was able to classify those correctly. With internet images, this green light was incorrectly classified as yellow, and some of the yellow lights were predicted as red:

![image](https://user-images.githubusercontent.com/74683142/131137865-e48588cd-7ada-4beb-a36c-c5c839352ed5.png) ![image](https://user-images.githubusercontent.com/74683142/131141058-2e1f56b9-a3d8-476e-9ff9-3b7fb603ca06.png)

It seemed like the images being classified incorrectly had different shades of green or yellow compared to the majority. After including more internet images in the training dataset and retraining the model, the accuracy was improving with new images.

##### Classifier Results

Using transfer learning with the VGG-16 model, the classifier had a high accuracy when a traffic light was detected. I have not integrated the traffic light classifier with the capstone project yet, but this model works well on my local computer.

To improve the results, I could include simulation images in the object detection training dataset. The keras model is able to classify the lights, but there are still some cases where the object detection lab does not see the traffic lights.

![image](https://user-images.githubusercontent.com/74683142/131136381-c4202b45-afbe-4acd-ad58-6141eaeba634.png) ![image](https://user-images.githubusercontent.com/74683142/131136400-6a8ac27d-833d-45ed-ac54-c22dc35997de.png)