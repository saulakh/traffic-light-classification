import keras
import numpy as np
import cv2
import glob

source = './traffic_light_imgs/'

# Load keras model for traffic lights
model = keras.models.load_model('./lights_model.h5')
print("Model loaded")

# Load new images as test dataset
images = []
images_list = glob.glob(source + 'image*.jpg')

for image in images_list:
    img = cv2.imread(image)
    img_resize = cv2.resize(img, (32,72), interpolation = cv2.INTER_AREA)
    images.append(img_resize)

np_images = np.asarray(images)
print("Shape of test images: ", np_images.shape)

# Best traffic light prediction from sofmax probabilities
softmax = model.predict(np_images)
pred = np.argmax(softmax, axis=1)
print("Predicted light ID for test images: ", pred)

# Print predicted traffic light color
labels = ['red', 'yellow', 'green']
pred_labels = []

true_labels = [2, 2, 2, 0, 0, 1]
print("True labels: ", true_labels)
incorrect_count = 0

for i in range(len(pred)):
    traffic_color = labels[pred[i]]
    pred_labels.append(traffic_color)
    if pred[i] != true_labels[i]:
        print("Image", i, ", Predicted:", labels[pred[i]], ", Actual:", labels[true_labels[i]])
        incorrect_count +=1

print("Predicted traffic lights: ", pred_labels)

acc = (len(pred) - incorrect_count)/len(pred)
print("Accuracy: ", acc*100, "%")