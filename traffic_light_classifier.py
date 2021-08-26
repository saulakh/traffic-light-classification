import tensorflow as tf
import keras
import numpy as np
from PIL import Image
from PIL import ImageDraw
import cv2

source = './images/'

# Path for COCO SSD model graph
COCO_GRAPH = 'ssd_mobilenet_v1_coco_11_06_2017/frozen_inference_graph.pb'

def load_image(path):
    image = Image.open(path)
    return image

def filter_boxes(min_score, boxes, scores, classes):
    """Return boxes with a confidence >= `min_score`"""
    n = len(classes)
    idxs = []
    for i in range(n):
        if scores[i] >= min_score:
            idxs.append(i)
    
    filtered_boxes = boxes[idxs, ...]
    filtered_scores = scores[idxs, ...]
    filtered_classes = classes[idxs, ...]
    return filtered_boxes, filtered_scores, filtered_classes

def to_image_coords(boxes, height, width):
    """
    The original box coordinate output is normalized, i.e [0, 1].
    
    This converts it back to the original coordinate based on the image
    size.
    """
    box_coords = np.zeros_like(boxes)
    box_coords[:, 0] = boxes[:, 0] * height
    box_coords[:, 1] = boxes[:, 1] * width
    box_coords[:, 2] = boxes[:, 2] * height
    box_coords[:, 3] = boxes[:, 3] * width
    
    return box_coords

def draw_boxes(image, boxes, classes, thickness=4):
    """Draw bounding boxes on the image"""
    draw = ImageDraw.Draw(image)
    for i in range(len(boxes)):
        bot, left, top, right = boxes[i, ...]
        class_id = int(classes[i])
        color = 'blueviolet'
        draw.line([(left, top), (left, bot), (right, bot), (right, top), (left, top)], width=thickness, fill=color)
        
def load_graph(graph_file):
    """Loads a frozen inference graph"""
    graph = tf.Graph()
    with graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(graph_file, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return graph

# Load graph file
detection_graph = load_graph(COCO_GRAPH)
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
print("Graph loaded")

# Load keras model for traffic lights
model = keras.models.load_model('./lights_model.h5')
print("Model loaded")

def predict_label(image_np):
    # Best traffic light prediction from sofmax probabilities
    softmax = model.predict(image_np)
    pred = np.argmax(softmax, axis=1)

    # Print predicted traffic light color
    labels = ['Red', 'Yellow', 'Green']
    pred_label = labels[pred[0]]

    return pred_label

def get_traffic_light(image):
    # Convert image to np array
    image_np = np.expand_dims(np.asarray(image, dtype=np.uint8), 0)

    with tf.compat.v1.Session(graph=detection_graph) as sess:

        # Actual detection.
        (boxes, scores, classes) = sess.run([detection_boxes, detection_scores, detection_classes], 
                                            feed_dict={image_tensor: image_np})

        # Remove unnecessary dimensions
        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes)

        confidence_cutoff = 0.8
        # Filter boxes with a confidence score less than `confidence_cutoff`
        boxes, scores, classes = filter_boxes(confidence_cutoff, boxes, scores, classes)

        # The current box coordinates are normalized to a range between 0 and 1.
        # This converts the coordinates actual location on the image.
        width, height = image.size
        box_coords = to_image_coords(boxes, height, width)

        # Each class will be represented by a differently colored box
        draw_boxes(image, box_coords, classes)

        print("Objects found:", len(boxes))
    
        # Save cropped image from each bounding box
        for i in range(len(boxes)):
            img = np.array(image)
            light_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Get box coordinates for original image size 
            box_coord = box_coords[i]
            bot = int(box_coord[0])
            left = int(box_coord[1])
            top = int(box_coord[2])
            right = int(box_coord[3])

            # Crop image within bounding box
            light_img = light_rgb[bot:top, left:right]
            light_output = cv2.resize(light_img, (48,108), interpolation = cv2.INTER_AREA)
            cv2.imwrite(source + "light" + str(i) + ".jpg", light_output)

            # Save image with bounding boxes
            image_rgb = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
            cv2.imwrite(source + "output_image.jpg", image_rgb)

            return light_output


# Load image
path = source + 'image0.jpg'
image = load_image(path)

# Get cropped traffic light image
light_output = get_traffic_light(image)

# Predict traffic light color
image_exp = np.expand_dims(np.asarray(light_output), axis=0)
label = predict_label(image_exp)
print("Predicted traffic light:", label)