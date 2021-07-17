import cv2
import numpy as np

# Loading Mask R-CNN model from Tensorflow (do not need to install Tensorflow package)
net = cv2.dnn.readNetFromTensorflow("dnn/frozen_inference_graph_coco.pb",
                                    "dnn/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt")

# Generate random colors
colors = np.random.randint(0, 255, (80, 3))

print(colors)

# Load image
img = cv2.imread("road.jpg")  # Loading the image
height, width, _ = img.shape   # Image dimension

print(height, width)


cv2.imshow("Image", img)
cv2.waitKey(0)