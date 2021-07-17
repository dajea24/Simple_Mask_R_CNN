# Architecture and weight files for the model
import cv2

textGraph = "./mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"
modelWeights = "./frozen_inference_graph.pb"
# Load the network
net = cv2.dnn.readNetFromTensorflow(modelWeights, textGraph)
