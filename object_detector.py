#!/usr/bin/env python
import os.path as osp
import sys

sys.path.append(osp.join(osp.dirname(__file__), 'tools'))

import _init_paths
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
import caffe
import cv2
import numpy as np
import time, os, sys

import rospy
import cv2
from cv_bridge import CvBridge
from hector_worldmodel_msgs.msg import ImagePercept
from sensor_msgs.msg import Image

import object

class TopicDetect:
	'''
	Topic we want to observe for object detections, and the rate to do it at.
	TODO: Move detection image publishing in here, and make new topic have similar name to old one
	'''
	def __init__(self, name, rate):
		self.name = name
		self.rate = rospy.Rate(rate)

class ObjectDetector:
	'''
	This is a ROS node that keeps a faster rcnn model alive and ready to work on the GPU. 
	It accepts images from specified topics, and publishes subsequent object detections.
	'''

	detection_threshold = .05 	# Minimum softmax score
	pubImagePercept = None		# Publisher for object detection ImagePercept output
	pubObjectDetector = None	# Publisher for object detection Image output
	imageSubChannels = []		# ROS topics we subscribe to for images to classify
	imageMsg = None			# Most recent image message from Image subscription topic
	CvBridge = None				# ROS CVBridge object
	objectDefinitions = None	# List of Objects

	def __init__(self, gpu_id = 0, cfg_file = "experiments/cfgs/msu.yml", 
					prototxt = "models/msupool/ZF/faster_rcnn_alt_opt/faster_rcnn_test.pt", 
					caffemodel = "output/faster_rcnn_alt_opt/msupool/ZF_faster_rcnn_final.caffemodel", 
					imageSubChannels = [TopicDetect("sensors/camF", 60)]):
		'''
		@param gpu_id: The ID of the GPU to use for caffe model
		@param cfg_file: Path to the config file used for the caffe model
		@param prototxt: Path to network structure definition for caffe model
		@param caffemodel: Path to caffemodel containing trained network weights
		@param imageSubChannels: ROS topics we subscribe to for images to classify
		'''
		global cfg

		#Initialize faster r-cnn
		cfg_from_file(cfg_file)
		self.cfg = cfg
		self.cfg.GPU_ID = gpu_id
		
		caffe.set_mode_gpu()
		caffe.set_device(self.cfg.GPU_ID)
		self.net = caffe.Net(prototxt, caffemodel, caffe.TEST)
		self.net.name = os.path.splitext(os.path.basename(caffemodel))[0]

		#Initialize ROS
		self.pubImagePercept = rospy.Publisher('worldmodel/image_percept', ImagePercept, queue_size=10)
		self.pubObjectDetector = rospy.Publisher('object_detector', Image, queue_size=10)
		self.imageSubChannels = imageSubChannels
		for sub in self.imageSubChannels:
			rospy.Subscriber(sub.name, Image, self.subImageCB)
		rospy.init_node("object_detector")

		self.CvBridge = CvBridge()

		rate = rospy.Rate(10)
		while not rospy.is_shutdown():
			if self.imageMsg is not None:
				image = self.CvBridge.imgmsg_to_cv2(self.imageMsg)
				objects = self.detect(image)
				self.publishDetections(objects)
				self.imageMsg = None
			rate.sleep()

	def subImageCB(self, image):
		#TODO: multiple image topics		
		if self.imageMsg is None:
			self.imageMsg = image

	def publishDetections(self, objects):
		for o in objects:
			#ImagePercept
			msgImagePercept = ImagePercept()
			msgImagePercept.header = self.imageMsg.header
			msgImagePercept.camera_info = None #TODO
			msgImagePercept.info.class_id = o.classID
			msgImagePercept.info.object_id = o.objectID
			msgImagePercept.info.name = o.name()
			msgImagePercept.x = (o.xMin + o.xMax) / 2 #Center point of object
			msgImagePercept.y = (o.yMin + o.yMax) / 2 
			msgImagePercept.width = (o.xMax - o.xMin) / msgImagePercept.camera_info.width
			msgImagePercept.height = (o.yMax - o.yMin) / msgImagePercept.camera_info.height
			msgImagePercept.distance = o.distance()
			self.pubImagePercept.publish(msgImagePercept)

			#Image
			image = self.CvBridge.bridge.imgmsg_to_cv2(self.imageMsg)

			for o in objects:
				cv2.rectangle(image, (o.xMin, o.yMin), (o.xMax, o.yMax), (0, 255, 0))
				cv2.putText(image, o.name() + ":" + o.distance(), FONT_HERSHEY_SIMPLEX, 1, (0,255,0)) 

			self.pubObjectDetector.publish(self.CvBridge.bridge.cv2_to_imgmsg(image))

	# Wrapper for faster-rcnn detections
	# Excluded max_per_image - look into it if it becomes a problem
	def detect(self, image):
		'''
		@param image: cv2 image to detect objects in
		'''
		scores, boxes = im_detect(self.net, image)

		objects = []

		for j in xrange(1, Object.num_classes):
			inds = np.where(scores[:, j] > self.detection_threshold)[0]
			cls_scores = scores[inds, j]
			cls_boxes = boxes[inds, j*4:(j+1)*4]
			cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32, copy=False)
			keep = nms(cls_dets, cfg.TEST.NMS)
			cls_dets = cls_dets[keep, :]
			#all_boxes[j][i] = cls_dets
			#print cls_dets
			
			# Each detection is an array containing [xMin, yMin, xMax, yMax, confidence]
			for det in cls_dets:
				avgColor = self.avgColor(image, det[0], det[1], det[2]-det[0], det[3]-det[1])
				objects.append(Object(j, Rect(det[0], det[1], det[2], det[3]), det[4], avgColor))

		return objects

	def avgColor(self, image, x, y, width, height):
		'''
		Get average color of middle crop of object
		'''
		crop = .5
		xSub =  (width * crop)/2
		ySub = (height * crop)/2
		return cv2.mean(image[x+xSub:y+ySub, width-xSub:height-ySub])

if __name__ == '__main__':
	detector = ObjectDetector()
