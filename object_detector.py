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


class ObjectDetector:

	detection_threshold = .05 # Minimum softmax score


	def __init__(self, gpu_id, cfg_file, prototxt, caffemodel):
		global cfg
		cfg_from_file(cfg_file)
		self.cfg = cfg
		self.cfg.GPU_ID = gpu_id
		
		caffe.set_mode_gpu()
		caffe.set_device(self.cfg.GPU_ID)
		self.net = caffe.Net(prototxt, caffemodel, caffe.TEST)
		self.net.name = os.path.splitext(os.path.basename(caffemodel))[0]



	# Wrapper for faster-rcnn detections
	# Excluded max_per_image - look into it if it becomes a problem
	# TODO:
	# 	num classes - actually load imdb?
	#	convert each detection to ROS message
	#		Format: [[x1 y1 x2 y2 confidence]] for each class, and for each object of that class
	#	merge with existing ROS thing
	#	store parameters in main function somewhere
	def detect(self, image):
		scores, boxes = im_detect(self.net, image)

		for j in xrange(1, self.num_classes):
			inds = np.where(scores[:, j] > self.detection_threshold)[0]
			cls_scores = scores[inds, j]
			cls_boxes = boxes[inds, j*4:(j+1)*4]
			cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32, copy=False)
			keep = nms(cls_dets, cfg.TEST.NMS)
			cls_dets = cls_dets[keep, :]
			#all_boxes[j][i] = cls_dets
			print cls_dets





if __name__ == '__main__':
	gpu_id = 0
	cfg_file = "experiments/cfgs/msu.yml"
	prototxt = "models/msupool/ZF/faster_rcnn_alt_opt/faster_rcnn_test.pt"
	caffemodel = "output/faster_rcnn_alt_opt/msupool/ZF_faster_rcnn_final.caffemodel"

	detector = ObjectDetector(gpu_id, cfg_file, prototxt, caffemodel)

	image = cv2.imread("data/MSUPool/Images/GOPR0424.JPG")

	detector.detect(image)
