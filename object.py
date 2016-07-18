#!/usr/bin/env python
import rospy
import cv2
from hector_object_tracker.msgs import ImagePercept
from sensor_msgs.msgs import Image

msgImagePercept = None
pubImagePercept = None

objectDefs = None # Object definitions - color, size, etc

class Color:
	r = 0
	g = 0
	b = 0

	def __init__(self, r, g, b):
		self.r = r
		self.g = g
		self.b = b

class Rect:

	xMin = None
	yMin = None
	xMax = None
	yMax = None

	def __init__(self, xMin = 0, yMin = 0, xMax = 0, yMax = 0)
		self.xMin = xMin
		self.yMin = yMin
		self.xMax = xMax
		self.yMax = yMax


# Shared properties for all objects of a given type
class ObjectDef:
	classID = 0		# frcnn class ID
	width = 0		# Object width in meters
	height = 0		# Object height in meters
	
	classNames = ["qual_gate", "buoy", "channel", "bin", "bin_cover", "torpedo_target", "torpedo_hole", "torpedo_cover", "object_pickup", "object_dropoff", "path_marker"]
	
	def __init__(self, classID, width, height):
		self.classID = classID
		self.width = width 
		self.height = height

	def getName(self, clss = -1):
		if clss == -1:
			return classNames[self.classID]
		else:
			return classNames[clss]

class Object:
	objects.append(Object(2, "red buoy", 0.2, 0.24, Color(255, 0, 0)))
	objects.append(Object(2, "green buoy", 0.2, 0.24, Color(0, 255, 0)))
	objects.append(Object(2, "yellow buoy", 0.2, 0.24, Color(255, 255, 0)))
	objects.append(Object(3, "channel", 2.4, 1.2, Color(255, 255, 0)))
	objects.append(Object(4, "bin", 0.3, 0.6, Color(255, 255, 0)))
	objects.append(Object(5, "bin_cover", 0.3, 0.6, Color(0, 0, 0)))
	objects.append(Object(6, "torpedo target", 1.2, 1.2, Color(255, 255, 0)))
	objects.append(Object(7, "torpedo hole", 0.3, 0.3, Color(0, 0, 0)))
	objects.append(Object(8, "torpedo cover", 0.25, 0.25, Color(0, 0, 0)))
	objects.append(Object(9, "path marker", 1, 0.2, Color(0, 0, 0)))#Fix size

	objectNames = ""


	'''
	Pixel ratio for our webcam
	This is the number of pixels for an object 1 meter wide at a distance of 1 meter
	'''
	pixelRatio = 1280

	def __init__(self, classID, rect, confidence, color)
		self.classID = classID
		self.rect = rect
		self.confidence = confidence
		self.color = color

	def getObjectID(self):
		for o in self.objects:
			if o.classID = self.classID:
				

	#Estimate the distance to an object using its angular diameter (in pixels),
	# and a pixel ratio calibrated for our specific camera
	def distance(self, yMin = -1, yMax = -1):
		yMin = yMin == -1 ? self.rect.yMin : yMin
		yMax = yMax == -1 ? self.rect.yMax : yMax
		return self.pixelRatio/self.height/(yMax - yMin)

def initObjects(): 
	global objects
	objects.append(Object(2, "red buoy", 0.2, 0.24, Color(255, 0, 0)))
	objects.append(Object(2, "green buoy", 0.2, 0.24, Color(0, 255, 0)))
	objects.append(Object(2, "yellow buoy", 0.2, 0.24, Color(255, 255, 0)))
	objects.append(Object(3, "channel", 2.4, 1.2, Color(255, 255, 0)))
	objects.append(Object(4, "bin", 0.3, 0.6, Color(255, 255, 0)))
	objects.append(Object(5, "bin_cover", 0.3, 0.6, Color(0, 0, 0)))
	objects.append(Object(6, "torpedo target", 1.2, 1.2, Color(255, 255, 0)))
	objects.append(Object(7, "torpedo hole", 0.3, 0.3, Color(0, 0, 0)))
	objects.append(Object(8, "torpedo cover", 0.25, 0.25, Color(0, 0, 0)))
	objects.append(Object(9, "path marker", 1, 0.2, Color(0, 0, 0)))#Fix size

if __name__ == '__main__':
	try:
		objectDetector()
	except rospy.ROSInterruptException:
		pass
