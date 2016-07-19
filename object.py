#!/usr/bin/env python
import rospy
import json


# Shared properties for all objects of a given type
class Object:
	'''
	This class contains the definitions of the objects we are searching for
	'''
	classes = [] # Static list of classes we're looking for
	objects = [] # Static list of objects we're looking for

	def __init__(self, classID, width, height, color):
		self.classID = classID
		self.width = width 
		self.height = height
		self.color = color

	@staticmethod
	def initDefinitions():
		with open("objects.json") as data:
			data = json.load(data)
		for c in data["classes"]:
			Object.classes.append(c)

		for obj in data["objects"]:
			Object.objects.append(Object(Object.classes.index(obj["class"]), obj["width"], obj["height"], obj["color"]))
	
	@staticmethod
	def getObject(classID, color):
		objects = []
		for obj in Objects.objects:
			if obj.classID == classID:
				objects.append(obj)

		if len(objects) == 1:
			return objects[0]
		else:
			#get object with nearest color
			closest = objects[0]
			distance = 999999
			for obj in objects:
				d = Object.colorDistance(obj.color, color)
				if d < distance:
					closest = obj
					distance = d

			return closest
			
	def getColorDistance(c1, c2):
		# Get euclidean distance between two colors
		# I'm sure there's a better way to do this
		return math.sqrt(math.pow(c1[0] - c2[0], 2) + math.pow(c1[1] - c2[1], 2) + math.pow(c1[2] - c2[2]))
			

	def name(self, classID = -1):
		if classID == -1:
			classID = self.classID
		return classNames[classID]


class ObjectDetection:
	'''
	This class is a container for info pertaining to a detection event for an object
	'''
	#This is the number of pixels for an object 1 meter wide at a distance of 1 meter for our particular webcam
	pixelRatio = 1280

	def __init__(self, classID, xMin, yMin, xMax, yMax, confidence, color):
		self.classID = classID
		self.obj = Object.getObject(classID, color)
		self.xMin = xMin
		self.yMin = yMin
		self.xMax = xMax
		self.yMax = yMax
		self.confidence = confidence
		self.color = color

	#Estimate the distance to an object using its angular diameter (in pixels),
	# and a pixel ratio calibrated for our specific camera
	def distance(self, yMin = -1, yMax = -1):
		yMin = self.yMin if yMin == -1 else yMin
		yMax = self.yMax if yMax == -1 else yMax
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
		Object.initDefinitions()
	except rospy.ROSInterruptException:
		pass
