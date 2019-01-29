'''
ECE276A WI19 HW1
Blue Barrel Detector
'''

import os
import cv2
from skimage.measure import label, regionprops
from skimage import data, util, img_as_ubyte
from skimage.morphology import erosion, dilation, opening, closing, square
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pickle
import math
import train

class BarrelDetector(object):
	def __init__(self):
		'''
			Initialize your blue barrel detector with the attributes you need
			eg. parameters of your classifier
		'''
		
	def segment_image(self, img):
		'''
			Calculate the segmented image using a classifier
			eg. Single Gaussian, Gaussian Mixture, or Logistic Regression
			call other functions in this class if needed
			
			Inputs:
				img - original image
			Outputs:
				mask_img - a binary image with 1 if the pixel in the original image is blue and 0 otherwise
		'''
		#use model to find blue parts of image
		img2 = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
		model = train.LR_Model()
		model.load('weights.pickle')
		mask_img = model.test(img)
		selem = square(4)
		mask_img = closing(mask_img,selem=selem)
		img2 = np.ones(img.shape)
		#show mask
		img2[:,:,0] = img2[:,:,0]*mask_img
		img2[:,:,1] = img2[:,:,1]*mask_img
		img2[:,:,2] = img2[:,:,2]*mask_img
		#plt.imshow(img2)
		#plt.show()
		return np.uint8(mask_img)

	def get_bounding_box(self, img):
		'''
			Find the bounding box of the blue barrel
			call other functions in this class if needed
			
			Inputs:
				img - original image
			Outputs:
				boxes - a list of lists of bounding boxes. Each nested list is a bounding box in the form of [x1, y1, x2, y2] 
				where (x1, y1) and (x2, y2) are the top left and bottom right coordinate respectively. The order of bounding boxes in the list
				is from left to right in the image.
				
			Our solution uses xy-coordinate instead of rc-coordinate. More information: http://scikit-image.org/docs/dev/user_guide/numpy_images.html#coordinate-conventions
		'''

		boxes = []
		binary_img = self.segment_image(img)
		#clean up image
		#find connected components
		contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)		
        #iterate through all the top-level contours,
		#draw each connected component with its own random color
		for idx in range(len(contours)):
			color = 255*np.random.random([3])
			cv2.drawContours(binary_img, contours, idx, color, -1 )
		
		#go through each region
		#find apply shape statistic to include or exclude as barrel
		props = regionprops(binary_img)
		for reg in props:
			#make sure area seen is sizable enough
			if reg.area > 8:
				major = reg.major_axis_length
				minor = reg.minor_axis_length + 0.001
				#make sure area is shaped like barrel (longer than wider)
				if major/minor < 2.4 and major/minor > 1.7:
					minr, minc, maxr, maxc = reg.bbox
					boxes.append([minc,maxr,maxc,minr])
		fig,ax = plt.subplots()
		img2 = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
		# Display the image
		#ax.imshow(img2)
		# Create a bounding box
		for i in range(len(boxes)):
			rect = patches.Rectangle((boxes[i][0], boxes[i][3]), boxes[i][2] - boxes[i][0], boxes[i][1] - boxes[i][3],linewidth=1,edgecolor='r',facecolor='none')
			ax.add_patch(rect)
		#plt.show()

		return boxes


if __name__ == '__main__':
	folder = "trainset"
	my_detector = BarrelDetector()

	for filename in os.listdir(folder):
		# read one test image
		img = cv2.imread(os.path.join(folder,filename))
		#cv2.imshow('image', img)
		#cv2.waitKey(0)
		#cv2.destroyAllWindows()

		#Display results:
		#(1) Segmented images
		mask_img = my_detector.segment_image(img)
		#(2) Barrel bounding box		
		boxes = my_detector.get_bounding_box(img)
		#The autograder checks your answers to the functions segment_image() and get_bounding_box()
		#Make sure your code runs as expected on the testset before submitting to Gradescope
		
