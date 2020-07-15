from lxml.etree import Element, SubElement, tostring, ElementTree
from xml.dom.minidom import parseString
import cv2 as cv
import numpy as np
from PIL import Image
import os
import skimage.io as io
from skimage import data_dir
import matplotlib.pyplot as plt
import math
import RPi.GPIO as GPIO
from time import sleep
from picamera import PiCamera


############### INPUT SETTING ###########################

objectname = input('Enter object name:')                                                    # Nmae of the object
file_path = os.path.dirname(os.getcwd()) + '/DataCollection/' + objectname                  # Path of the images (data)
result_path = os.path.dirname(os.getcwd()) + '/DataCollection/' + objectname + '/Labeled/'  # Path of the result labeled images

if not os.path.exists(file_path):
	os.mkdir(file_path)

if not os.path.exists(result_path):                                                 # Create folders if its not exist
	os.mkdir(result_path)

#### 1. Camera Setting ###################
camera = PiCamera()
camera.resolution = (1024,768)                                                      # Resolution
camera.brightness = 58
STEP = 16
DIR = 18
SPR = 199                                                                           # Number of Pics one round (Start from 0)
GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)
GPIO.setup(STEP, GPIO.OUT)
GPIO.setup(DIR, GPIO.OUT)
GPIO.output(DIR, 1)                                                                 # Turn table direction
delay = 0.33

#### 2. Labeling Processing Setting ######
kernel2 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (25,25))                       # kernel for morphology closing
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3))                          # kernel for morphology opening
kernel3 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5))
ListX = []                                                                          # Create list to filling number of x, y, w, h
ListY = []
ListW = []
ListH = []
threshold_average = 15                                                              # Threshold number of absulute value of difference that x, y, w, h should replace by average number
Number_Pic = 9                                                                     # Number of picture to calculate average
Object_Pic = math.ceil((Number_Pic-1)/2)                                            # Aim picture to be labeled


############### PROCESSING ###########################

#### 1. Data Collection ##################
camera.start_preview()
sleep(2)
for x in range(SPR):
	for i, file in enumerate(camera.capture_continuous(objectname+'{counter}.jpg')):
			print('Captured %s' % file)
			sleep(delay)
			GPIO.output(STEP, True)
			sleep(delay)
			GPIO.output(STEP, False)
			sleep(delay)

sleep(0.5)
camera.stop_preview()
camera.close()
GPIO.cleanup()

#### 2. Get Object Location Information First Time ###
dataset = file_path + '/*.jpg'                                                    
dataset = io.ImageCollection(dataset)                                               # Read dataset with RGB format
print(len(dataset))                                                                 ### Print how many data we collecte

for i in range(len(dataset)-2):                                                     # Start process images with for loop
	Origin = cv.cvtColor(dataset[i], cv.COLOR_RGB2BGR)                              # Set origin images change RGB to BGR
	imgA = cv.cvtColor(Origin, cv.COLOR_BGR2GRAY)                                   # Set images to gray scale
	i=i+1
	imgB = cv.cvtColor(dataset[i], cv.COLOR_RGB2GRAY)
	i=i+1
	imgC = cv.cvtColor(dataset[i], cv.COLOR_RGB2GRAY)
	i=i-2

	imgA_Blur = cv.GaussianBlur(imgA, (9,9), 0)                                     # Use GaussianBlur to denoising
	imgB_Blur = cv.GaussianBlur(imgB, (9,9), 0)
	imgC_Blur = cv.GaussianBlur(imgC, (9,9), 0)

	imgA_Canny=cv.Canny(imgA_Blur,10, 35)                                           # Use canny to detecte edges
	imgB_Canny=cv.Canny(imgB_Blur,10, 35)
	imgC_Canny=cv.Canny(imgC_Blur,10, 35)

	imgA_dst = cv.GaussianBlur(imgA_Canny, (9,9), 0)                                # Do GaussianBlur again to decrease disturbance
	imgB_dst = cv.GaussianBlur(imgB_Canny, (9,9), 0)
	imgC_dst = cv.GaussianBlur(imgC_Canny, (9,9), 0)

	imgA_Arr = np.array(imgA_dst)                                                   # Change pic to array form
	imgB_Arr = np.array(imgB_dst)
	imgC_Arr = np.array(imgC_dst)

	_, imgA_thresh = cv.threshold(imgA_Arr, 30,255,0)                               # Make all the point have left with above number of 30
	_, imgB_thresh = cv.threshold(imgB_Arr, 30,255,0)                               # to 255. Pre process to keep the feature of position
	_, imgC_thresh = cv.threshold(imgC_Arr, 30,255,0)

	imgAB_Position_Diff_Array = np.array(imgA_thresh) == np.array(imgB_thresh)          # Find out which position image A and B is different
	imgAB_Position_Feature_Array = np.where(imgAB_Position_Diff_Array == True, 0, imgA) # Keep the differece position of pixels, the result will be the position feature of image A and B
																						# After find the difference of position, use pixel of imgA replace them
	imgAC_Position_Diff_Array = np.array(imgA_thresh) == np.array(imgC_thresh)          # Same as above, but with image A and C
	imgAC_Position_Feature_Array = np.where(imgAC_Position_Diff_Array == True, 0, imgA)

	_, imgAB_thresh = cv.threshold(imgAB_Position_Feature_Array, 30,255,0)              # Do binarization again to decrease disturbance
	_, imgAC_thresh = cv.threshold(imgAC_Position_Feature_Array, 30,255,0)

	Result_Position_Feature = np.array(imgAB_thresh) == np.array(imgAC_thresh)          # Find out which position of image Position_Feature_AB and image Position_Feature_AC different
	Result_Feature = np.where(Result_Position_Feature == False, 0, imgAB_Position_Feature_Array) # From above boolean array, if it is different change to 0, otherwise use position_feature replace it

	_, result_thresh = cv.threshold(Result_Feature, 50,255,0)                           # Reduce the error

	closing = cv.morphologyEx(result_thresh, cv.MORPH_CLOSE, kernel2)                   # Using morphology to adjust image. closing is to dilated first then eroded used to connect objects that are misclassified into many small blocks
	opening = cv.morphologyEx(closing, cv.MORPH_OPEN, kernel)                           # Opening is to eroded first then dilated, used to remove spots formed by image noise
	closing1 = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel3)

	cnts = cv.findContours(opening1.copy(), cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)       # Detect object outlines
	cnts = cnts[0]
	Rec = sorted(cnts, key=cv.contourArea, reverse=True)[0:1]                           # Sorted all the rectangle has been detected, and choose the first one

	for c in Rec:
		(x,y,w,h) = cv.boundingRect(c)                                                  # Save result images
	ListX.append(x)                                                                     # Save x, y, w, h, in saperate list
	ListY.append(y)
	ListW.append(w)
	ListH.append(h)

#### 3. Labeling Object and Give Output ##############
for j in range(len(dataset)-2-Number_Pic+1):

	write_name = 'Labeled'+str(j+Object_Pic)+'.jpg'                                     # Name of save result images
	write_xml_name = 'Labeled'+str(j+Object_Pic) + '.xml'                               # Name of save xml format file
	img = dataset[j+Object_Pic]                                                         # Get images height width and channel
	shape = img.shape
	height = str(shape[0])
	width = str(shape[1])
	channel = str(shape[2])

	Origin = cv.cvtColor(dataset[j+Object_Pic], cv.COLOR_RGB2BGR)                       # Get the picture that should be labeled

	average_x = math.ceil(sum(ListX[j:j+Number_Pic])/Number_Pic)                        # Calculate average number to reduce bad rate
	average_y = math.ceil(sum(ListY[j:j+Number_Pic])/Number_Pic)
	average_w = math.ceil(sum(ListW[j:j+Number_Pic])/Number_Pic)
	average_h = math.ceil(sum(ListH[j:j+Number_Pic])/Number_Pic)
	if abs(average_x - ListX[j+Object_Pic]) > threshold_average:                        # If the difference of average_x - object parameter is higher than threshold number
		x = average_x                                                                   # Use average number to replacement
	else:
		x = ListX[j+Object_Pic]
	if abs(average_y - ListY[j+Object_Pic]) > threshold_average:
		y = average_y
	else:
		y = ListY[j+Object_Pic]
	if abs(average_w - ListW[j+Object_Pic]) > threshold_average:
		w = average_w
	else:
		w = ListW[j+Object_Pic]
	if abs(average_h - ListH[j+Object_Pic]) > threshold_average:
		h = average_h
	else:
		h = ListH[j+Object_Pic]
	cv.rectangle(Origin, (x,y), (x+w, y+h), (0,0,255), 2)                               # Draw rectangle
	os.chdir(result_path)                                                               # Open output path
	cv.imwrite( write_name, Origin)                                                     # Save labeled image

#### 4. Save XML file With Necessary Imformations ######    
	node_root = Element('annotation')                                                   # Start build xml file tree structure

	node_folder = SubElement(node_root, 'folder')
	node_folder.text = 'Labeled'

	node_filename = SubElement(node_root, 'filename')
	node_filename.text = objectname + str(j+Object_Pic) + '.jpg'

	node_path = SubElement(node_root, 'path')
	node_path.text = file_path + objectname + str(j+Object_Pic) + '.jpg'

	node_source = SubElement(node_root, 'source')
	node_database = SubElement(node_source, 'database')
	node_database.text = ' Unknow'

	node_size = SubElement(node_root, 'size')
	node_width = SubElement(node_size, 'width')
	node_width.text = width

	node_height = SubElement(node_size, 'height')
	node_height.text = height

	node_depth = SubElement(node_size, 'depth')
	node_depth.text = channel

	node_object = SubElement(node_root, 'object')
	node_name = SubElement(node_object, 'name')
	node_name.text = objectname

	node_segmented = SubElement(node_root, 'segmented')
	node_segmented.text = '0'

	node_difficult = SubElement(node_object, 'difficult')
	node_difficult.text = '0'

	node_bndbox = SubElement(node_object, 'bndbox')
	node_xmin = SubElement(node_bndbox, 'xmin')
	node_xmin.text = str(x)
	node_ymin = SubElement(node_bndbox, 'ymin')
	node_ymin.text = str(y)
	node_xmax = SubElement(node_bndbox, 'xmax')
	node_xmax.text = str(x+w)
	node_ymax = SubElement(node_bndbox, 'ymax')
	node_ymax.text = str(y+h)

	print(x,y,x+w,y+h)

	xml = ElementTree(node_root)                                                            # Save xml file
	xml.write(write_xml_name, pretty_print=True, xml_declaration=False, encoding='utf-8')

	print(j)
