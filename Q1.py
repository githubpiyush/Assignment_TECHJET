from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import os
import matplotlib.pyplot as plt 
import cv2
from align import AlignDlib
def load_image(path):
    img = cv2.imread(path, 1)
    return img[...,::-1]

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
args = vars(ap.parse_args())
result = []
image_list = []
index = 0
alignment = AlignDlib('models/landmarks.dat')
path = 'input'
path1 = 'input_data'

if not os.path.exists(path):
	print ('Dataset folder does not exists')
if not os.path.exists(path1):
	os.makedirs(path1)
   
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])
for (j,filename) in enumerate(os.listdir(path)):
	if filename.endswith(".png"):
		jc_orig = load_image(os.path.join(path , filename))
		bb = alignment.getLargestFaceBoundingBox(jc_orig)
		jc_aligned = alignment.align(96, jc_orig, bb, landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)
		new_file = str(j)+'.jpg'
		plt.imsave(os.path.join(path1,new_file),jc_aligned)
		#print(j)
		image = jc_aligned
		jc_aligned = jc_aligned/255
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		rects = detector(gray, 1)

		if rects is not None:
			for rect in rects:
				shape = predictor(gray, rect)
				shape = face_utils.shape_to_np(shape) 
				result1 = []
		
				result1.append(shape[8][0])
				result1.append(shape[8][1])
				result1.append(int((shape[36][0]+shape[39][0])/2))
				result1.append(int((shape[36][1]+shape[39][1])/2))
				result1.append(int((shape[42][0]+shape[45][0])/2))
				result1.append(int((shape[42][1]+shape[45][1])/2))
				result1.append(int((shape[21][0]+shape[22][0])/2))
				result1.append(int((shape[21][1]+shape[22][1])/2))
			
				result.append(result1)
				image_list.append(jc_aligned)
#print(result)
np.save('image_train.npy', image_list)
np.save('landmark.npy', result)
#print(np.shape(image_list),np.shape(result))
