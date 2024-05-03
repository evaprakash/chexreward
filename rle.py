import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
from shapely import Polygon
from PIL import Image, ImageDraw

def rle_decode(mask_rle, shape=(1024,1024)):   # (height,width) 
	'''
	mask_rle: run-length as string formated (start length)
	shape: (height,width) of array to return 
	Returns numpy array, 1 - mask, 0 - background
	'''
	s = mask_rle.split()
	starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
	starts -= 1
	ends = starts + lengths
	img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    
	for lo, hi in zip(starts, ends):
		img[lo:hi] = 1
	mask = img.reshape(shape)
	return mask
	#return (mask * 255).astype(np.uint8)

def rle_poly(rle):
	canvas = np.zeros((1024, 1024))
	mask_image = Image.fromarray(np.uint8(canvas))
	print(rle)
	assert False
	mask_polygon = Polygon(rle)
	mask_poly_coords = list(mask_polygon.exterior.coords)
	ImageDraw.Draw(mask_image).polygon(mask_poly_coords, fill = 255)
	return mask_image

def rle2mask(rle, height=1024, width=1024, fill_value=1, out_height=1024, out_width=1024):
	component = np.zeros((height, width), np.float32)
	component = component.reshape(-1)
	rle = np.array([int(s) for s in rle.strip().split(' ')])
	rle = rle.reshape(-1, 2)
	start = 0
	for index, length in rle:
		start = start + index
		end = start + length
		component[start: end] = fill_value
		start = end
	component = component.reshape(width, height).T
	if height != out_height or width != out_width:
		component = cv2.resize(component, (out_height, out_width)).astype(bool)
	return component.astype(np.uint8)

def load_ids(file_path):
	strings_list = []

	with open(file_path, 'r') as file:
		for line in file:
        		stripped_line = line.strip()
        		strings_list.append(stripped_line)

	return strings_list

df = pd.read_csv("MIMIC-CXR-JPG.csv")

for i in range(len(df["Left Lung"])):
	name = df["dicom_id"][i]
	file_path = name + '.jpg'
	if (os.path.exists(file_path) == False):
		continue
	rle_l = df["Left Lung"][i]
	rle_r = df["Right Lung"][i]
	if (rle_l == "-1" and rle_r == "-1"):
		continue
	else:
		mask_l = rle_decode(rle_l)
		mask_r = rle_decode(rle_r)
		mask = np.logical_or(mask_l, mask_r)
		mask = (mask * 255).astype(np.uint8)
		cv2.imwrite('mask.jpg', mask) 
	print("Done ", str(i), " out of ", str(len(df["ImageID"])))
	assert False
