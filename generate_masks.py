import pandas as pd
import csv
import cv2
import numpy as np

def rle_decode(mask_rle, shape=(1024,1024)):   # (height,width) 
	s = mask_rle.split()
	starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
	starts -= 1
	ends = starts + lengths
	img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    
	for lo, hi in zip(starts, ends):
		img[lo:hi] = 1
	mask = img.reshape(shape)
	return mask

df = pd.read_csv("/deep/group/mimic-cxr/MIMIC-CXR-JPG.csv")
dicom_df = pd.read_csv('tm2i.csv')
dicom_ids = dicom_df['dicom_id'].tolist()

df = df[df['dicom_id'].isin(dicom_ids)]

for i in range(len(df["Left Lung"])):
    name = df["dicom_id"].iloc[i]
    rle_l = df["Left Lung"].iloc[i]
    rle_r = df["Right Lung"].iloc[i]
    rle_h = df["Heart"].iloc[i]
    if (rle_l == "-1" and rle_r == "-1" and rle_h == "-1"):
        continue
    else:
        mask_l = rle_decode(rle_l)
        mask_r = rle_decode(rle_r)
        mask_h = rle_decode(rle_h)
        mask = np.logical_or(mask_l, mask_r)
        mask = (mask * 255).astype(np.uint8)
        mask_h = (mask_h * 255).astype(np.uint8)
        cv2.imwrite('mimic_masks/' + name + '_lung_mask.jpg', mask) 
        cv2.imwrite('mimic_masks/' + name + '_heart_mask.jpg', mask_h) 
        print("Done ", str(i), " out of ", str(len(df["Left Lung"])))
