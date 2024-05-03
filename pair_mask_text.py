import pandas as pd
import csv
import string
import os
from openai import OpenAI
from collections import Counter

def filter_study_ids(study_ids):
    filtered_ids = [float(id[1:]) if id.startswith('s') else id for id in study_ids]
    return filtered_ids

def synchronize_lists(list1, list2):
    counter = Counter(list1)
    repeated_indices = {item: [i for i, x in enumerate(list1) if x == item] for item, count in counter.items() if count > 1}
    indices_to_remove = [index for indices in repeated_indices.values() for index in indices[1:]]
    indices_to_remove.sort(reverse=True)

    for index in indices_to_remove:
        del list1[index]
        del list2[index]

    return list1, list2

split_df = pd.read_csv('/deep/group/mimic-cxr/mimic-cxr-2.0.0-split.csv')
validate_split_df = split_df[split_df['split'] == 'validate'].copy()
validate_split_df['study_id'] = 's' + validate_split_df['study_id'].astype(str)

section_df = pd.read_csv('/deep/group/mimic-cxr/mimic_cxr_sectioned.csv')
matched_studies = section_df[section_df['study'].isin(validate_split_df['study_id'])]
impressions = matched_studies['impression']
impressions = impressions.tolist()

matched_study_ids = matched_studies['study'].tolist()
filtered_impressions = []
filtered_matched_study_ids = []
for i in range(len(impressions)):
    impression = impressions[i]
    study_id = matched_study_ids[i]
    if impression == impression:
        filtered_impressions.append(impression.replace('\n', ''))
        filtered_matched_study_ids.append(study_id)
impressions = filtered_impressions
matched_study_ids = filtered_matched_study_ids
impressions, matched_study_ids = synchronize_lists(impressions, matched_study_ids)

metadata_df = pd.read_csv('/deep/group/mimic-cxr/mimic-cxr-2.0.0-metadata.csv')
validation_study_ids = filter_study_ids(matched_study_ids)
df = pd.read_csv('/deep/group/mimic-cxr/MIMIC-CXR-JPG.csv')
mask_dicom_ids = df['dicom_id'].tolist()
filtered_by_study_id = metadata_df[metadata_df['study_id'].isin(validation_study_ids)]
filtered_by_dicom_id = filtered_by_study_id[filtered_by_study_id['dicom_id'].isin(mask_dicom_ids)]
final_filtered_df = filtered_by_dicom_id[filtered_by_dicom_id['ViewPosition'] == 'PA']

print(final_filtered_df['dicom_id'], final_filtered_df['study_id'], len(final_filtered_df))

dicom_ids = final_filtered_df['dicom_id'].tolist()
study_ids = final_filtered_df['study_id'].tolist()
study_ids = ['s' + str(int(s)) for s in study_ids]
img_folders = final_filtered_df['Img_Folder'].tolist()

filename = 'tm2i.csv'

with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['dicom_id', 'study_id', 'img_folder'])
    for d, s, f in zip(dicom_ids, study_ids, img_folders):
        writer.writerow([d, s, f])
