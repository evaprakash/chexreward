import json
import random
import os

prefix = "train_data"
dest = "tm2i_temp"
task = 'heart'
json_file_path = prefix + "/splits/val.jsonl"

def filter_elements_with_keywords(file_path, keywords):
    filtered_elements = []
    with open(file_path, 'r') as file:
        for line in file:
            json_data = json.loads(line)
            conditioning_image = json_data.get("conditioning_image", '')
            if any(keyword in conditioning_image for keyword in keywords):
                filtered_elements.append(json_data)
    return filtered_elements

keywords_to_search = ["heart"]

directory = 'm2i_samples'
file_names = os.listdir(directory)
filtered_file_names = list(set([file_name.split("_")[0] for file_name in file_names if 'mask' not in file_name]))
images = filter_elements_with_keywords(json_file_path, keywords_to_search)
unused_images = []
for data in images:
    img_id = data['image'].split("/")[1].split(".jpg")[0]
    if img_id in filtered_file_names:
        continue
    else:
        unused_images.append(data)

random.shuffle(unused_images)

new_images = unused_images[:55]

fp = open("tm2i.txt", "a")

for data in new_images:
    mask_file = data['conditioning_image'].split("/")[2].split("_")[0] + "_" + task + "_mask.jpg"
    cmd = "cp " + prefix + "/" + data['conditioning_image'] + " " + dest + "/" + mask_file
    fp.write(mask_file + "\n")
    os.system(cmd)
