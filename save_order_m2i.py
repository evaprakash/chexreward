import os

directory = 'm2i_samples'
file_names = []

for filename in os.listdir(directory):
    if filename.endswith('.jpg') and 'mask' not in filename:
        file_names.append(filename.split(".jpg")[0][:-2])

file_names = list(set(file_names))

with open('m2i_ranking_order.txt', 'w') as f:
    for name in file_names:
        f.write(f"{name}\n")
