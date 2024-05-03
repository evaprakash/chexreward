import random
import csv
import pandas as pd

def load_and_filter_csv(file_path, prefix):
    data = pd.read_csv(file_path)
    masks = [prefix + mask for mask in data['mask']]
    prompts = data['prompt'].tolist()
    return masks, prompts

'''
prompt_df = pd.read_csv('prompt_final_filtered.csv')
prompts = prompt_df['GPT-4 Summary'].tolist()
image_ids = []
'''
prompt_df = pd.read_csv('prompt_dicom_pairs_final.csv')
prompts = prompt_df['prompt'].tolist()
dicoms = prompt_df['dicom_id'].tolist()
ptx_images, ptx_prompts = load_and_filter_csv('ptx_prompt_final_filtered.csv', prefix="train_data/")

task = ["heart", "lung"]

with open('tm2i_rank_order.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['image_id', 'prompt'])
    for item1, item2 in zip(dicoms, prompts):
        random_value = random.random()
        selected_task = task[0] if random_value < 0.8 else task[1]
        writer.writerow([item1 + "_" + selected_task, item2])
    for item1, item2 in zip(ptx_images, ptx_prompts):
        image_id = item1.split("/")[-1].split("_")[0] 
        writer.writerow([item1 + "_ptx", item2])
