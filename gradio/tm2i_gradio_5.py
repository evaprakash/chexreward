import gradio as gr
import random
import time
import os
import pandas as pd
from glob import glob
from PIL import Image

person = "5"
L = 60
H = 75
image_prefix = "tm2i_samples_final/"
mask_prefix = "tm2i_samples_final/"
image_ids = []
img_list = "tm2i_rank_order_final.csv"
df = pd.read_csv(img_list)
image_ids = df['image_id'].tolist()
for i in range(len(image_ids)):
    if 'lung' not in image_ids[i] and 'heart' not in image_ids[i]:
        image_ids[i] = image_ids[i] + '_ptx'
prompts = df['prompt'].tolist()
save_path_m = "tm2i_ranks/" + person + "/mask/"
save_path_c = "tm2i_ranks/" + person + "/text/"
image_ids = image_ids[L:H]
prompts = prompts[L:H]
num_rank = len(image_ids)

def is_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def load_img(img_path, size=512):
    img = Image.open(img_path).convert('RGB')
    return img

def find_completed_idxs(save_path):
    files = os.listdir(save_path)
    incorrect_files = []
    if len(files) == 0:
        return [-1], []
    else:
        file_list = []
        for f in files:
            f_name = int(f.split(".")[0])
            with open(save_path + "/" + f) as fp:
                for line in fp:
                    items = line.strip().split(",")
                    if (len(items) != 5 and f_name != -1):
                        incorrect_files.append(f_name)
                    else:
                        if ((not is_int(items[1].strip()) or not is_int(items[2].strip()) or not is_int(items[3].strip()) or not is_int(items[4].strip())) and f_name != -1):
                            incorrect_files.append(f_name)
            file_list.append(f_name)
        file_list = sorted(file_list)
        incorrect_files = sorted(incorrect_files)
        return file_list, incorrect_files

def load_next(rank_m_1, rank_m_2, rank_m_3, rank_m_4, rank_c_1, rank_c_2, rank_c_3, rank_c_4, caption, img_1, mask_1, img_2, mask_2, img_3, mask_3, img_4, mask_4, example, ids=image_ids, image_prefix=image_prefix, save_path_m=save_path_m, save_path_c=save_path_c):
    file_list_m, incorrect_files_m = find_completed_idxs(save_path_m)
    file_list_c, incorrect_files_c = find_completed_idxs(save_path_c)
    file_list = list(set(file_list_m).union(set(file_list_c)))
    incorrect_files = list(set(incorrect_files_m).union(set(incorrect_files_c)))
    print(str(file_list) + " " + str(incorrect_files))
    if (int(example) not in file_list or int(example) in incorrect_files):
        r = str(image_ids[int(example)]) + "," + rank_m_1 + "," + rank_m_2 + "," + rank_m_3 + "," + rank_m_4
        r_fp = open(save_path_m + "/" + str(int(example)) +".txt", "w")
        r_fp.write(r + "\n")
        r_fp.close()
        
        r = str(image_ids[int(example)])+  "," + rank_c_1 + "," + rank_c_2 + "," + rank_c_3 + "," + rank_c_4
        r_fp = open(save_path_c + "/" + str(int(example)) +".txt", "w")
        r_fp.write(r + "\n")
        r_fp.close()
    file_list_m, incorrect_files_m = find_completed_idxs(save_path_m)
    file_list_c, incorrect_files_c = find_completed_idxs(save_path_c)
    file_list = list(set(file_list_m).union(set(file_list_c)))
    incorrect_files = list(set(incorrect_files_m).union(set(incorrect_files_c)))
    if (len(incorrect_files) != 0):
        example = incorrect_files[-1]
    else:
        example = file_list[-1] + 1
    if int(example) == num_rank:
        rank_m_1, rank_m_2, rank_m_3, rank_m_4 = "DONE!", "DONE!", "DONE!", "DONE!"
        rank_c_1, rank_c_2, rank_c_3, rank_c_4 = "DONE!", "DONE!", "DONE!", "DONE!"
        example = -1
        mask_1 = gr.Image(label="Mask", value=load_img("blank.jpg"), interactive=False)
        img_1 = gr.Image(label="Sample #1", value=load_img("blank.jpg"), interactive=False)
        mask_2 = gr.Image(label="Mask", value=load_img("blank.jpg"), interactive=False)
        img_2 = gr.Image(label="Sample #2", value=load_img("blank.jpg"), interactive=False)
        mask_3 = gr.Image(label="Mask", value=load_img("blank.jpg"), interactive=False)
        img_3 = gr.Image(label="Sample #3", value=load_img("blank.jpg"), interactive=False)
        mask_4 = gr.Image(label="Mask", value=load_img("blank.jpg"), interactive=False)
        img_4 = gr.Image(label="Sample #4", value=load_img("blank.jpg"), interactive=False)
        caption = gr.Textbox(label="Caption:", value="N/A", interactive=False)
    else:
        rank_m_1, rank_m_2, rank_m_3, rank_m_4 = "", "", "", ""
        rank_c_1, rank_c_2, rank_c_3, rank_c_4 = "", "", "", ""
        img_1 = gr.Image(label="Sample #1", value=load_img(image_prefix + str(image_ids[int(example)]) + "_1.jpg"), interactive=False)
        mask_1 = gr.Image(label="Mask", value=load_img(mask_prefix + str(image_ids[int(example)]) + "_mask_1.jpg"), interactive=False)
        img_2 = gr.Image(label="Sample #2", value=load_img(image_prefix+ str(image_ids[int(example)]) + "_2.jpg"), interactive=False)
        mask_2 = gr.Image(label="Mask", value=load_img(mask_prefix + str(image_ids[int(example)]) + "_mask_2.jpg"), interactive=False)
        img_3 = gr.Image(label="Sample #3", value=load_img(image_prefix + str(image_ids[int(example)]) + "_3.jpg"), interactive=False)
        mask_3 = gr.Image(label="Mask", value=load_img(mask_prefix + str(image_ids[int(example)]) + "_mask_3.jpg"), interactive=False)
        img_4 = gr.Image(label="Sample #4", value=load_img(image_prefix + str(image_ids[int(example)]) + "_4.jpg"), interactive=False)
        mask_4 = gr.Image(label="Mask", value=load_img(mask_prefix + str(image_ids[int(example)]) + "_mask_4.jpg"), interactive=False)
        caption = gr.Textbox(label="Caption:", value=prompts[int(example)], interactive=False)
    return [rank_m_1, rank_m_2, rank_m_3, rank_m_4, rank_c_1, rank_c_2, rank_c_3, rank_c_4, caption, img_1, mask_1, img_2, mask_2, img_3, mask_3, img_4, mask_4, example]

with gr.Blocks() as demo:
    last_idx = -1
    example = gr.Number(label="Example #. Click next for #-1 (blank starting page).", value=last_idx, interactive=False)
    caption = gr.Textbox(label="Caption:", value="N/A", interactive=False)
    with gr.Column(scale=1):
        with gr.Row():
            mask_1 = gr.Image(label="Mask", value=load_img("blank.jpg"), interactive=False)
            img_1 = gr.Image(label="Sample #1", value=load_img("blank.jpg"), interactive=False)
            with gr.Row():
                rank_m_1 = gr.Textbox(label="Mask score (between 0 and 5)")
                rank_c_1 = gr.Textbox(label="Caption score (between 0 and 5)")
        with gr.Row():
            mask_2 = gr.Image(label="Mask", value=load_img("blank.jpg"), interactive=False)
            img_2 = gr.Image(label="Sample #2", value=load_img("blank.jpg"), interactive=False)
            with gr.Row():
                rank_m_2 = gr.Textbox(label="Mask score (between 0 and 5)")
                rank_c_2 = gr.Textbox(label="Caption score (between 0 and 5)")
        with gr.Row():
            mask_3 = gr.Image(label="Mask", value=load_img("blank.jpg"), interactive=False)
            img_3 = gr.Image(label="Sample #3", value=load_img("blank.jpg"), interactive=False)
            with gr.Row():
                rank_m_3 = gr.Textbox(label="Mask score (between 0 and 5)")
                rank_c_3 = gr.Textbox(label="Caption score (between 0 and 5)")
        with gr.Row():
            mask_4 = gr.Image(label="Mask", value=load_img("blank.jpg"), interactive=False)
            img_4 = gr.Image(label="Sample #4", value=load_img("blank.jpg"), interactive=False)
            with gr.Row():
                rank_m_4 = gr.Textbox(label="Mask score (between 0 and 5)")
                rank_c_4 = gr.Textbox(label="Caption score (between 0 and 5)")
    next_btn = gr.Button(value="Next")
    next_btn.click(fn=load_next, inputs=[rank_m_1, rank_m_2, rank_m_3, rank_m_4, rank_c_1, rank_c_2, rank_c_3, rank_c_4, caption, img_1, mask_1, img_2, mask_2, img_3, mask_3, img_4, mask_4, example], outputs=[rank_m_1, rank_m_2, rank_m_3, rank_m_4, rank_c_1, rank_c_2, rank_c_3, rank_c_4, caption, img_1, mask_1, img_2, mask_2, img_3, mask_3, img_4, mask_4, example], queue=False)
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=7872)
