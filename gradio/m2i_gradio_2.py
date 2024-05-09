import gradio as gr
import random
import time
import os
from glob import glob
from PIL import Image
#import torchvision.transforms as transforms

person = "2"
L = 20
H = 40
image_prefix = "m2i_samples/"
mask_prefix = "m2i_samples/"
image_ids = []
img_list = "m2i_rank_order.txt"
with open(img_list) as fp:
    for line in fp:
        image_ids.append(line.strip())
save_path = "m2i_ranks/" + person
image_ids = image_ids[L:H]
num_rank = len(image_ids)

def is_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def load_img(img_path):
    img = Image.open(img_path).convert('RGB')
    return img

def find_completed_idxs(save_path=save_path):
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

def load_next(rank_1, rank_2, rank_3, rank_4, img_1, mask_1, img_2, mask_2, img_3, mask_3, img_4, mask_4, example, ids=image_ids, image_prefix=image_prefix, save_path=save_path):
    file_list, incorrect_files = find_completed_idxs()
    print(str(file_list) + " " + str(incorrect_files))
    if (int(example) not in file_list or int(example) in incorrect_files):
        r = str(image_ids[int(example)])+ "," + rank_1 + "," + rank_2 + "," + rank_3 + "," + rank_4
        r_fp = open(save_path + "/" + str(int(example)) +".txt", "w")
        r_fp.write(r + "\n")
        r_fp.close()
    file_list, incorrect_files = find_completed_idxs()
    if (len(incorrect_files) != 0):
        example = incorrect_files[-1]
    else:
        example = file_list[-1] + 1
    if int(example) == num_rank:
        rank_1 = "DONE!"
        rank_2 = "DONE!"
        rank_3 = "DONE!"
        rank_4 = "DONE!"
        example = -1
        mask_1 = gr.Image(label="Mask", value=load_img("blank.jpg"), interactive=False)
        img_1 = gr.Image(label="Sample #1", value=load_img("blank.jpg"), interactive=False)
        mask_2 = gr.Image(label="Mask", value=load_img("blank.jpg"), interactive=False)
        img_2 = gr.Image(label="Sample #2", value=load_img("blank.jpg"), interactive=False)
        mask_3 = gr.Image(label="Mask", value=load_img("blank.jpg"), interactive=False)
        img_3 = gr.Image(label="Sample #3", value=load_img("blank.jpg"), interactive=False)
        mask_4 = gr.Image(label="Mask", value=load_img("blank.jpg"), interactive=False)
        img_4 = gr.Image(label="Sample #4", value=load_img("blank.jpg"), interactive=False)
    else:
        rank_1 = ""
        rank_2 = ""
        rank_3 = ""
        rank_4 = ""
        img_1 = gr.Image(label="Sample #1", value=load_img(image_prefix + str(image_ids[int(example)]) + "_1.jpg"), interactive=False)
        mask_1 = gr.Image(label="Mask", value=load_img(mask_prefix + str(image_ids[int(example)]) + "_mask_1.jpg"), interactive=False)
        img_2 = gr.Image(label="Sample #2", value=load_img(image_prefix+ str(image_ids[int(example)]) + "_2.jpg"), interactive=False)
        mask_2 = gr.Image(label="Mask", value=load_img(mask_prefix + str(image_ids[int(example)]) + "_mask_2.jpg"), interactive=False)
        img_3 = gr.Image(label="Sample #3", value=load_img(image_prefix + str(image_ids[int(example)]) + "_3.jpg"), interactive=False)
        mask_3 = gr.Image(label="Mask", value=load_img(mask_prefix + str(image_ids[int(example)]) + "_mask_3.jpg"), interactive=False)
        img_4 = gr.Image(label="Sample #4", value=load_img(image_prefix + str(image_ids[int(example)]) + "_4.jpg"), interactive=False)
        mask_4 = gr.Image(label="Mask", value=load_img(mask_prefix + str(image_ids[int(example)]) + "_mask_4.jpg"), interactive=False)
    return [rank_1, rank_2, rank_3, rank_4, img_1, mask_1, img_2, mask_2, img_3, mask_3, img_4, mask_4, example]

with gr.Blocks() as demo:
    last_idx = -1
    example = gr.Number(label="Example #. Click next for #-1 (blank starting page).", value=last_idx, interactive=False)
    with gr.Column(scale=1):
        with gr.Row():
            mask_1 = gr.Image(label="Mask", value=load_img("blank.jpg"), interactive=False)
            img_1 = gr.Image(label="Sample #1", value=load_img("blank.jpg"), interactive=False)
            rank_1 = gr.Textbox(label="Score (between 0 and 5)")
        with gr.Row():
            mask_2 = gr.Image(label="Mask", value=load_img("blank.jpg"), interactive=False)
            img_2 = gr.Image(label="Sample #2", value=load_img("blank.jpg"), interactive=False)
            rank_2 = gr.Textbox(label="Score (between 0 and 5)")
        with gr.Row():
            mask_3 = gr.Image(label="Mask", value=load_img("blank.jpg"), interactive=False)
            img_3 = gr.Image(label="Sample #3", value=load_img("blank.jpg"), interactive=False)
            rank_3 = gr.Textbox(label="Score (between 0 and 5)")
        with gr.Row():
            mask_4 = gr.Image(label="Mask", value=load_img("blank.jpg"), interactive=False)
            img_4 = gr.Image(label="Sample #4", value=load_img("blank.jpg"), interactive=False)
            rank_4 = gr.Textbox(label="Score (between 0 and 5)")
    next_btn = gr.Button(value="Next")
    next_btn.click(fn=load_next, inputs=[rank_1, rank_2, rank_3, rank_4, img_1, mask_1, img_2, mask_2, img_3, mask_3, img_4, mask_4, example], outputs=[rank_1, rank_2, rank_3, rank_4, img_1, mask_1, img_2, mask_2, img_3, mask_3, img_4, mask_4, example], queue=False)
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=7861)
