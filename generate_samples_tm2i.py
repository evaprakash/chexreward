from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from torchvision import transforms
from diffusers.utils import load_image
import torch
from PIL import Image
import random
from transformers import AutoTokenizer, AutoModelForMaskedLM, PretrainedConfig
import json
import random
import pandas as pd
import csv

base_model_path = "roent-gen-v1-0"
controlnet_path = "saved_model/checkpoint-10000/"

controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(base_model_path, controlnet=controlnet, torch_dtype=torch.float16, safety_checker=None)#, tokenizer=tokenizer, text_encoder=text_encoder, radbert=True)

# speed up diffusion process with faster scheduler and memory optimization
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
# memory optimization.
pipe.enable_model_cpu_offload()

num_samples = 4
conditioning_image_transforms = transforms.Compose([transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR), transforms.CenterCrop(512)])

prompt_df = pd.read_csv('prompt_final_filtered.csv')
dicom_df = pd.read_csv('tm2i.csv')

tm2i_dicom_ids = dicom_df['dicom_id'].tolist()
tm2i_study_ids = dicom_df['study_id'].tolist()
t2i_study_ids = prompt_df['Study ID'].tolist()
t2i_prompts = prompt_df['GPT-4 Summary'].tolist()

matched_idxs = [t2i_study_ids.index(x) for x in tm2i_study_ids if x in t2i_study_ids]
matched_prompts = [t2i_prompts[i] for i in matched_idxs]
prompt_dicom_pairs = [(matched_prompts[i], tm2i_dicom_ids[tm2i_study_ids.index(t2i_study_ids[idx])]) for i, idx in enumerate(matched_idxs)]
prompt_dicom_pairs = list(set(prompt_dicom_pairs))

with open('prompt_dicom_pairs_final.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['prompt', 'dicom_id'])  # Writing header
    writer.writerows(prompt_dicom_pairs) 

# generate images
for ex in range(len(prompt_dicom_pairs)):
    print("Starting " + str(ex + 1) + "/" + str(len(prompt_dicom_pairs)) + "...") 
    prompt = prompt_dicom_pairs[ex][0]
    image_id_1 = prompt_dicom_pairs[ex][1] + '_heart'
    image_id_2 = prompt_dicom_pairs[ex][1] + '_lung'
    conditioning_image_1 = conditioning_image_transforms(Image.open('mimic_masks/' + image_id_1 + '_mask.jpg'))
    conditioning_image_2 = conditioning_image_transforms(Image.open('mimic_masks/' + image_id_2 + '_mask.jpg'))
    print("Prompt: ", prompt) 
    for info in [(image_id_1, conditioning_image_1), (image_id_2, conditioning_image_2)]:
        image_id = info[0]
        conditioning_image = info[1]
        for i in range(num_samples):
            seed = int(str(ex) + str(i))
            generator = torch.manual_seed(seed)
            image = pipe(prompt, num_inference_steps=75, generator=generator, image=conditioning_image, clip_skip=None).images[0]
            image.save("tm2i_samples_final/" + image_id + "_" + str(i + 1) + ".jpg")
            conditioning_image.save("tm2i_samples_final/" + image_id + "_mask_" + str(i + 1) + ".jpg")
            print("Done sample " + str(i + 1) + "/" + str(num_samples) + " with seed " + str(seed) + "...")
    print("Done!")
