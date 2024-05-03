from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from torchvision import transforms
from diffusers.utils import load_image
import torch
from PIL import Image
import random
from transformers import AutoTokenizer, AutoModelForMaskedLM, PretrainedConfig
import json
import random

base_model_path = "roent-gen-v1-0"
controlnet_path = "saved_model/checkpoint-10000/"

controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(base_model_path, controlnet=controlnet, torch_dtype=torch.float16, safety_checker=None)#, tokenizer=tokenizer, text_encoder=text_encoder, radbert=True)

# speed up diffusion process with faster scheduler and memory optimization
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
# memory optimization.
pipe.enable_model_cpu_offload()

num_samples = 4
prefix = "train_data"
# load data

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

images = filter_elements_with_keywords(json_file_path, keywords_to_search)

random.shuffle(images)

images = images[:45]

# generate images
conditioning_image_transforms = transforms.Compose([transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR), transforms.CenterCrop(512)])
for ex in range(len(images)):
    print("Starting " + str(ex + 1) + "/" + str(len(images)) + "...") 
    conditioning_image_path = images[ex]["conditioning_image"]
    task_id = conditioning_image_path.split("/")[1]
    conditioning_image = Image.open(prefix + "/" + conditioning_image_path)
    image_id = images[ex]["image"].split("/")[1].split(".jpg")[0]
    conditioning_image = conditioning_image_transforms(conditioning_image)
    prompt = images[ex]["text"]
    
    for i in range(num_samples):
        seed = int(str(ex) + str(i))
        generator = torch.manual_seed(seed)
        image = pipe(prompt, num_inference_steps=75, generator=generator, image=conditioning_image, clip_skip=None).images[0]
        image.save("m2i_samples/" + image_id + "_" + task_id + "_" +  str(i + 1) + ".jpg")
        conditioning_image.save("m2i_samples/" + image_id + "_" + task_id + "_mask_" + str(i + 1) + ".jpg")
        print("Done sample " + str(i + 1) + "/" + str(num_samples) + " with seed " + str(seed) + "...")
    
    print("Done!")
