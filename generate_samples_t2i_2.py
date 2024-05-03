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

prompt_df = pd.read_csv('prompt_2.csv')
prompts = prompt_df['GPT-4 Summary'].tolist()
images = [conditioning_image_transforms(Image.new("RGB", (512, 512))) for i in range(len(prompts))]

# generate images
for ex in range(len(images)):
    print("Starting " + str(ex + 1) + "/" + str(len(images)) + "...") 
    prompt = prompts[ex]
    conditioning_image = images[ex]
    image_id = "mimic_" + str(ex)
    print("Prompt: ", prompt) 
    for i in range(num_samples):
        seed = int(str(ex) + str(i))
        generator = torch.manual_seed(seed)
        image = pipe(prompt, num_inference_steps=75, generator=generator, image=conditioning_image, clip_skip=None).images[0]
        image.save("t2i_samples_2/" + image_id + "_" + str(i + 1) + ".jpg")
        print("Done sample " + str(i + 1) + "/" + str(num_samples) + " with seed " + str(seed) + "...")
    
    print("Done!")
