from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from torchvision import transforms
from diffusers.utils import load_image
import torch
from PIL import Image
import random
from transformers import AutoTokenizer, AutoModelForMaskedLM, PretrainedConfig


base_model_path = "roent-gen-v1-0"
controlnet_path = "saved_model/checkpoint-10000/"

tokenizer = AutoTokenizer.from_pretrained("StanfordAIMI/RadBERT", revision=None, use_fast=False)
text_encoder = AutoModelForMaskedLM.from_pretrained("StanfordAIMI/RadBERT")

controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(base_model_path, controlnet=controlnet, torch_dtype=torch.float16, safety_checker=None)#, tokenizer=tokenizer, text_encoder=text_encoder, radbert=True)

# speed up diffusion process with faster scheduler and memory optimization
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
# memory optimization.
pipe.enable_model_cpu_offload()

#control_image = Image.new("RGB", (512, 512))
control_image = Image.open("val_masks/ptx_mask.jpg")
conditioning_image_transforms = transforms.Compose([transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR), transforms.CenterCrop(512)])
control_image = conditioning_image_transforms(control_image)
prompt="N/A"
seed=7
# generate image
for i in range(5):
    generator = torch.manual_seed(seed)
    image = pipe(prompt, num_inference_steps=200, generator=generator, image=control_image, clip_skip=None).images[0]
    image.save("outputs/output_" + str(i + 1) + ".png")
