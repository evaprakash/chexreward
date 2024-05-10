import numpy as np
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler, UNet2DConditionModel
from torchvision import transforms
from diffusers.utils import load_image
import torch
from PIL import Image
import random
from transformers import AutoTokenizer, AutoModelForMaskedLM, PretrainedConfig


base_model_path = "roent-gen-v1-0"
controlnet_path = "saved_model/checkpoint-1000/"

unet = UNet2DConditionModel.from_pretrained("saved_model", subfolder="unet", revision=None, variant=None, torch_dtype=torch.float16)

controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(base_model_path, controlnet=controlnet, torch_dtype=torch.float16, safety_checker=None)

# speed up diffusion process with faster scheduler and memory optimization
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
# memory optimization.
pipe.enable_model_cpu_offload()

control_image = Image.new("RGB", (512, 512))
#control_image = Image.open("train_data/images/m2i/9.7.46.943600.77.0.4.6.04465311658.1763944012815.4_ptx_mask_1.jpg")
conditioning_image_transforms = transforms.Compose([transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR), transforms.CenterCrop(512), transforms.RandomHorizontalFlip(p=1)])
control_image = conditioning_image_transforms(control_image)
prompt="mild pulmonary congestion and cardiomegaly"
#prompt="N/A"
print(np.array(control_image).dtype, print(control_image))
# generate image
for i in range(4):
    seed = int(str(10) + str(i))
    generator = torch.manual_seed(seed)
    image = pipe(prompt, num_inference_steps=200, generator=generator, image=control_image, clip_skip=None).images[0]
    control_image.save("outputs/mask.jpg")
    image.save("outputs/output_" + str(i + 1) + ".png")
