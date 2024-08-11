import numpy as np
import torch
from tqdm import tqdm
from model import DiffusionUnet
from noise_scheudle import CosineSchedule
import torchvision
import os
import gc

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
saved = torch.load("pretrained_models/model_6.pth")
model_hyperparameters = saved["model_hyperparameters"]
image_size = saved["image_size"]

# load the model from pth
model = DiffusionUnet(
    **model_hyperparameters
)
noise_schedule = CosineSchedule(model_hyperparameters["timesteps"], device=device)
model.load_state_dict(saved["weights"])

model.to(device)
model.eval()


del saved
gc.collect()
torch.cuda.empty_cache()

collect_latents = False


n_total = 1000
iteration_size = 20**2

for beta_mult in tqdm(np.linspace(0.0, 1.0, 20)):
    output_dir = f"generations/beta_{beta_mult}"
    print(output_dir)
    os.makedirs(output_dir,exist_ok=True)

    for it in range(0,n_total, iteration_size):
        z = torch.randn((iteration_size, 3, image_size, image_size), device=device) * 1
        z = z.half()
        
        with torch.autocast(device_type="cuda"):
            sample = model.sample(z, noise_schedule, collect_latents=collect_latents,beta_mult=beta_mult)

        sample = (sample + 1) / 2
        for i, img in enumerate(sample):
            image_id = i + it
            torchvision.utils.save_image(img, os.path.join(output_dir, f"image_{image_id:04d}.png"))
