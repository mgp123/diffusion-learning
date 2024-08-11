import numpy as np
import torch
from model import DiffusionUnet
from noise_scheudle import CosineSchedule
import torchvision
import os
from torchvision import transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
saved = torch.load("pretrained_models/model_6.pth")
model_hyperparameters = saved["model_hyperparameters"]
image_size = saved["image_size"]

model = DiffusionUnet(
    **model_hyperparameters
)
noise_schedule = CosineSchedule(model_hyperparameters["timesteps"], device=device)
model.load_state_dict(saved["weights"])

model.to(device)
model.eval()

collect_latents = False
save_grid = True
save_individual = False

block = 3
all_modules = [(f"encoder_{i}" , f"decoder_{i}", f"residual_{i}") for i in range(block)]
all_modules = [item for sublist in all_modules for item in sublist]
all_modules = [None] + all_modules
all_results = {}

beta_muls = [0.15, 0.25,0.35, 0.45, 0.5]

for beta_mul in beta_muls:

    torch.manual_seed(2008098024)

    z = torch.randn((4, 3, image_size, image_size), device=device) * 1
    zeroing_modules = set()

    with torch.autocast(device_type="cuda"):

        sample = model.sample(z, noise_schedule, collect_latents=collect_latents, beta_mult=beta_mul, zeroing_modules=zeroing_modules)

    sample = (sample + 1) / 2
    
    all_results[beta_mul] = sample




import matplotlib.pyplot as plt

images = [all_results[beta].cpu().detach().numpy() for beta in beta_muls]
images = [np.concatenate(image, axis=1) for image in images]
images = np.concatenate(images, axis=2)
images = images.transpose((1, 2, 0))
plt.imshow(images)
plt.axis('off')
plt.savefig('output.png', bbox_inches='tight', pad_inches=0, dpi=400)
plt.show()
