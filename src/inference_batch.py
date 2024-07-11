import torch
from tqdm import tqdm
from model import DiffusionUnet
from noise_scheudle import CosineSchedule
import torchvision
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
saved = torch.load("weights/model_313.pth")
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
model = model.half()

collect_latents = False


n_total = 1000
iteration_size = 20**2

for beta_mult in tqdm([x / 10.0 + 0.05 for x in range(0, 11)]):
    output_dir = f"generations/beta_{beta_mult}"
    print(output_dir)
    os.makedirs(output_dir,exist_ok=True)

    for it in range(0,n_total, iteration_size):
        z = torch.randn((20**2, 3, image_size, image_size), device=device) * 1
        z = z.half()
        
        with torch.autocast(device_type="cuda"):
            sample = model.sample(z, noise_schedule, collect_latents=collect_latents,beta_mult=beta_mult)

        sample = (sample + 1) / 2
        for i, img in enumerate(sample):
            image_id = i + it
            torchvision.utils.save_image(img, os.path.join(output_dir, f"image_{image_id:04d}.png"))