import torch
from model import DiffusionUnet
from noise_scheudle import CosineSchedule
import torchvision
import os
from torchvision import transforms
from torchsummary import summary

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
saved = torch.load("pretrained_models/medminst.pth")
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


collect_latents = False
save_grid = True
save_individual = False

# seed
torch.manual_seed(2008098024)
#model = model.half()

z = torch.randn((9, 3, image_size, image_size), device=device) * 1
zeroing_modules = set()


sample = model.sample(z, noise_schedule, collect_latents=collect_latents, beta_mult=0.45, zeroing_modules=zeroing_modules)

sample = (sample + 1) / 2
output_dir = "generations"
if save_individual:
    for i, img in enumerate(sample):
        torchvision.utils.save_image(img, os.path.join(output_dir, f"image_{i:03d}.png"))
    print(f"Saved {len(sample)} images in {output_dir}")



if save_grid:
    torchvision.utils.save_image(
        sample, 
        f"abc.png", 
        nrow=torch.sqrt(torch.tensor(sample.shape[0])).int(),
        padding=0
        )


# save video using torchvision
# torchvision.io.write_video("sample.mp4", images, fps=30)

