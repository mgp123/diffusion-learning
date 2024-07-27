import torch
from model import DiffusionUnet
from noise_scheudle import CosineSchedule
import torchvision
import os
from torchvision import transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
saved = torch.load("weights/model_6.pth")
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

z = torch.randn((5**2, 3, image_size, image_size), device=device) * 1
with torch.autocast(device_type="cuda"):

    sample = model.sample(z, noise_schedule, collect_latents=collect_latents, beta_mult=0.27)

# images = images.cpu()
# images = (images + 1) / 2
# images = torch.clamp(images, 0, 1)
# images = images.permute(0, 2, 3, 1)
# images = images*255

sample = (sample + 1) / 2
output_dir = "generations"
if save_individual:
    for i, img in enumerate(sample):
        torchvision.utils.save_image(img, os.path.join(output_dir, f"image_{i:03d}.png"))
    print(f"Saved {len(sample)} images in {output_dir}")



if save_grid:
    torchvision.utils.save_image(
        sample, 
        f"pepe2.png", 
        nrow=torch.sqrt(torch.tensor(sample.shape[0])).int(),
        padding=0
        )


# save video using torchvision
# torchvision.io.write_video("sample.mp4", images, fps=30)

