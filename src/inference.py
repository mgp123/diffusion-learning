import torch
from model import DiffusionUnet
from noise_scheudle import LinearSchedule
import torchvision

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# load the model from pth
model = DiffusionUnet(
    in_channels = 3,
    blocks = 2,
    timesteps = 1000,
    initial_channels = 64
)
noise_schedule = LinearSchedule(1000, device=device)


model.load_state_dict(torch.load("weights/model_0.pth"))
model.to(device)
model.eval()
z = torch.randn((9, 3, 64, 64), device=device)
sample, images = model.sample(z, noise_schedule, collect_latents=True)

images = images.cpu()
images = (images + 1) / 2
images = torch.clamp(images, 0, 1)
images = images.permute(0, 2, 3, 1)
images = images*255

# save video using torchvision
torchvision.io.write_video("sample.mp4", images, fps=30)

