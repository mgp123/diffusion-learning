import torch
from model import DiffusionUnet
from noise_scheudle import LinearSchedule
import torchvision

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
saved = torch.load("weights/model_3.pth")
model_hyperparameters = saved["model_hyperparameters"]
image_size = saved["image_size"]

# load the model from pth
model = DiffusionUnet(
    **model_hyperparameters
)
noise_schedule = LinearSchedule(model_hyperparameters["timesteps"], device=device)
model.load_state_dict(saved["weights"])

model.to(device)
model.eval()

def display_t_embeddings():
    ts = torch.arange(0, noise_schedule.timesteps, device=device)

    t_embeddings = model.t_embedding(ts)
    normalized_t_embeddings = t_embeddings / t_embeddings.norm(dim=1, keepdim=True)

    cos_sim = normalized_t_embeddings @ normalized_t_embeddings.T

    import matplotlib.pyplot as plt
    plt.imshow(cos_sim.cpu().detach())

    plt.show()


display_t_embeddings()


z = torch.randn((9, 3, image_size, image_size), device=device)
sample, images = model.sample(z, noise_schedule, collect_latents=True)

images = images.cpu()
images = (images + 1) / 2
images = torch.clamp(images, 0, 1)
images = images.permute(0, 2, 3, 1)
images = images*255

# save video using torchvision
torchvision.io.write_video("sample.mp4", images, fps=30)

