import torch
from model import DiffusionUnet
from noise_scheudle import CosineSchedule
from torchprofile import profile_macs


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
saved = torch.load("pretrained_models/medminst.pth")
model_hyperparameters = saved["model_hyperparameters"]
image_size = saved["image_size"]

model = DiffusionUnet(
    **model_hyperparameters
)
noise_schedule = CosineSchedule(model_hyperparameters["timesteps"], device=device)
model.load_state_dict(saved["weights"])

model.to(device)
model.eval()
t = torch.randint(0, 100, (1,), device=device)
cumul_alpha = noise_schedule.cumul_alpha(t).to(device)

dummy_input = torch.randn(9, 3, image_size, image_size).to(device)
flops = profile_macs(model, args=(dummy_input, cumul_alpha))

print(f"Approximate flops for a single forward pass: {flops}")
