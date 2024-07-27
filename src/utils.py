import torch
from model import DiffusionUnet
from noise_scheudle import CosineSchedule
import torchvision
import os
from torchvision import transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def generate_and_upscale(n):
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
    
    torch.manual_seed(2024)
    
    z = torch.randn((n, 3, image_size, image_size), device=device)
    with torch.autocast(device_type="cuda"):
        sample_low = model.sample(z, noise_schedule, beta_mult=0.34)
        
    del model
    del saved

    torch.cuda.empty_cache()
    
    saved = torch.load("pretrained_models/super.pth")
    model_hyperparameters = saved["model_hyperparameters"]
    image_size = saved["image_size"]
    
    low_resolution_transform = transforms.Resize(image_size)
    sample = low_resolution_transform(sample_low)
    z = torch.randn((n, 3, image_size, image_size), device=device)

    noise_schedule = CosineSchedule(model_hyperparameters["timesteps"], device=device)
    model = DiffusionUnet(
        **model_hyperparameters
    )
    model.load_state_dict(saved["weights"])
    model.to(device)
    model.eval()
    del saved


    with torch.autocast(device_type="cuda"):
        sample_high = model.sample(z, noise_schedule, conditioning=sample,beta_mult=0.8, step_size=5,collect_latents=False)
        
    #torchvision.io.write_video("sample.mp4", images, fps=30)

    # interleave sample_high and sample
    out = [sample[i//2] if i%2==0 else sample_high[i//2] for i in range(len(sample)*2)]
    out = torch.stack(out)
    out = (out + 1) / 2
    output_dir = "generations"
    torchvision.utils.save_image(
        out, 
        f"pepe3.png", 
        nrow=torch.sqrt(torch.tensor(out.shape[0])).int(),
        padding=0
        )
    sample_high = (sample_high + 1) / 2
    torchvision.utils.save_image(
            sample_high, 
            f"pepe3_high.png", 
            nrow=torch.sqrt(torch.tensor(out.shape[0])).int(),
            padding=0
            )
    
    return sample_high

generate_and_upscale(80)