import torch
from model import DiffusionUnet
from noise_scheudle import CosineSchedule
import torchvision
import os
from torchvision import transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def generate_and_upscale(n):
    saved = torch.load("pretrained_models/generation_model.pth")
    model_hyperparameters = saved["model_hyperparameters"]
    image_size = saved["image_size"]
    
    model = DiffusionUnet(
    **model_hyperparameters
)
    noise_schedule = CosineSchedule(model_hyperparameters["timesteps"], device=device)
    model.load_state_dict(saved["weights"])

    model.to(device)
    model.eval()
    
    
    z = torch.randn((n, 3, image_size, image_size), device=device)
    z = z.half()
    with torch.autocast(device_type="cuda"):
        sample_low = model.sample(z, noise_schedule, beta_mult=0.55)
        
    del model
    del saved

    torch.cuda.empty_cache()
    
    saved = torch.load("weights_super_resolution/model_3.pth")
    model_hyperparameters = saved["model_hyperparameters"]
    image_size = saved["image_size"]
    
    low_resolution_transform = transforms.Resize(image_size)
    sample = low_resolution_transform(sample_low)
    z = torch.randn((n, 3, image_size, image_size), device=device)
    z = z.half()

    noise_schedule = CosineSchedule(model_hyperparameters["timesteps"], device=device)
    model = DiffusionUnet(
        **model_hyperparameters
    )
    model.load_state_dict(saved["weights"])
    model.to(device)
    model.eval()


    with torch.autocast(device_type="cuda"):
        sample_high, images = model.sample(z, noise_schedule, conditioning=sample,beta_mult=0.3, step_size=10,collect_latents=True)
        
    
    images = images.cpu()
    images = (images + 1) / 2
    images = torch.clamp(images, 0, 1)
    images = images.permute(0, 2, 3, 1)
    images = images*255
    torchvision.io.write_video("sample.mp4", images, fps=30)

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


generate_and_upscale(18)