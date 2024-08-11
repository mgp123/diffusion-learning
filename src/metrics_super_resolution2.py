import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from ignite.engine import Engine, Events
from ignite.metrics import SSIM, PSNR, MeanSquaredError
import numpy as np
from model import DiffusionUnet
import json
import torchvision
from torchvision import transforms
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
downsample = transforms.Resize(64)
upsample = transforms.Resize(128)
model, noise_schedule = DiffusionUnet.load_from_file('pretrained_models/super.pth')
beta_mult = 0.5
model = model.to(device)

def process_function(engine, entry):
    model.eval()
    batch = entry[0]
    with torch.no_grad():
        batch = batch.to(device)
        x = batch * 2 - 1
        
        low_resolution = downsample(x)
        upsampled = upsample(low_resolution)
        
        z = torch.randn_like(batch)
        output = model.sample(z, noise_schedule, conditioning=upsampled,beta_mult=beta_mult, conditioning_strength=engine.state.c)
        output = (output + 1) / 2
    return output, batch


dataset = torchvision.datasets.ImageFolder(
    root="reference_images_128", 
    transform=transforms.ToTensor()
)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)


evaluator = Engine(process_function)

SSIM_metric = SSIM(data_range=1.0)
PSNR_metric = PSNR(data_range=1.0)
MSE_metric = MeanSquaredError()

MSE_metric.attach(evaluator, 'MSE')
SSIM_metric.attach(evaluator, 'SSIM')
PSNR_metric.attach(evaluator, 'PSNR')


results = []

# read the results from the file if it exists
if os.path.exists("metrics_super_resolution.json"):
    with open("metrics_super_resolution.json", "r") as f:
        results = json.load(f)

#beta_mult_seen = [r["beta_mult"] for r in results]


@evaluator.on(Events.EPOCH_COMPLETED)
def log_metrics(engine):
    metrics = engine.state.metrics
    print(f"conditioning_strength = {engine.state.c}, SSIM: {metrics['SSIM']:.4f}, PSNR: {metrics['PSNR']:.4f}, MSE: {metrics['MSE']:.4f}, beta_mult: {beta_mult}")
    local_results = {
        "SSIM": metrics['SSIM'],
        "PSNR": metrics['PSNR'],
        "MSE": metrics['MSE'],
        "beta_mult": beta_mult,
        "conditioning_strength": engine.state.c
    }
    results.append(local_results)
    
    with open("metrics_super_resolution.json", "w") as f:
        json.dump(results, f) 
    
# Define the range of c values you want to evaluate
c_values = [0.15, 0.5, 0.75, 0.95]

for c in c_values:
    evaluator.state.c = c
    evaluator.run(dataloader)