import torch
import torchvision
from torchvision import transforms
from torch.utils import tensorboard

from tqdm import tqdm
from model import DiffusionUnet
from noise_scheudle import LinearSchedule

dataset =torchvision.datasets.ImageFolder(
    root='dataset', 
    transform= transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
        transforms.Resize(128),
        transforms.RandomCrop(128)
    ])
    )

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# training hyperparameters and other stuff like that
channels = 3
timesteps = 1000
lr=1e-3
sample_every = 500
epochs = 10
batch_size=64
unet_blocks = 2

summary_writer = tensorboard.SummaryWriter()
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# model and optimizer
noise_schedule = LinearSchedule(timesteps, device=device)
model = DiffusionUnet(channels,blocks=unet_blocks, timesteps=timesteps).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# training loop
steps = 0
for epoch in range(epochs):
    for x,_ in tqdm(dataloader):
        # optimizer stuff
        optimizer.zero_grad()
        x = x.to(device)
        
        # our three random variables, X_0, t, and epsilon
        t = torch.randint(0, timesteps, (x.shape[0],), device=device)
        epsilon = torch.randn_like(x)
        
        
        # getting the noise variables and reshaping it for the 2d image case
        target_shape = [x.shape[0]] + [1]*len(x.shape[1:])
        cumul_alpha = noise_schedule.cumul_alpha(t).to(device).view(*target_shape)
        cumul_beta = noise_schedule.cumul_beta(t).to(device).view(*target_shape) 
        
        
        x_t = torch.sqrt(cumul_alpha) * x + torch.sqrt(cumul_beta) * epsilon
        
        # torchvision.utils.save_image(x_t, f"test_image_{steps}.png")

        predicted_epsilon = model(x_t, t)
        
        # error prediction and backprop
        loss = torch.nn.functional.mse_loss(predicted_epsilon, epsilon)
        loss.backward()
        optimizer.step()
        
        # utility stuff
        steps += 1
        summary_writer.add_scalar('loss', loss.item(), steps)
        
        if steps % sample_every == 0:
            with torch.no_grad():
                model.eval()
                z = torch.randn(9, channels, 128, 128, device=device)
                sample = model.sample(z, noise_schedule)
                torchvision.utils.save_image(sample, f"img/generations/sample_{steps}.png", nrow=3)
                model.train()
        
        
summary_writer.flush()
    