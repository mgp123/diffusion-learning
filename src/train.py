import torch
import torchvision
from torchvision import transforms
from torch.utils import tensorboard

from tqdm import tqdm
from model import DiffusionUnet
from noise_scheudle import CosineSchedule, LinearSchedule

image_size =  64
dataset =torchvision.datasets.ImageFolder(
    root='dataset', 
    transform= transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(image_size),
        transforms.RandomCrop(image_size)
    ])
    )

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# training hyperparameters 
lr=2e-4
sample_every = 500
epochs = 40
batch_size = 0 + 64
iters_to_accumulate = 4

# model hyperparameters
model_hyperparameters= {
    "in_channels" : 3,
    "blocks" : 3,
    "timesteps" : 1000,
    "initial_channels" : 128,
    "channel_multiplier" : 2,
    }
timesteps = model_hyperparameters["timesteps"]
in_channels = model_hyperparameters["in_channels"]


summary_writer = tensorboard.SummaryWriter()
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
scaler = torch.cuda.amp.GradScaler()


# model and optimizer
noise_schedule = CosineSchedule(timesteps, device=device)
model = DiffusionUnet(
    **model_hyperparameters
    ).to(device)


if False:
    saved = torch.load("weights/model_214.pth")
    model_hyperparameters = saved["model_hyperparameters"]
    image_size = saved["image_size"]

    # load the model from pth
    model = DiffusionUnet(
        **model_hyperparameters
    )
    noise_schedule = CosineSchedule(model_hyperparameters["timesteps"], device=device)
    model.load_state_dict(saved["weights"])
    model.to(device)


optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# training loop
steps = 0
for epoch in range(300,epochs+300):
    for x,_ in tqdm(dataloader):
        # optimizer stuff
        x = x.to(device)
        
        # normalization
        x = x * 2 - 1
        
        # our three random variables, x_0, t, and epsilon
        t = torch.randint(0, timesteps, (x.shape[0],), device=device)
        epsilon = torch.randn_like(x)
        
        
        # getting the noise variables and reshaping it for the 2d image case
        target_shape = [x.shape[0]] + [1]*len(x.shape[1:])
        cumul_alpha = noise_schedule.cumul_alpha(t).to(device).view(*target_shape)
        cumul_beta = noise_schedule.cumul_beta(t).to(device).view(*target_shape) 
        
        with torch.autocast(device_type="cuda"):

            x_t = torch.sqrt(cumul_alpha)* x + torch.sqrt(cumul_beta) * epsilon
        
            # predicted_epsilon = model(x_t, t) 
            v = torch.sqrt(cumul_beta)[:,:,0,0]
            predicted_epsilon = model(x_t, v )
            
            # error prediction and backprop
            loss = torch.nn.functional.mse_loss(predicted_epsilon, epsilon)
            
        scaler.scale(loss).backward()
        steps += 1
        if (steps) % iters_to_accumulate == 0:
            # may unscale_ here if desired (e.g., to allow clipping unscaled gradients)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            summary_writer.add_scalar('loss', loss.item(), steps//iters_to_accumulate)
        
        
        # utility stuff
    
        if steps % sample_every == 0:
            with torch.no_grad():
                model.eval()
                with torch.autocast(device_type="cuda"):
                    z = torch.randn((9, in_channels, image_size, image_size), device=device)
                    sample = model.sample(z, noise_schedule)
                    sample = torch.clamp(sample, -1, 1)
                    # denormalize
                    sample = (sample + 1) / 2
                    
                torchvision.utils.save_image(sample, f"img/generations/sample_{steps}.png", nrow=3)
                model.train()
    
    # save the model
    torch.save(
        {
            "weights":model.state_dict(),
            "model_hyperparameters":model_hyperparameters,
            "image_size":image_size,
        },
        f"weights/model_{epoch}.pth")    
    
summary_writer.flush()
    