from torch import nn
import torch
import torchvision 
from tqdm import tqdm

class TimeEmbedding(nn.Module):
    def __init__(self, timesteps, embedding_dim=256):
        super(TimeEmbedding, self).__init__()
        self.embed = torch.linspace(0, 1, timesteps).repeat(embedding_dim, 1).T

    def forward(self, t):
        self.embed = self.embed.to(t.device)
        x = self.embed[t]
        return x
    
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, timesteps, embedding_dim=256):
        super(SinusoidalTimeEmbedding, self).__init__()
        freq_constant = 10000
        
        t_matrix = torch.arange(timesteps).float().repeat(embedding_dim, 1).T
        p_matrix = torch.arange(embedding_dim).float().repeat(timesteps, 1)
        w = t_matrix / freq_constant**(2*p_matrix/embedding_dim)
        
        w = w.T
        w[::2] = torch.sin(w[::2])
        w[1::2] = torch.cos(w[1::2])
        w = w.T
        
        self.embed = w
        self.model = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        
    def forward(self, t):
        self.embed = self.embed.to(t.device)
        x = self.embed[t]
        x = self.model(x)
        return x


class TimeBlock(nn.Module):
    def __init__(self, t_embedding, out_channels):
        super(TimeBlock, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(t_embedding, out_channels)
        )
        
    def forward(self, x):
        x = self.model(x)
        # to (batch, out_channels, 1, 1)
        x = x.unsqueeze(-1).unsqueeze(-1)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
        )
        
        self.skip = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
            nn.BatchNorm2d(out_channels)
        )
        
    def forward(self, x):
        inv_root2 = 1 #2**(-0.5)
        x = inv_root2*(self.model(x) + self.skip(x) )
        x = torch.nn.functional.relu(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
        )
        
        self.skip = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.Upsample(scale_factor=2),
        )
        
    def forward(self, x):
        inv_root2 = 1 # 2**(-0.5)
        x =  inv_root2*(self.model(x) + self.skip(x) )
        x = torch.nn.functional.relu(x)
        return x

class DiffusionUnet(nn.Module):
    def __init__(self, in_channels, blocks=5, t_embedding_size=256, timesteps=100, initial_channels=4):
        super(DiffusionUnet, self).__init__()
        
        self.preprocess = nn.Sequential(
            nn.Conv2d(in_channels, initial_channels, kernel_size=1),
            nn.BatchNorm2d(initial_channels),
            nn.ReLU()
        )
        
        self.postprocess = nn.Sequential(
            nn.Conv2d(initial_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )
        
        self.t_embedding = SinusoidalTimeEmbedding(timesteps, t_embedding_size)
        self.encoder_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        self.time_blocks_down = nn.ModuleList()
        self.time_blocks_up = nn.ModuleList()
        
        
        in_channels = initial_channels
        out_channels =  in_channels*2
        
        for i in range(blocks):
            self.encoder_blocks.append(EncoderBlock(in_channels, out_channels))
            self.decoder_blocks.append(DecoderBlock(out_channels, in_channels))
            
            # one for the encoder and one for the decoder
            self.time_blocks_down.append(TimeBlock(t_embedding_size, out_channels))
            self.time_blocks_up.append(TimeBlock(t_embedding_size, in_channels))

            in_channels = out_channels
            out_channels *= 2
            
        # reverse the decoder list as it goes from low to high
        self.decoder_blocks = self.decoder_blocks[::-1]
        self.time_blocks_up = self.time_blocks_up[::-1]
        
    def forward(self, x, t):
        
        x = self.preprocess(x)
        
        skip_connections = []
        t = self.t_embedding(t)
        inv_root2 = 1
        
        for encoder_block, time_block in zip(self.encoder_blocks, self.time_blocks_down):
            x = encoder_block(x)
            skip_connections.append(x)
            x = inv_root2 * (x + time_block(t))
            
        
        for decoder_block, time_block in zip(self.decoder_blocks, self.time_blocks_up):
            residual = skip_connections.pop()
            x = inv_root2 * (x + residual)
            x = inv_root2 * (decoder_block(x) + time_block(t))
            
        x = self.postprocess(x)
            
        return x
    
    @torch.no_grad()
    def sample(self, x, scheudler, collect_latents=False):
        ts = torch.arange(0, scheudler.timesteps, device=x.device)
        ts = torch.flip(ts, [0])
        collected_latents = []
        for t in tqdm(ts, leave=False):
            epsilon =  self(x, t)
            

                
            x0 = (x - epsilon * torch.sqrt(scheudler.cumul_beta(t))) / torch.sqrt(scheudler.cumul_alpha(t))
            
            
            t_prev = torch.clamp(t - 1, 0, scheudler.timesteps - 1)
            
            coeff_x0 = torch.sqrt(scheudler.cumul_alpha(t_prev))   * scheudler.beta(t) / scheudler.cumul_beta(t_prev)
            coeff_xt =  torch.sqrt(scheudler.alpha(t)) * scheudler.cumul_beta(t_prev) / scheudler.cumul_beta(t) 

            x_prev = coeff_x0 * x0 + coeff_xt * x

            # epsilon_multiplier =  scheudler.beta(t)/ torch.sqrt(scheudler.cumul_beta(t))
            # x = (x - epsilon * epsilon_multiplier) / torch.sqrt(scheudler.alpha(t))
            
            # we are going to use sigma = beta for the backward pass
            sigma = torch.sqrt(scheudler.beta(t))  * 1
            # x = x + torch.randn_like(x) * sigma
            x = x_prev + torch.randn_like(x) * sigma
            x = torch.clamp(x, -1, 1)
            
            if collect_latents:
                image_grid = torchvision.utils.make_grid(x, nrow=3) 
                collected_latents.append(image_grid)
        
        if collect_latents:
            collected_latents = torch.stack(collected_latents)
            return x, collected_latents
        else:
            return x

            
# m = DiffusionUnet(3,3)

# x = torch.randn(6, 3, 256, 256)
# t = torch.randint(0, 100, (6,))

