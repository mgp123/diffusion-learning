from torch import nn
import torch
import torchvision 
from tqdm import tqdm
from noise_scheudle import LinearSchedule

class PositionalEmbedding(nn.Module):
    def __init__(self, timesteps, embedding_dim=256):
        super(PositionalEmbedding, self).__init__()
        self.embed = torch.linspace(0, 1, timesteps).repeat(embedding_dim, 1).T

    def forward(self, t):
        self.embed = self.embed.to(t.device)
        x = self.embed[t]
        return x

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, timesteps, embedding_dim=256):
        super(SinusoidalPositionalEmbedding, self).__init__()
        freq_constant = 10000
        
        t_matrix = torch.arange(timesteps).half().repeat(embedding_dim, 1).T
        p_matrix = torch.arange(embedding_dim).half().repeat(timesteps, 1)
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


class VarianceEmbedding(nn.Module):
    def __init__(self, timesteps, embedding_dim=256, noise_schedule=LinearSchedule):
        super(VarianceEmbedding, self).__init__()
        self.cumul_alpha = noise_schedule(timesteps)._cumul_alphas
        self.model = nn.Sequential(
            nn.Linear(1, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, t):
        self.cumul_alpha = self.cumul_alpha.to(t.device)
        x = self.cumul_alpha[t].unsqueeze(-1)  
        x = self.model(x)
        return x


class TimeBlock(nn.Module):
    def __init__(self, t_embedding, out_channels):
        super(TimeBlock, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(t_embedding, out_channels*2),
        )
        
    def forward(self, x, t):
        mult = self.model(t)
        # to (batch, out_channels, 1, 1)
        mult = mult.unsqueeze(-1).unsqueeze(-1)
        scale, bias = mult.chunk(2, dim=1)
        x = nn.functional.group_norm(x, 2)
        x = x * scale + bias
        return x

class VarianceBlock(nn.Module):
    def __init__(self, out_channels):
        super(VarianceBlock, self).__init__()
        # this is basically a vector, but whatever
        self.scaler =  nn.Linear(1, out_channels)
        self.biaser = nn.Linear(1, out_channels)
        
    def forward(self, x, v):
        scale, bias = self.scaler(v), self.biaser(v)
        scale, bias = scale.unsqueeze(-1).unsqueeze(-1), bias.unsqueeze(-1).unsqueeze(-1)
        x = nn.functional.group_norm(x, 2)
        x = x * scale + bias
        return x


class SelfAttentionBlock(nn.Module):
    def __init__(self, in_channels, in_shape, latent_size) -> None:
        super(SelfAttentionBlock, self).__init__()
        
        self.query = nn.Conv2d(in_channels, latent_size, kernel_size=1)
        self.key = nn.Conv2d(in_channels, latent_size, kernel_size=1)
        self.value = nn.Conv2d(in_channels, latent_size, kernel_size=1)
        self.out = nn.Conv2d(latent_size, in_channels, kernel_size=1)
        
        self.in_shape = in_shape
        full_size = torch.prod(torch.tensor(in_shape)).item()
        self.positional_embedding = SinusoidalPositionalEmbedding(full_size, latent_size)
        self.latent_size = latent_size
        
    
    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        batch, channels, height, width = q.shape

        pos_embedding = self.positional_embedding(torch.arange(0, height*width, device=x.device))
        pos_embedding = pos_embedding.reshape(1, self.latent_size, height, width)
        pos_embedding = pos_embedding.repeat(batch, 1, 1, 1)
        
        q = q + pos_embedding
        k = k + pos_embedding
        v = v + pos_embedding
        
        # flatten to vector
        q = q.view(batch, channels, -1)
        k = k.view(batch, channels, -1)
        v = v.view(batch, channels, -1)
        
        q = q.permute(0, 2, 1)
        
        attention = torch.bmm(q, k)/torch.sqrt(torch.tensor(channels, device=x.device))
        attention = torch.nn.functional.softmax(attention, dim=1 )
              
        res = torch.bmm(v, attention)
        res = res.view(batch, channels, height, width)
        res = self.out(res)
        
        return res
        

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=2),
            nn.GroupNorm(2, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(2, out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(2, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(2, out_channels),
        )

        
    def forward(self, x):
        return self.model(x)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.model = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(2, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(2, out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(2, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(2, out_channels),        
        )
        
        
    def forward(self, x):
        return self.model(x)

class MidBlockUnit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MidBlockUnit, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(2, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(2, out_channels),
        )
        
    def forward(self, x):
        return self.model(x)

class DiffusionUnet(nn.Module):
    def __init__(self, in_channels, blocks=5, t_embedding_size=256, timesteps=100, initial_channels=4, channel_multiplier=None):
        super(DiffusionUnet, self).__init__()
        
        self.preprocess = nn.Sequential(
            nn.Conv2d(in_channels, initial_channels, kernel_size=1),
            nn.GroupNorm(2, initial_channels),
            nn.GELU()
        )
        
        self.postprocess = nn.Sequential(
            nn.Conv2d(initial_channels, in_channels, kernel_size=1),
        )
        
        # self.t_embedding = SinusoidalPositionalEmbedding(timesteps, t_embedding_size)
        self.encoder_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        self.time_blocks_down = nn.ModuleList()
        self.time_blocks_up = nn.ModuleList()
        
        if channel_multiplier is None:
            channel_multiplier = 2
        
        if type(channel_multiplier) == int:
            channel_multiplier = [channel_multiplier] * blocks
        
        
        in_channels = initial_channels
        out_channels =  in_channels * channel_multiplier[0]
        
        for i in range(blocks):
            self.encoder_blocks.append(EncoderBlock(in_channels, out_channels))
            self.decoder_blocks.append(DecoderBlock(out_channels*2, in_channels)) # times 2 because of the skip connection cat
            
            # one for the encoder and one for the decoder
            self.time_blocks_down.append(VarianceBlock(out_channels))
            self.time_blocks_up.append(VarianceBlock(in_channels))

            in_channels = out_channels
            out_channels *= channel_multiplier[i+1] if i+1 < len(channel_multiplier) else 1
        
        
        self.middle_block = nn.Sequential(
            SelfAttentionBlock(in_channels, (16, 16), 128),
            MidBlockUnit(in_channels, in_channels),
            MidBlockUnit(in_channels, in_channels),
            MidBlockUnit(in_channels, in_channels),
        )

        
        # reverse the decoder list as it goes from low to high
        self.decoder_blocks = self.decoder_blocks[::-1]
        self.time_blocks_up = self.time_blocks_up[::-1]
        
    def forward(self, x, v):
        
        x = self.preprocess(x)
        
        skip_connections = []
        inv_root2 = (2**(-0.5))
        
        for encoder_block, time_block in zip(self.encoder_blocks, self.time_blocks_down):
            x = encoder_block(x)
            x = time_block(x, v)
            skip_connections.append(x)
            
        
        x = self.middle_block(x) + x
        
        for decoder_block, time_block in zip(self.decoder_blocks, self.time_blocks_up):
            residual = skip_connections.pop()
            x = decoder_block(torch.cat([x, residual], dim=1))
            x = time_block(x, v)
            
        x = self.postprocess(x)
            
        return x
    
    @torch.no_grad()
    def sample(self, x, scheudler, collect_latents=False):
        ts = torch.arange(0, scheudler.timesteps, device=x.device)
        step_size = 5
        ts = torch.flip(ts, [0])[:-1][::step_size]
        collected_latents = []
        for t in tqdm(ts, leave=False):
            t_vect = t.repeat(x.shape[0])
            v = torch.sqrt(scheudler.cumul_beta(t_vect)).unsqueeze(-1)
            epsilon =  self(x, v)
            
            x_prev = (x - epsilon * scheudler.beta(t) *  torch.sqrt(1/ scheudler.cumul_beta(t))) / torch.sqrt(scheudler.alpha(t))
            x0 =  (x - epsilon  *  torch.sqrt(scheudler.cumul_beta(t))) / torch.sqrt(scheudler.cumul_alpha(t))
            t_prev = torch.clamp(t - step_size, 0, scheudler.timesteps - 1)
            beta_cum_prev = scheudler.cumul_beta(t_prev)
            alpha_cum_prev = scheudler.cumul_alpha(t_prev)
            alpha_cum = scheudler.cumul_alpha(t)
            
            x0 = torch.clamp(x0, -1, 1)

            # we are going to use sigma = beta for the backward pass
            sigma =   torch.sqrt(beta_cum_prev * 0.5) 
            noise = torch.sqrt(beta_cum_prev - sigma**2) * epsilon + sigma * torch.randn_like(x) 
            
            x0_multiplier = torch.sqrt(alpha_cum_prev)
            noise_multiplier = 1
            x = x0 * x0_multiplier + noise * noise_multiplier
            clip_power = 2 # 1 + torch.sqrt(beta_cum_prev)
            
            
            # x = torch.clamp(x, -clip_power, clip_power)

            
      
            
            if collect_latents:
                image_grid = torchvision.utils.make_grid(x0, nrow=torch.sqrt(torch.tensor(x0.shape[0])).int()) 
                collected_latents.append(image_grid)
        
        if collect_latents:
            collected_latents = torch.stack(collected_latents)
            return x, collected_latents
        else:
            return x

            
# m = DiffusionUnet(3,3)

# x = torch.randn(6, 3, 256, 256)
# t = torch.randint(0, 100, (6,))

