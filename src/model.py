import torch


class EncoderBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=2),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU()
        )
        
    def forward(self, x):
        return self.model(x)

class DecoderBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.Upsample(scale_factor=2),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
        )
        
    def forward(self, x):
        return self.model(x)

class DiffusionUnet(torch.nn.Module):
    def __init__(self, in_channels, blocks=5):
        super(DiffusionUnet, self).__init__()
        
        
        self.encoder_blocks = torch.nn.ModuleList()
        self.decoder_blocks = torch.nn.ModuleList()
        out_channels =  in_channels*2
        
        for i in range(blocks):
            self.encoder_blocks.append(EncoderBlock(in_channels, out_channels))
            self.decoder_blocks.append(DecoderBlock(out_channels, in_channels))
            
            in_channels = out_channels
            out_channels *= 2
            
        # reverse the decoder list
        self.decoder_blocks = self.decoder_blocks[::-1]
        
    def forward(self, x):
        skip_connections = []
        
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)
            skip_connections.append(x)
            
        
        for decoder_block in self.decoder_blocks:
            residual = skip_connections.pop()
            x = (2**(-0.5)) * (x + residual)
            x = decoder_block(x)
            
        return x
            
            
            
# m = DiffusionUnet(3)

# x = torch.randn(1, 3, 256, 256)

# print(m(x).shape)