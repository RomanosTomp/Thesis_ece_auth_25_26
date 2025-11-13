import math
import torch
import torch.nn as nn


class UNetBlock2x(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.b = nn.Sequential(
            nn.Conv2d(cin,cin,kernel_size=3,stride =1,padding=1, bias =False),
            nn.BatchNorm2d(cin), nn.ReLU(inplace=True),
            nn.Conv2d(cin,cout,kernel_size=3,stride =1,padding =1, bias =False),
            nn.BatchNorm2d(cout),
        )

    def forward(self, x):
        return self.b(x)
    
class DecoderConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.up = nn.ConvTranspose2d(cin, cin, kernel_size=4, stride=2, padding=1, bias=True)
        self.conv = UNetBlock2x(cin, cout)
    
    def forward(self, x):
        x = self.up(x)
        return self.conv(x)
    
class LightDecoder(nn.Module):
    def __init__(self, decoder_fea_dim, upsample_ratio):
        super().__init__()
        self.fea_dim = decoder_fea_dim

        n = round(math.log2(upsample_ratio))
        channels = [self.fea_dim // (2 ** i) for i in range(n)] + [1]
        self.dec = nn.ModuleList([
            DecoderConv(cin, cout,) for (cin, cout) in zip(channels[:-1], channels[1:]
        ])
        self.proj = nn.Conv2d(channels[-1], 3, kernel_size=1, stride=1, bias=True)

    def forward(self, to_dec):
        x = 0 
        for i,d in enumerate(self.dec):
            if i < len(to_dec) and to_dec[i] is not None:
                x = x + to_dec[i]
            x = self.dec[i](x)
        return self.proj(x)