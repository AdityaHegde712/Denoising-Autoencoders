def enblock(in_channels: int, out_channels: int, stride: int = 2,
            act_fn: nn.Module = nn.SiLU(inplace=True)) -> nn.Sequential:
    # Use 8 groups when divisible; otherwise fall back to InstanceNorm-like GroupNorm(1, C)
    g = 8 if out_channels % 8 == 0 else 1
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                  padding=1, padding_mode='reflect', bias=False),
        nn.GroupNorm(g, out_channels),
        act_fn,
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                  padding=1, padding_mode='reflect', bias=False),
        nn.GroupNorm(g, out_channels),
        act_fn,
    )


def deblock(in_channels: int, out_channels: int, stride: int = 2,
            act_fn: nn.Module = nn.SiLU(inplace=True)) -> nn.Sequential:
    # First ConvTranspose2d upsamples; prefer k=4, s=2, p=1 (output_padding=0) to avoid checkerboard
    k1, s1, p1, op1 = (4, 2, 1, 0) if stride == 2 else (3, 1, 1, 0)
    g = 8 if out_channels % 8 == 0 else 1
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=k1, stride=s1,
                           padding=p1, output_padding=op1, bias=False),
        nn.GroupNorm(g, out_channels),
        act_fn,
        nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=1,
                           padding=1, output_padding=0, bias=False),
        nn.GroupNorm(g, out_channels),
        act_fn,
    )


# Define Encoder
class Encoder(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 32, latent_dim: int = LATENT_DIMS, act_fn: nn.Module = nn.SiLU(inplace=True)):
        super().__init__()
        stride = 2
        assert isinstance(act_fn, nn.Module), "act_fn must be a nn.Module"
    
        C1 = out_channels
        C2, C3, C4, C5 = C1*2, C1*4, C1*8, C1*16

        # 6× downsample: 512→256→128→64→32→16→8
        # Break up into blocks so that skips can be collected
        self.net = nn.Sequential(
            self.block1 = enblock(in_channels, C1, stride)  # 512→256
            self.block2 = enblock(C1, C2, stride)           # 256→128
            self.block3 = enblock(C2, C3, stride)           # 128→64
            self.block4 = enblock(C3, C4, stride)           # 64→32
            self.block5 = enblock(C4, C5, stride)           # 32→16
            self.block6 = enblock(C5, C5, stride)           # 16→8
        )
        
        self.latent_channels = latent_dim
        self.to_latent = nn.Conv2d(C5, self.latent_channels, kernel_size=1, bias=False)

    def forward(self, x):
        # Collect skip features
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)
        x5 = self.block5(x4)
        x6 = self.block6(x5)

        z = self.to_latent(x6)
        return z, (x1, x2, x3, x4, x5, x6)


# Define Decoder
class Decoder(nn.Module):
    def __init__(self, in_channels: int = 32, out_channels: int = 3, latent_dim: int = LATENT_DIMS):
        super().__init__()
        stride = 2

        C1 = in_channels
        C2, C3, C4, C5 = C1*2, C1*4, C1*8, C1*16

        self.latent_channels = latent_dim
        self.from_latent = nn.Conv2d(self.latent_channels, C5, kernel_size=1, bias=False)

        # U-net 1x1 corrections after concat
        self.fix6 = nn.Conv2d(C5*2, C5, kernel_size=1, bias=False)
        self.fix5 = nn.Conv2d(C5*2, C4, kernel_size=1, bias=False)
        self.fix4 = nn.Conv2d(C4*2, C3, kernel_size=1, bias=False)
        self.fix3 = nn.Conv2d(C3*2, C2, kernel_size=1, bias=False)
        self.fix2 = nn.Conv2d(C2*2, C1, kernel_size=1, bias=False)
        self.fix1 = nn.Conv2d(C1*2, C1, kernel_size=1, bias=False)

        # Break up net so skips can be inserted
        self.up6 = deblock(C5, C5, stride)  # 8→16
        self.up5 = deblock(C5, C4, stride)  # 16→32
        self.up4 = deblock(C4, C3, stride)  # 32→64
        self.up3 = deblock(C3, C2, stride)  # 64→128
        self.up2 = deblock(C2, C1, stride)  # 128→256
        
        self.head = nn.Sequential(
            nn.ConvTranspose2d(C1, out_channels, kernel_size=4, stride=2, padding=1, bias=False),  # 256→512
            nn.Sigmoid()
        )

    def forward(self, z, skips):
        x1, x2, x3, x4, x5, x6 = skips

        x = self.from_latent(z)

        x = self.up6(x)
        x = torch.cat([x, x6], dim=1)
        x = self.fix6(x)

        x = self.up5(x)
        x = torch.cat([x, x5], dim=1)
        x = self.fix5(x)

        x = self.up4(x)
        x = torch.cat([x, x4], dim=1)
        x = self.fix4(x)

        x = self.up3(x)
        x = torch.cat([x, x3], dim=1)
        x = self.fix3(x)

        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.fix2(x)

        x = torch.cat([x, x1], dim=1)
        x = self.fix1(x)

        return self.head(x)


# Define the autoencoder
class Autoencoder(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        z, skips = self.encoder(x)
        out = self.decoder(z, skips)
        return out
