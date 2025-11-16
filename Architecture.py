import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    def __init__(self, in_channels=1, base_channels=16, latent_dim=1000, act_fn=nn.ReLU()):
        super().__init__()
        self.in_channels = in_channels

        # 32x32
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            act_fn,
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            act_fn
        )

        # Downsample to 16x16, double channels
        self.layer2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, 3, padding=1, stride=2),
            act_fn,
            nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1),
            act_fn
        )

        # Downsample to 8x8, increase channels
        self.layer3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, padding=1, stride=2),
            act_fn,
            nn.Conv2d(base_channels * 4, base_channels * 4, 3, padding=1),
            act_fn
        )

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(base_channels * 4 * 8 * 8, latent_dim)

    def forward(self, x):
        x1 = self.layer1(x)  # 32x32 (no downsampling)
        x2 = self.layer2(x1)  # 16x16 (half resolution)
        x3 = self.layer3(x2)  # 8x8 (quarter resolution)

        latent = self.fc(self.flatten(x3))
        return latent, (x1, x2, x3)  # Return skips for decoder


# Decoder with skip connection inputs
class Decoder(nn.Module):
    def __init__(self, out_channels=1, base_channels=16, latent_dim=1000, act_fn=nn.ReLU()):
        super().__init__()
        self.base_channels = base_channels
        self.latent_dim = latent_dim

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, base_channels * 4 * 8 * 8),
            act_fn
        )

        # 8x8 refinement
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels * 4, 3, padding=1),
            act_fn
        )

        # 1x1 conv after concat with x3 (64+64=128 --> 64)
        self.fix3 = nn.Conv2d(base_channels * 8, base_channels * 4, kernel_size=1)

        # Upsample to 16x16, refine
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 3, stride=2, padding=1, output_padding=1),
            act_fn,
            nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1),
            act_fn
        )

        # 1x1 conv after concat with x2 (32+32=64 --> 32)
        self.fix2 = nn.Conv2d(base_channels * 4, base_channels * 2, kernel_size=1)

        # Final upsampling to 32x32
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, base_channels, 3, stride=2, padding=1, output_padding=1),
            act_fn,
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            act_fn
        )

        # 1x1 conv after concat with x1 (16+16=32 --> 16)
        self.fix1 = nn.Conv2d(base_channels * 2, base_channels, kernel_size=1)

        # Final reconstruction layer
        self.final_conv = nn.Conv2d(base_channels, out_channels, 3, padding=1)

    def forward(self, latent, skips):
        x1, x2, x3 = skips

        out = self.fc(latent)
        out = out.view(-1, self.base_channels * 4, 8, 8)
        out = self.up1(out)

        # Skip connection from encoder layer3 (concat along channels)
        out = torch.cat([out, x3], dim=1)
        out = self.fix3(out)

        out = self.up2(out)

        # Skip 2
        out = torch.cat([out, x2], dim=1)
        out = self.fix2(out)

        out = self.up3(out)

        # Skip 1
        out = torch.cat([out, x1], dim=1)
        out = self.fix1(out)

        out = self.final_conv(out)
        return out


class Autoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        latent, skips = self.encoder(x)
        reconstructed = self.decoder(latent, skips)
        return reconstructed


# Forward pass test
encoder = Encoder(in_channels=1)
decoder = Decoder(out_channels=1)
autoencoder = Autoencoder(encoder, decoder).to(device)

# Test batch of grayscale 32x32 images
test_input = torch.randn(4, 1, 32, 32).to(device)
output = autoencoder(test_input)

print("Input shape:", test_input.shape)
print("Output shape:", output.shape)