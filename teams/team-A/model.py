import torch
import torch.nn as nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Define common blocks to make it easier to read
def enblock(in_channels: int, out_channels: int, stride: int = 2, act_fn: nn.Module = nn.ReLU(inplace=True)) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride),  # Downsample
        nn.BatchNorm2d(out_channels),
        act_fn,
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        act_fn
    )


def deblock(in_channels: int, out_channels: int, stride: int = 2, act_fn: nn.Module = nn.ReLU(inplace=True)) -> nn.Sequential:
    output_padding = stride // 2
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, output_padding=output_padding),
        nn.BatchNorm2d(out_channels),
        act_fn,
        nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        act_fn
    )


# Define Encoder
class Encoder(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 32, latent_dim: int = 1500, act_fn: nn.Module = nn.ReLU()):
        super().__init__()
        activation_fn = act_fn if act_fn else nn.ReLU(inplace=True)
        stride = 2

        C1 = out_channels
        C2, C3, C4, C5 = C1*2, C1*4, C1*8, C1*16

        self.net = nn.Sequential(
            # If my math is right
            enblock(in_channels, C1, stride, activation_fn),  # 512→256
            enblock(C1, C2, stride, activation_fn),  # 256→128
            enblock(C2, C3, stride, activation_fn),  # 128→64
            enblock(C3, C4, stride, activation_fn),  # 64→32
            enblock(C4, C5, stride, activation_fn),  # 32→16
            enblock(C5, C5, stride, activation_fn),  # 16→8
            nn.Flatten(),
            nn.Linear(C5 * 8 * 8, latent_dim),
            activation_fn
        )

    def forward(self, x):
        return self.net(x)


# Define Decoder
class Decoder(nn.Module):
    def __init__(self, in_channels: int = 32, out_channels: int = 3, latent_dim: int = 1500, act_fn: nn.Module = nn.ReLU(inplace=True)):
        super().__init__()
        activation_fn = act_fn if act_fn else nn.ReLU(inplace=True)
        stride = 2

        C1 = in_channels
        C2, C3, C4, C5 = C1*2, C1*4, C1*8, C1*16
        self.latent_feature_maps = C5

        self.linear = nn.Sequential(
            nn.Linear(latent_dim, C5 * 8 * 8),
            act_fn
        )

        self.net = nn.Sequential(
            deblock(C5, C5, stride, activation_fn),  # 8→16
            deblock(C5, C4, stride, activation_fn),  # 16→32
            deblock(C4, C3, stride, activation_fn),  # 32→64
            deblock(C3, C2, stride, activation_fn),  # 64→128
            deblock(C2, C1, stride, activation_fn),  # 128→256

            # Block with a sigmoid to ensure values remain in [0, 1]
            nn.ConvTranspose2d(C1, out_channels, kernel_size=3, padding=1, stride=stride, output_padding=stride//2),  # 256→512
            nn.Sigmoid()  # Output gate to ensure [0, 1] range
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.view(-1, self.latent_feature_maps, 8, 8)
        return self.net(x)


# Define the autoencoder
class Autoencoder(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder):
        super().__init__()
        self.encoder = encoder
        self.encoder.to(DEVICE)

        self.decoder = decoder
        self.decoder.to(DEVICE)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # instantiate your modules (using your exact class signatures/defs)
    enc = Encoder(in_channels=3, out_channels=32, latent_dim=1500).to(device)
    dec = Decoder(in_channels=32, out_channels=3, latent_dim=1500).to(device)
    ae = Autoencoder(enc, dec).to(device)

    # eval-mode + no grad for a shape/range check
    ae.eval()
    with torch.no_grad():
        x = torch.rand(2, 3, 512, 512, device=device)  # batch of 2 RGB images in [0,1]
        y = ae(x)
        print("input shape :", x.shape)
        print("output shape:", y.shape)
        print("output range:", float(y.min()), "→", float(y.max()))
        assert y.shape == (2, 3, 512, 512)
