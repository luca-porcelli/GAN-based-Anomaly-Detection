import torch
import torch.nn as nn

# ----- Building blocks -----
def conv_block(in_c, out_c, k=4, s=2, p=1, bn=True):
    layers = [nn.Conv2d(in_c, out_c, k, s, p, bias=not bn)]
    if bn:
        layers += [nn.BatchNorm2d(out_c)]
    layers += [nn.LeakyReLU(0.2, inplace=False)]
    return nn.Sequential(*layers)

def deconv_block(in_c, out_c, k=4, s=2, p=1, bn=True):
    layers = [nn.ConvTranspose2d(in_c, out_c, k, s, p, bias=not bn)]
    if bn:
        layers += [nn.BatchNorm2d(out_c)]
    layers += [nn.ReLU(inplace=False)]
    return nn.Sequential(*layers)

class Encoder(nn.Module):
    def __init__(self, in_c=3, z_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            conv_block(in_c, 64, bn=False),
            conv_block(64, 128),
            conv_block(128, 256),
            conv_block(256, 512),
        )
        self.to_z = nn.Conv2d(512, z_dim, 4, 1, 0)

    def forward(self, x):
        h = self.net(x)
        z = self.to_z(h)
        return z

class Decoder(nn.Module):
    def __init__(self, out_c=3, z_dim=128):
        super().__init__()
        self.from_z = nn.ConvTranspose2d(z_dim, 512, 4, 1, 0)
        self.net = nn.Sequential(
            deconv_block(512, 256),
            deconv_block(256, 128),
            deconv_block(128, 64),
            nn.ConvTranspose2d(64, out_c, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, z):
        h = self.from_z(z)
        x = self.net(h)
        return x

class PatchDiscriminator(nn.Module):
    def __init__(self, in_c=3):
        super().__init__()
        self.net = nn.Sequential(
            conv_block(in_c, 64, bn=False),
            conv_block(64, 128),
            conv_block(128, 256),
            nn.Conv2d(256, 1, 4, 1, 0),
        )

    def forward(self, x):
        return self.net(x)

class GANomaly(nn.Module):
    def __init__(self, in_c=3, z_dim=128, lambda_img=50.0, lambda_latent=1.0):
        super().__init__()
        self.E = Encoder(in_c, z_dim)
        self.DEC = Decoder(in_c, z_dim)
        self.E2 = Encoder(in_c, z_dim)
        self.D = PatchDiscriminator(in_c)
        self.lambda_img = lambda_img
        self.lambda_latent = lambda_latent

    def generator(self, x):
        z = self.E(x)
        x_rec = self.DEC(z)
        z_rec = self.E2(x_rec)
        return x_rec, z, z_rec

    def forward(self, x):
        x_rec, z, z_rec = self.generator(x)
        return x_rec, z, z_rec

    def anomaly_score(self, x, reduction="mean"):
        with torch.no_grad():
            x_rec, z, z_rec = self.generator(x)
            img_err = torch.mean(torch.abs(x - x_rec), dim=[1,2,3])
            lat_err = torch.mean(torch.abs(z - z_rec), dim=[1,2,3])
            score = self.lambda_img * img_err + self.lambda_latent * lat_err
            if reduction == "mean":
                return score.mean().item()
            return score
