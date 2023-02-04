import torch
from torch import nn

from edgeconv import EdgeConv

class ParticleNet(nn.Module):
    def __init__(self, features, encoded_space_dim):
        super().__init__()

        # EDGE CONV PART
        self.edge_conv = nn.Sequential(
            EdgeConv(d=features, k=10, C=[64,64,64]),
            EdgeConv(d=64, k=10, C=[128,128,128]),
            EdgeConv(d=128, k=10, C=[256,256,256])) #output shape = [B,n,256]

        self.final_part = nn.Sequential(
            nn.Linear(in_features=256, out_features=256),
            nn.ReLU(),
            nn.Dropout(0.1))

        # LATENT SPACE PROJECTION
        # output size -> dimension of the latent space
        self.latent_space = nn.Sequential(
            nn.Linear(in_features = 256, out_features = encoded_space_dim),
            nn.Softmax(0),
        )
    
    def forward(self, x):
        y = self.edge_conv(x)
        y = nn.AvgPool1d(kernel_size=y.shape[1], stride=1)(y.transpose(1,2)).squeeze()
        y = self.final_part(y)
        y = self.latent_space(y)

        return y
    