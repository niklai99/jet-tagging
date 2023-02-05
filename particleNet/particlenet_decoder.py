from torch import nn
import torch

from edgeconv import EdgeConv


# In this particular implementation the filters of the ConvTransposed1d
# learn how to extract 1 particle features from the jet latent space 
# representation (specilized over particles)
class ParticleNetDecoder(nn.Module):
    
    def __init__(self, encoded_dim, n, d=16):
        super().__init__()

        self.n = n

        self.latent_to_up = nn.Sequential(
            nn.Softmax(0),
            nn.Linear(encoded_dim, 256),
            nn.Dropout(0.1),
            nn.Linear(256, 256)
        )
     
        self.edgeconvs = nn.Sequential(
            EdgeConv(256, 16, (256, 256, 128)),
            EdgeConv(128, 16, (128, 128, 64)),
            EdgeConv(64, 16, (64, 64, d))
        )

        self.upsampling = nn.ConvTranspose1d(1, self.n, 1, 1)


    def forward(self, x):

        y = self.latent_to_up(x)

        # up(dim)sampling through Conv2DTransposed on unsqueezed array
        y = self.upsampling(y.unsqueeze(1))
        
        y = self.edgeconvs(y)
        
        return y
    


# In this particular implementation the filters of the ConvTransposed1d
# learn how to map one feature to n different particles (specilized over features)
class ParticleNetDecoder_1feat(ParticleNetDecoder):
    def __init__(self, encoded_dim, n, d=16):
        super().__init__(encoded_dim, n, d)

        self.upsampling_feat = nn.ConvTranspose1d(256, 256, 1, 1)

    def forward(self, x):

        y = self.latent_to_up(x)

        # up(dim)sampling through Conv2DTransposed on unsqueezed array
        y = self.upsampling(y.unsqueeze(1))
        y = self.upsampling_feat(y.transpose(1, 2)).transpose(1, 2)

        y = self.edgeconvs(y)
        
        return y