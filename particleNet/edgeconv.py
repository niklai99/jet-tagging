import torch
from torch import nn


########## Edge Convolution Block ###########
# The root block of our DNN.
# Initialized by:
#   - d     the number of features
#   - k     number of nearest neighbours to consider in the concolution
#   - C     a list-like or an int with the number of neurons of the three linear layers
#   - aggr  the aggregation function, must be symmetric

class EdgeConv(nn.Module):
    
    def __init__(self, d, k, C, aggr=None):
        super().__init__()
        
        if type(C) == int:
            self.C = [C]*3
        else:
            self.C = C
        
        self.k = k

        if aggr is None:
            self.aggr = torch.mean
        else:
            self.aggr = aggr

        self.act = nn.ReLU()


        ### Shortcut path
        self.shortcut = nn.Sequential(
            nn.Conv1d(in_channels = d, out_channels = self.C[-1], kernel_size = 1, stride = 1),
            nn.BatchNorm1d(self.C[-1])
        )

        ### Linear section, approximation of a MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(2*d, self.C[0], 1, 1),
            nn.BatchNorm2d(self.C[0]),
            nn.ReLU(),
            nn.Conv2d(self.C[0], self.C[1], 1, 1),
            nn.BatchNorm2d(self.C[1]),
            nn.ReLU(),
            nn.Conv2d(self.C[1], self.C[2], 1, 1),
            nn.BatchNorm2d(self.C[2]),
            nn.ReLU()
        )


    def kNN(self, x):
        """input: single jet data
            output: tensor with shape [B, n, k, d] where d are the features of the knn particles"""
        # expand the input tensor s.t. x_knn.shape = [B, n, n, d]
        x_knn = x.unsqueeze(1).expand(-1, x.shape[1], -1, -1)

        # calculate both delta_phi and delta_eta
        delta_phieta = x_knn[:, :, :, :2] - x_knn[:, :, :, :2].transpose(1, 2)

        # calculate distances and sort them in ascending order, keep only the indeces
        _, indeces = torch.sqrt(torch.sum(delta_phieta**2, 3)**0.5).sort(dim=2, stable=True)

        # keep the indeces of k nearest neighbours and use them to sort and cut the initial tensor
        knn = indeces[:,:,:self.k]
        x_knn = torch.gather(x_knn, 2, knn.unsqueeze(-1).expand(-1, -1, -1, x_knn.shape[-1]))

        del delta_phieta, indeces, knn, _

        return x_knn    # x_knn.shape = [B, n, k, d]

    
    def linear_aggregate(self, x):

        # accepts as input [B, d, n, k]

        # take the features of the particle and repeat them on the third axis
        p_feat = x[:, :, :, 0].unsqueeze(3).expand(-1, -1, -1, self.k)

        # now we can calculate knn features for each particle as a simple difference
        knn_feat = x - p_feat

        pairs = torch.concat([p_feat, knn_feat], dim=1)
        del p_feat, knn_feat

        mlp_result = self.mlp(pairs)
        del pairs

        # aggregate
        aggr_result = self.aggr(mlp_result, dim=3)

        if type(aggr_result) is tuple:
            aggr_result = aggr_result[0]
        
        return aggr_result


    def forward(self, x):
        # x.size = [B, n, d]
        # x_knn.size = [B, n, k, d]

        x_knn = self.kNN(x).transpose(1, 3).transpose(2, 3)
        shortcut = self.shortcut(x.transpose(1, 2))
        x = self.linear_aggregate(x_knn)

        x = self.act(x + shortcut).transpose(1, 2)
        
        del x_knn, shortcut
        return x