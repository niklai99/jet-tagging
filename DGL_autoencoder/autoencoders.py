
import numpy as np
import torch
import os
os.environ['DGLBACKEND'] = 'pytorch'
import dgl

from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dgl.nn import NNConv, EdgeConv
from dgl.nn.pytorch import Sequential as dglSequential
from dgl.dataloading import GraphDataLoader
from torch.utils.data import Dataset
from tqdm import tqdm

data_path = '../data/graphdataset/'      

class Encoder(nn.Module):
    
    def __init__(self, latent_space_dim, ch=[256,128,64,32]):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(3, ch[0]),
            nn.Dropout(0.2),
            nn.ReLU(),
            
            nn.Linear(ch[0],ch[1]),
            nn.Dropout(0.2),
            nn.ReLU(),
            
            nn.Linear(ch[1],ch[2]),
            nn.Dropout(0.2),
            nn.ReLU(),

            nn.Linear(ch[2], ch[3]),
            nn.Dropout(0.2),
            nn.ReLU(),

            nn.Linear(ch[3], 7*128),
            nn.Sigmoid()
            )
        
        self.conv = dglSequential(
            NNConv(
                in_feats  = 7,   # number of node features
                out_feats = 128, # output number of node features
                edge_func = self.mlp),
            EdgeConv(128, 64, batch_norm=False),
            EdgeConv(64, 32, batch_norm=False),
            EdgeConv(32, latent_space_dim, batch_norm=False)
        )

    def forward(self, graph, n_feat=None):

        x = self.conv(graph, n_feat if n_feat else graph.ndata['f'], graph.edata['d'])
        return x
    

class Decoder(nn.Module):
    
    def __init__(self, latent_space_dim, n_feat=7):
        super().__init__()

        self.shared_path = dglSequential(
            EdgeConv(latent_space_dim, 32, batch_norm=False),
            EdgeConv(32, 64, batch_norm=False),
            EdgeConv(64, 128, batch_norm=False)
        )
        
        self.node_reconstruct = EdgeConv(128, n_feat)    # output are the reconstructed node features

        self.edge_reconstruct1 = dglSequential(
            EdgeConv(128, 32, batch_norm=False),
            EdgeConv(32,16, batch_norm=False),
            EdgeConv(16,8, batch_norm=False)
        )

        self.edge_reconstruct2 = dglSequential(
            EdgeConv(128, 32, batch_norm=False),
            EdgeConv(32,16, batch_norm=False),
            EdgeConv(16,8, batch_norm=False)
        )

        self.edge_reconstruct3 = dglSequential(
            EdgeConv(128, 32, batch_norm=False),
            EdgeConv(32,16, batch_norm=False),
            EdgeConv(16,8, batch_norm=False)
        )

    def forward(self, graph, n_feat=None):
        
        if n_feat is None:
            n_feat = graph.ndata['l']

        # shared path
        shared = self.shared_path(graph, n_feat)

        # node reconstruction
        n = self.node_reconstruct(graph, shared)
        
        # edges reconstruction
        e1 = self.edge_reconstruct1(graph, shared)
        e2 = self.edge_reconstruct2(graph, shared)
        e3 = self.edge_reconstruct3(graph, shared)

        # inner product o matmul?
        e1 = torch.inner(e1, e1)  #their elements are A_{ij}
        e2 = torch.inner(e2, e2) 
        e3 = torch.inner(e3, e3)

        return n, torch.stack([e1, e2, e3], 2)
    

class NNConv_handy(nn.Module):

    def __init__(self, in_feats, out_feats, ch=[256,128,64,32]):
        super().__init__()

        self.mlp = nn.Sequential(
            *[self.block(cin, cout) for cin, cout in zip([3]+ch, ch)],
            nn.Linear(ch[-1], in_feats*out_feats),
            nn.Sigmoid()
        )

        self.nnconv = NNConv(in_feats, out_feats, edge_func=self.mlp)

    def block(self, cin, cout):
        return nn.Sequential(
            nn.Linear(cin, cout),
            nn.Dropout(0.2),
            nn.BatchNorm1d(cout),
            nn.ReLU())
    
    def forward(self, graph, nfeat, efeat=None):
        return self.nnconv(graph, nfeat, graph.edata['d'] if efeat is None else efeat)

class Encoder1(nn.Module):
    """
    adding a NN_conv(64) at the begginning
    """
    def __init__(self, latent_space_dim, ch=[256,128,64,32]):
        super().__init__()

        self.mlp_64 = nn.Sequential(
            nn.Linear(3, ch[0]),
            nn.Dropout(0.2),
            nn.BatchNorm1d(ch[0]),
            nn.ReLU(),
            
            nn.Linear(ch[0],ch[1]),
            nn.Dropout(0.2),
            nn.BatchNorm1d(ch[1]),
            nn.ReLU(),
            
            nn.Linear(ch[1],ch[2]),
            nn.Dropout(0.2),
            nn.BatchNorm1d(ch[2]),
            nn.ReLU(),

            nn.Linear(ch[2], ch[3]),
            nn.Dropout(0.2),
            nn.BatchNorm1d(ch[3]),
            nn.ReLU(),

            nn.Linear(ch[3], 7*64),
            nn.Sigmoid()
            )
        self.mlp_128 = nn.Sequential(
            nn.Linear(3, ch[0]),
            nn.Dropout(0.2),
            nn.BatchNorm1d(ch[0]),
            nn.ReLU(),
            
            nn.Linear(ch[0],ch[1]),
            nn.Dropout(0.2),
            nn.BatchNorm1d(ch[1]),
            nn.ReLU(),
            
            nn.Linear(ch[1],ch[2]),
            nn.Dropout(0.2),
            nn.BatchNorm1d(ch[2]),
            nn.ReLU(),

            nn.Linear(ch[2], ch[3]),
            nn.Dropout(0.2),
            nn.BatchNorm1d(ch[3]),
            nn.ReLU(),

            nn.Linear(ch[3], 64*128),
            nn.Sigmoid()
            )

        self.conv1 = NNConv(
                in_feats  = 7,   # number of node features
                out_feats = 64,  # output number of node features
                edge_func = self.mlp_64)
        
        self.conv2 = dglSequential(
            NNConv(
                in_feats  = 64,  # number of node features
                out_feats = 128, # output number of node features
                edge_func = self.mlp_128),
            EdgeConv(128, 64, batch_norm=True),
            EdgeConv(64, 32, batch_norm=True),
            EdgeConv(32, latent_space_dim, batch_norm=True)
        )

    def forward(self, graph, n_feat=None):

        x = self.conv1(graph, n_feat if n_feat else graph.ndata['f'], graph.edata['d'])   
        x = self.conv2(graph, x, graph.edata['d'])
        return x
    

class Decoder1(nn.Module):
    """Adding 2 edge conv block(64,32) in the node reconstruction path"""
    def __init__(self, latent_space_dim, n_feat=7):
        super().__init__()

        self.shared_path = dglSequential(
            EdgeConv(latent_space_dim, 32, batch_norm=True),
            EdgeConv(32, 64, batch_norm=True),
            EdgeConv(64, 128, batch_norm=True)
        )
        
        self.node_reconstruct = dglSequential(
            EdgeConv(128, 64),
            EdgeConv(64, 32),
            EdgeConv(32, n_feat),   # output are the reconstructed node features
        )

        self.edge_reconstruct1 = dglSequential(
            EdgeConv(128, 32, batch_norm=True),
            EdgeConv(32,16, batch_norm=True),
            EdgeConv(16,8, batch_norm=True)
        )

        self.edge_reconstruct2 = dglSequential(
            EdgeConv(128, 32, batch_norm=True),
            EdgeConv(32,16, batch_norm=True),
            EdgeConv(16,8, batch_norm=True)
        )

        self.edge_reconstruct3 = dglSequential(
            EdgeConv(128, 32, batch_norm=True),
            EdgeConv(32,16, batch_norm=True),
            EdgeConv(16,8, batch_norm=True)
        )

    def forward(self, graph, n_feat=None):
        
        if n_feat is None:
            n_feat = graph.ndata['l']

        # shared path
        shared = self.shared_path(graph, n_feat)

        # node reconstruction
        n = self.node_reconstruct(graph, shared)
        
        # edges reconstruction
        e1 = self.edge_reconstruct1(graph, shared)
        e2 = self.edge_reconstruct2(graph, shared)
        e3 = self.edge_reconstruct3(graph, shared)

        # inner product o matmul?
        e1 = torch.inner(e1, e1)  #their elements are A_{ij}
        e2 = torch.inner(e2, e2) 
        e3 = torch.inner(e3, e3)

        return n, torch.stack([e1, e2, e3], 2)
    
"""Autoencoder 2"""
class Encoder2(nn.Module):
    """
    modify the MLP and the encoder
    """
    def __init__(self, latent_space_dim, ch=[32,64,128,256]):
        super().__init__()

        self.mlp_256= nn.Sequential(
            nn.Linear(3, ch[0]),
            nn.Dropout(0.2),
            nn.BatchNorm1d(ch[0]),
            nn.ReLU(),
            
            nn.Linear(ch[0],ch[1]),
            nn.Dropout(0.2),
            nn.BatchNorm1d(ch[1]),
            nn.ReLU(),
            
            nn.Linear(ch[1],ch[2]),
            nn.Dropout(0.2),
            nn.BatchNorm1d(ch[2]),
            nn.ReLU(),

            nn.Linear(ch[2], ch[3]),
            nn.Dropout(0.2),
            nn.BatchNorm1d(ch[3]),
            nn.ReLU(),

            nn.Linear(ch[3], 7*256),
            nn.Sigmoid()
            )

       
        self.conv = dglSequential(
            NNConv(
                in_feats  = 7,  # number of node features
                out_feats = 256, # output number of node features
                edge_func = self.mlp_256),
            EdgeConv(256, 128, batch_norm=True),
            EdgeConv(128, 64, batch_norm=True),
            EdgeConv(64, 32, batch_norm=True),
            EdgeConv(32, latent_space_dim, batch_norm=True)
        )

    def forward(self, graph, n_feat=None):

        x = self.conv(graph, n_feat if n_feat else graph.ndata['f'], graph.edata['d'])   
        return x
    


"""Autoencoder 3"""
class Encoder3(nn.Module):
    """
    proviamo a dog dick
    """
    def __init__(self, latent_space_dim, ch=[32,64,128,256]):
        super().__init__()
        
        self.conv = dglSequential(
            NNConv_handy(
                in_feats  = 7,  # number of node features
                out_feats = 64, # output number of node features
                ch=[32,64,128]),
            NNConv_handy(
                in_feats  = 64,  # number of node features
                out_feats = 128, # output number of node features
                ch=[64,128,256]),
            EdgeConv(128, 64, batch_norm=True),
            EdgeConv(64, 32, batch_norm=True),
            EdgeConv(32, latent_space_dim, batch_norm=True)
            )

    def forward(self, graph, n_feat=None):

        x = self.conv(graph, graph.ndata['f'] if n_feat is None else n_feat, graph.edata['d'])   
        return x
    


class Encoder4(nn.Module):
    
    def __init__(self, latent_space_dim, ch=[256,128,64,32]):
        super().__init__()

        
        self.conv = dglSequential(
            NNConv_handy(
                in_feats  = 7,  
                out_feats = 256), 
            EdgeConv(256, 128, batch_norm=False),
            EdgeConv(128, 64, batch_norm=False),
            EdgeConv(64,  32, batch_norm=False),
            EdgeConv(32,  16, batch_norm=False),
            EdgeConv(16,  latent_space_dim, batch_norm=False)
        )

    def forward(self, graph, n_feat=None):

        x = self.conv(graph, n_feat if n_feat else graph.ndata['f'], graph.edata['d'])
        return x



class Decoder4(nn.Module):
    
    def __init__(self, latent_space_dim, n_feat=7):
        super().__init__()

        self.shared_path = dglSequential(
            EdgeConv(latent_space_dim, 32, batch_norm=False),
            EdgeConv(32, 64, batch_norm=False),
            EdgeConv(64, 128, batch_norm=False)
        )
        
        self.node_reconstruct = dglSequential(
            EdgeConv(128, 64),
            EdgeConv(64, 32),
            EdgeConv(32, n_feat),   # output are the reconstructed node features
        )
        self.edge_reconstruct1 = dglSequential(
            EdgeConv(128, 64, batch_norm=False),
            EdgeConv(64, 32, batch_norm=False),
            EdgeConv(32,16, batch_norm=False),
            EdgeConv(16,8, batch_norm=False)
        )

        self.edge_reconstruct2 = dglSequential(
            EdgeConv(128, 64, batch_norm=False),
            EdgeConv(64, 32, batch_norm=False),
            EdgeConv(32,16, batch_norm=False),
            EdgeConv(16,8, batch_norm=False)
        )

        self.edge_reconstruct3 = dglSequential(
            EdgeConv(128, 64, batch_norm=False),
            EdgeConv(64, 32, batch_norm=False),
            EdgeConv(32,16, batch_norm=False),
            EdgeConv(16,8, batch_norm=False)
        )

    def forward(self, graph, n_feat=None):
        
        if n_feat is None:
            n_feat = graph.ndata['l']

        # shared path
        shared = self.shared_path(graph, n_feat)

        # node reconstruction
        n = self.node_reconstruct(graph, shared)
        
        # edges reconstruction
        e1 = self.edge_reconstruct1(graph, shared)
        e2 = self.edge_reconstruct2(graph, shared)
        e3 = self.edge_reconstruct3(graph, shared)

        # inner product o matmul?
        e1 = torch.inner(e1, e1)  #their elements are A_{ij}
        e2 = torch.inner(e2, e2) 
        e3 = torch.inner(e3, e3)

        return n, torch.stack([e1, e2, e3], 2)
    
class Encoder5(nn.Module):
    """
    proviamo a dog dick
    """
    def __init__(self, latent_space_dim, ch=[32,64,128,256]):
        super().__init__()
        
        self.conv = dglSequential(
            NNConv_handy(
                in_feats  = 7,  # number of node features
                out_feats = 32, # output number of node features
                ch=[64,32,16]),
            NNConv_handy(
                in_feats  = 32,  # number of node features
                out_feats = 128, # output number of node features
                ch=[32,64,128]),
            EdgeConv(128, 64, batch_norm=False),
            EdgeConv(64, 32, batch_norm=False),
            EdgeConv(32, latent_space_dim, batch_norm=False)
            )

    def forward(self, graph, n_feat=None):

        x = self.conv(graph, graph.ndata['f'] if n_feat is None else n_feat, graph.edata['d'])   
        return x

class Decoder5(nn.Module):
    """Adding 2 edge conv block(64,32) in the node reconstruction path"""
    def __init__(self, latent_space_dim, n_feat=7):
        super().__init__()

        self.shared_path = dglSequential(
            EdgeConv(latent_space_dim, 32),
            EdgeConv(32, 64),
            EdgeConv(64, 128)
        )
        
        self.node_reconstruct = dglSequential(
            EdgeConv(128, 64),
            EdgeConv(64, 32),
            EdgeConv(32, n_feat),   # output are the reconstructed node features
        )

        self.edge_reconstruct1 = dglSequential(
            EdgeConv(128, 64),
            EdgeConv(64, 32),
            EdgeConv(32,16),
            EdgeConv(16,8)
        )

        self.edge_reconstruct2 = dglSequential(
            EdgeConv(128, 64),
            EdgeConv(64, 32),
            EdgeConv(32,16),
            EdgeConv(16,8)
        )

        self.edge_reconstruct3 = dglSequential(
            EdgeConv(128, 64),
            EdgeConv(64, 32),
            EdgeConv(32,16),
            EdgeConv(16,8)
        )

    def forward(self, graph, n_feat=None):
        
        if n_feat is None:
            n_feat = graph.ndata['l']

        # shared path
        shared = self.shared_path(graph, n_feat)

        # node reconstruction
        n = self.node_reconstruct(graph, shared)
        
        # edges reconstruction
        e1 = self.edge_reconstruct1(graph, shared)
        e2 = self.edge_reconstruct2(graph, shared)
        e3 = self.edge_reconstruct3(graph, shared)

        # inner product o matmul?
        e1 = torch.inner(e1, e1)  #their elements are A_{ij}
        e2 = torch.inner(e2, e2) 
        e3 = torch.inner(e3, e3)

        return n, torch.stack([e1, e2, e3], 2)