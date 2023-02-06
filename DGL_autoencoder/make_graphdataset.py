import os
os.environ['DGLBACKEND'] = 'pytorch'

import numpy as np
import torch

from tqdm import tqdm
import dgl

from compute_edges_vectorized import compute_1jet_edge_features
from DataReader import DataReader_ragged


def compute_1jet_nodes_features(jet):
    mask = np.tile(np.array([False, False, True, True, True, True, False]), (jet.shape[0], 1))
    result = jet.copy()
    np.log(jet, where=mask, out=result)
    return result

def get_graphs(jets, desc):
    graphs = []

    for jet in tqdm(jets, desc):

        jet = jet.compressed().reshape((-1, jet.shape[-1]))
        n   = jet.shape[0]

        sources = torch.arange(0, n).unsqueeze(1).expand((n, n)).flatten()
        destinations = torch.arange(0, n).unsqueeze(0).expand((n, n)).flatten()
        
        edges_features = compute_1jet_edge_features(jet)
        nodes_features = compute_1jet_nodes_features(jet)

        g = dgl.graph((sources, destinations))
        g.edata['d'] = torch.tensor(edges_features, dtype=torch.float32)
        g.ndata['f'] = torch.tensor(nodes_features, dtype=torch.float32)

        graphs.append(g)

    return graphs



if __name__=='__main__':
    
    # load the dataset
    data_reader = DataReader_ragged("../data/train/")
    data_reader.read_files()
    
    jets = data_reader.get_features()
    split = 20_000

    for i in tqdm(range(0, jets.shape[0], split), 'files'):
        f = i+split if i+split < jets.shape[0] else jets.shape[0]

        graphs = get_graphs(jets[i:f], f'Jets [{i}-{f}]')

        dgl.save_graphs(f'../data/graphdataset/graphs{i}-{f}.dgl', graphs)

    print('All done!')