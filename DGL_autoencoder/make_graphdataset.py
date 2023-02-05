import os
os.environ['DGLBACKEND'] = 'pytorch'

import numpy as np
import torch

from tqdm import tqdm
import dgl

from compute_edges import compute_one_edge_feature
from DataReader import DataReader_ragged


def node_feature_one_jet(jet, i):
    return np.log(jet[i], where=[False, False, True, True, True, True, False])

def get_graphs(jets):
    graphs = []

    for jet in tqdm(jets):

        jet = jet.compressed().reshape((-1, jet.shape[-1]))
        edges_features = []
        nodes_features = []
        sources = []
        destinations = []
        for i in range(jet.shape[0]):
            for j in range(jet.shape[1]):

                edge_features = np.ones(3) if i == j else compute_one_edge_feature(jet, i, j)
            
                edges_features.append(edge_features)
                sources.append(i)
                destinations.append(j)

            nodes_features.append(node_feature_one_jet(jet, i))

        g = dgl.graph((torch.tensor(sources), torch.tensor(destinations)))
        g.edata['d'] = torch.tensor(np.stack(edges_features), dtype=torch.float32)
        g.ndata['f'] = torch.tensor(np.stack(nodes_features), dtype=torch.float32)

        graphs.append(g)

    return graphs



if __name__=='__main__':
    
    # load the dataset
    data_reader = DataReader_ragged("../data/train/")
    data_reader.read_files()
    
    jets = data_reader.get_features()

    graphs = get_graphs(jets)

    dgl.save_graphs('graphs.dgl', graphs)