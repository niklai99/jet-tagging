import numpy as np
import h5py
import os

from tqdm import tqdm

from DataReader import DataReader


def compute_dPhi_ij(dPhi_i, dPhi_j):
    """Compute the difference in azimuthal angle between two particles."""
    return dPhi_i - dPhi_j

def compute_dEta_ij(dEta_i, dEta_j):
    """Compute the difference in pseudorapidity between two particles."""
    return dEta_i - dEta_j

def compute_R_ij(dEta_ij, dPhi_ij):
    """Compute the distance between two particles in the transverse plane."""
    return np.sqrt(dEta_ij**2 + dPhi_ij**2)

def compute_m_ij(e_i, e_j, pt_i, pt_j, dPhi_ij):
    """Compute the invariant mass of two particles."""
    # invariant mass of two massive particles as a function of the two energies, the two transverse momenta and the angle between them
    return np.sqrt(2 * e_i * e_j * (1 - np.cos(dPhi_ij))) # CHECK THIS

def node_distance(pt_i, pt_j, r, r_ij, alpha):
    """Compute the distance between two nodes in the graph."""
    return np.min((pt_i**(2*alpha), pt_j**(2*alpha))) * r_ij**r

def compute_one_edge_feature(jet, i, j):
    """Compute the edge feature for one edge."""
    
    dEta_ij = compute_dEta_ij(jet[i, 0], jet[j, 0])
    dPhi_ij = compute_dPhi_ij(jet[i, 1], jet[j, 1])
    dR_ij   = compute_R_ij(dEta_ij, dPhi_ij)
    m_ij    = compute_m_ij(jet[i, 2], jet[j, 2], jet[i, 4], jet[j, 4], dPhi_ij)

    # compute the edge feature
    e_0 =        node_distance(pt_i=jet[i, 4], pt_j=jet[j, 4], r=jet[i, 6], r_ij=dR_ij, alpha=0)
    e_1 = np.log(node_distance(pt_i=jet[i, 4], pt_j=jet[j, 4], r=jet[i, 6], r_ij=dR_ij, alpha=1))
    e_2 = np.log(m_ij)

    return np.array([e_0, e_1, e_2])
    


def compute_edge_features(data):
    # compute the edge features for all the jets in the dataset
    # final shape of the edges: (n_jets, n_particles * n_particles, 3) 

    # list to store the edge features for all the jets
    edge_features_all = []

    # loop over all the jets
    for k in range(data.shape[0]):

        # get the current jet
        jet = data[k, :, :]
            
        # list to store the edge features for the current jet
        edge_features = []

        # loop over all the particles in the jet
        for i in range(jet.shape[0]):
            # loop over all the particles in the jet
            for j in range(jet.shape[0]):

                # if the two particles are the same, the edge feature is just a vector of ones
                edge_feature = np.ones(3) if i == j else compute_one_edge_feature(jet, i, j)
                edge_features.append(edge_feature)

        # store the edge features for the current jet
        edge_features = np.array(edge_features)
        edge_features_all.append(edge_features)
        
    return np.array(edge_features_all, dtype=object)


if __name__=='__main__':
    
    # load the dataset
    data_reader = DataReader("../data/train/")
    data_reader.read_files(n_files=1)
    
    data = data_reader.get_features()

    # compute the edge features
    print("Computing the edge features...")
    edge_features = compute_edge_features(data)
    
    # save the edge features into a h5 file
    if not os.path.exists("../data/train/edges"):
        os.mkdir("../data/train/edges")
        
    np.savez("../data/train/edges/edge_features.npz", edges=edge_features)