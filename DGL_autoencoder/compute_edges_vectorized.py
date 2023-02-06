import numpy as np

def compute_dPhi_ij(dPhi):
    """Compute the difference in azimuthal angle between two particles."""
    expanded_dPhi = np.tile(dPhi, (dPhi.shape[0], 1))
    return expanded_dPhi - expanded_dPhi.T

def compute_dEta_ij(dEta):
    """Compute the difference in pseudorapidity between two particles."""
    expanded_dEta = np.tile(dEta, (dEta.shape[0], 1))
    return expanded_dEta - expanded_dEta.T

def compute_R_ij(dEta_ij, dPhi_ij):
    """Compute the distance between two particles in the transverse plane."""
    return np.sqrt(dEta_ij**2 + dPhi_ij**2)

def compute_m_ij(pt, dEta_ij, dPhi_ij):
    """Compute the invariant mass of two particles."""
    # invariant mass of two massive particles as a function of the two transverse momenta and the angles between them
    expanded_pt = np.tile(pt, (pt.shape[0], 1))
    pt_ij = expanded_pt * expanded_pt.T
    return np.sqrt(2. * pt_ij * np.cosh(dEta_ij) - 2. * pt_ij * np.cos(dPhi_ij)) # RELATIVISTIC APPROX

def node_distance(pt, r, r_ij, alpha):
    """Compute the distance between two nodes in the graph."""
    expanded_pt = np.tile(pt, (pt.shape[0], 1))
    return np.minimum(expanded_pt**(2*alpha), expanded_pt.T**(2*alpha)) * r_ij**2/r**2

def compute_1jet_edge_features(jet):
    """Compute the edge feature for one edge."""
    
    dEta_ij = compute_dEta_ij(jet[:, 0])
    dPhi_ij = compute_dPhi_ij(jet[:, 1])
    dR_ij   = compute_R_ij(dEta_ij, dPhi_ij)
    m_ij    = compute_m_ij(jet[:, 4], dEta_ij, dPhi_ij)

    # compute the edge feature
    e_0 = node_distance(pt=jet[:, 4], r=np.max(jet[:, 6]), r_ij=dR_ij, alpha=0)
    e_1 = node_distance(pt=jet[:, 4], r=np.max(jet[:, 6]), r_ij=dR_ij, alpha=1)
    e_2 = m_ij

    # fill with ones on the diagonal
    np.fill_diagonal(e_0, 1.)
    np.fill_diagonal(e_1, np.e)
    np.fill_diagonal(e_2, np.e)

    if np.any(e_2 == 0.):
        print(f'got zero m_ij')
        e_2[e_2==0] = 1e-6

    if np.any(e_1 == 0.):
        print(f'got zero node_distance')
        e_1[e_1==0] = 1e-6

    e_1 = np.log(e_1)
    e_2 = np.log(e_2)

    # flatten
    e_0 = e_0.flatten()
    e_1 = e_1.flatten()
    e_2 = e_2.flatten()

    return np.stack([e_0, e_1, e_2], 1)
    


