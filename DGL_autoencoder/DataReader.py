import os
import numpy as np
import pandas as pd
import h5py

class DataReader:
    
    def __init__(self, path):
        
        self.path = path
        self.data = None
        
    def get_filenames(self):
        
        fnames = os.listdir(self.path)
        fnames = [filename for filename in fnames if filename.endswith(".h5")]

        return fnames
    
    
    def read_single_file(self, fname):
        
        file_path = os.path.join(self.path, fname)
        h5_file = h5py.File(file_path, "r")
        
        # turn "jets" into a dataframe with column names given by "jetFeaturesNames"

        jet_feature_names = h5_file["jetFeatureNames"][:]

        # remove the b' from the beginning and the ' from the end of each string
        jet_feature_names = [name.decode("utf-8") for name in jet_feature_names]

        df = pd.DataFrame(h5_file["jets"][:], columns=jet_feature_names)

        # keep features
        keep = ["j_g", "j_q", "j_w", "j_z", "j_t", "j_undef"]

        # rename features to "isGluon", "isQuark", "isW", "isZ", "isTop", "isUndefined"
        df = df[keep].rename(columns={"j_g": "isGluon", "j_q": "isQuark", "j_w": "isW", "j_z": "isZ", "j_t": "isTop", "j_undef": "isUndefined"}).astype(int)


        # LABELS
        labels = df.values

        mask = np.sum(np.abs(h5_file["jetConstituentList"]), axis=2)==0

        # FEATURES
        e      = np.reshape(h5_file["jetConstituentList"][:, :, 3],  (10000, 100)) # E
        e_rel  = np.reshape(h5_file["jetConstituentList"][:, :, 4],  (10000, 100)) # E
        pt     = np.reshape(h5_file["jetConstituentList"][:, :, 5],  (10000, 100)) # pT
        pt_rel = np.reshape(h5_file["jetConstituentList"][:, :, 6],  (10000, 100)) # pT
        dEta   = np.reshape(h5_file["jetConstituentList"][:, :, 8],  (10000, 100)) # dEta
        dPhi   = np.reshape(h5_file["jetConstituentList"][:, :, 11], (10000, 100)) # dPhi
        dR     = np.reshape(h5_file["jetConstituentList"][:, :, 13], (10000, 100)) # dR
        

        return {"labels": labels,"e": e, "e_rel": e_rel, "pt": pt, "pt_rel": pt_rel, "dEta": dEta, "dPhi": dPhi, "dR": dR}


    
    def read_files(self, n_files=None):
        
        fnames = self.get_filenames()
        
        if n_files is not None:
            fnames = fnames[:n_files]
            
        for i, fname in enumerate(fnames):
            
            print("Reading file", fname)
            
            data = self.read_single_file(fname)
            
            if i == 0:
                labels = data["labels"]
                e      = data["e"]
                e_rel  = data["e_rel"]
                pt     = data["pt"]
                pt_rel = data["pt_rel"]
                dEta   = data["dEta"]
                dPhi   = data["dPhi"]
                dR     = data["dR"]
                
            else:
                labels = np.concatenate((labels, data["labels"]))
                e      = np.concatenate((e, data["e"]))
                e_rel  = np.concatenate((e_rel, data["e_rel"]))
                pt     = np.concatenate((pt, data["pt"]))
                pt_rel = np.concatenate((pt_rel, data["pt_rel"]))
                dEta   = np.concatenate((dEta, data["dEta"]))
                dPhi   = np.concatenate((dPhi, data["dPhi"]))
                dR     = np.concatenate((dR, data["dR"]))
    
        self.data = {"labels": labels,"e": e, "e_rel": e_rel, "pt": pt, "pt_rel": pt_rel, "dEta": dEta, "dPhi": dPhi, "dR": dR}        
    
    
    def get_labels(self):
        if self.data:
            return self.data["labels"]
        else:
            raise RuntimeError('No data read, please call .read_files before!')
    
    def get_features(self):
        if self.data:
            return np.stack(
                (
                    self.data["dEta"],
                    self.data["dPhi"],
                    self.data["e"],
                    self.data["e_rel"],
                    self.data["pt"],
                    self.data["pt_rel"],
                    self.data["dR"],
                ),
                axis=2,
            ).astype(np.float32)
        else:
            raise RuntimeError('No data read, please call .read_files before!')
        



class DataReader_ragged:
    
    def __init__(self, path):
        
        self.path = path
        self.data = None
        
    def get_filenames(self):
        
        fnames = os.listdir(self.path)
        fnames = [filename for filename in fnames if filename.endswith(".h5")]

        return fnames
    
    
    def read_single_file(self, fname):
        
        file_path = os.path.join(self.path, fname)
        h5_file = h5py.File(file_path, "r")
        
        # turn "jets" into a dataframe with column names given by "jetFeaturesNames"

        jet_feature_names = h5_file["jetFeatureNames"][:]

        # remove the b' from the beginning and the ' from the end of each string
        jet_feature_names = [name.decode("utf-8") for name in jet_feature_names]

        df = pd.DataFrame(h5_file["jets"][:], columns=jet_feature_names)

        # keep features
        keep = ["j_g", "j_q", "j_w", "j_z", "j_t", "j_undef"]

        # rename features to "isGluon", "isQuark", "isW", "isZ", "isTop", "isUndefined"
        df = df[keep].rename(columns={"j_g": "isGluon", "j_q": "isQuark", "j_w": "isW", "j_z": "isZ", "j_t": "isTop", "j_undef": "isUndefined"}).astype(int)


        # LABELS
        labels = df.values

        mask = np.sum(np.abs(h5_file["jetConstituentList"]), axis=2)==0
        
        # FEATURES
        e      = np.ma.masked_array(h5_file["jetConstituentList"][:, :, 3],  mask) # E
        e_rel  = np.ma.masked_array(h5_file["jetConstituentList"][:, :, 4],  mask) # E
        pt     = np.ma.masked_array(h5_file["jetConstituentList"][:, :, 5],  mask) # pT
        pt_rel = np.ma.masked_array(h5_file["jetConstituentList"][:, :, 6],  mask) # pT particle / jet
        dEta   = np.ma.masked_array(h5_file["jetConstituentList"][:, :, 8],  mask) # dEta
        dPhi   = np.ma.masked_array(h5_file["jetConstituentList"][:, :, 11], mask) # dPhi
        dR     = np.ma.masked_array(h5_file["jetConstituentList"][:, :, 13], mask) # dR
        

        return {"labels": labels,"e": e, "e_rel": e_rel, "pt": pt, "pt_rel": pt_rel, "dEta": dEta, "dPhi": dPhi, "dR": dR}


    
    def read_files(self, n_files=None):
        
        fnames = self.get_filenames()
        
        if n_files is not None:
            fnames = fnames[:n_files]
            
        for i, fname in enumerate(fnames):
            
            print("Reading file", fname)
            
            data = self.read_single_file(fname)
            
            if i == 0:
                labels = data["labels"]
                e      = data["e"]
                e_rel  = data["e_rel"]
                pt     = data["pt"]
                pt_rel = data["pt_rel"]
                dEta   = data["dEta"]
                dPhi   = data["dPhi"]
                dR     = data["dR"]
                
            else:
                labels = np.ma.concatenate((labels, data["labels"]))
                e      = np.ma.concatenate((e, data["e"]))
                e_rel  = np.ma.concatenate((e_rel, data["e_rel"]))
                pt     = np.ma.concatenate((pt, data["pt"]))
                pt_rel = np.ma.concatenate((pt_rel, data["pt_rel"]))
                dEta   = np.ma.concatenate((dEta, data["dEta"]))
                dPhi   = np.ma.concatenate((dPhi, data["dPhi"]))
                dR     = np.ma.concatenate((dR, data["dR"]))
    
        self.data = {"labels": labels,"e": e, "e_rel": e_rel, "pt": pt, "pt_rel": pt_rel, "dEta": dEta, "dPhi": dPhi, "dR": dR}        
    
    
    def get_labels(self):
        if self.data:
            return self.data["labels"]
        else:
            raise RuntimeError('No data read, please call .read_files before!')
    
    def get_features(self):
        if self.data:
            return np.ma.stack(
                (
                    self.data["dEta"],
                    self.data["dPhi"],
                    self.data["e"],
                    self.data["e_rel"],
                    self.data["pt"],
                    self.data["pt_rel"],
                    self.data["dR"],
                ),
                axis=2,
            )
        else:
            raise RuntimeError('No data read, please call .read_files before!')