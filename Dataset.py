import uproot
from dataclasses import dataclass


@dataclass
class DataFormat:
    """Class that holds the structure of the .root file to be read.
    """
    
    # path to the main tree
    root_tree : str = "demo/Events"
    
    # branches
    numberFatJets   : str = "numberfatjet"
    jetID           : str = "jetID"
    jetArea         : str = "jetArea"
    jetVx           : str = "jet_vx"
    jetVy           : str = "jet_vy"
    jetVz           : str = "jet_vz"
    jetPx           : str = "jet_Px"
    jetPy           : str = "jet_Py"
    jetPz           : str = "jet_Pz"
    jetE            : str = "jet_E"
    jetPolarPx      : str = "jet_polarPx"
    jetPolarPy      : str = "jet_polarPy"
    jetPolarPz      : str = "jet_polarPz"
    jetPolarE       : str = "jet_polarE"
    jetPhi          : str = "jet_phi"
    jetTheta        : str = "jet_theta"
    particleType    : str = "particleType"
    particleVx      : str = "particle_vx"
    particleVy      : str = "particle_vy"
    particleVz      : str = "particle_vz"
    particlePx      : str = "particle_Px"
    particlePy      : str = "particle_Py"
    particlePz      : str = "particle_Pz"
    particleE       : str = "particle_E"
    particlePolarPx : str = "particle_polarPx"
    particlePolarPy : str = "particle_polarPy"
    particlePolarPz : str = "particle_polarPz"
    particlePolarE  : str = "particle_polarE"
    particlePhi     : str = "particle_phi"
    particleTheta   : str = "particle_theta"


    # getters
    
    @property
    def getNumberFatJets(self):
        return f"{self.root_tree}/{self.numberFatJets}"
    
    @property
    def getJetID(self):
        return f"{self.root_tree}/{self.jetID}"
    
    @property
    def getJetArea(self):
        return f"{self.root_tree}/{self.jetArea}"
    
    @property
    def getJetVx(self):
        return f"{self.root_tree}/{self.jetVx}"
    
    @property
    def getJetVy(self):
        return f"{self.root_tree}/{self.jetVy}"
    
    @property
    def getJetVz(self):
        return f"{self.root_tree}/{self.jetVz}"
    
    @property
    def getJetPx(self):
        return f"{self.root_tree}/{self.jetPx}"
    
    @property
    def getJetPy(self):
        return f"{self.root_tree}/{self.jetPy}"
    
    @property
    def getJetPz(self):
        return f"{self.root_tree}/{self.jetPz}"
    
    @property
    def getJetE(self):
        return f"{self.root_tree}/{self.jetE}"
    
    @property
    def getJetPolarPx(self):
        return f"{self.root_tree}/{self.jetPolarPx}"
    
    @property
    def getJetPolarPy(self):
        return f"{self.root_tree}/{self.jetPolarPy}"
    
    @property
    def getJetPolarPz(self):
        return f"{self.root_tree}/{self.jetPolarPz}"
    
    @property
    def getJetPolarE(self):
        return f"{self.root_tree}/{self.jetPolarE}"
    
    @property
    def getJetPhi(self):
        return f"{self.root_tree}/{self.jetPhi}"
    
    @property
    def getJetTheta(self):
        return f"{self.root_tree}/{self.jetTheta}"
    
    @property
    def getParticleType(self):
        return f"{self.root_tree}/{self.particleType}"
    
    @property
    def getParticleVx(self):
        return f"{self.root_tree}/{self.particleVx}"
    
    @property
    def getParticleVy(self):
        return f"{self.root_tree}/{self.particleVy}"
    
    @property
    def getParticleVz(self):
        return f"{self.root_tree}/{self.particleVz}"
    
    @property
    def getParticlePx(self):
        return f"{self.root_tree}/{self.particlePx}"
    
    @property
    def getParticlePy(self):
        return f"{self.root_tree}/{self.particlePy}"
    
    @property
    def getParticlePz(self):
        return f"{self.root_tree}/{self.particlePz}"
    
    @property
    def getParticleE(self):
        return f"{self.root_tree}/{self.particleE}"
    
    @property
    def getParticlePolarPx(self):
        return f"{self.root_tree}/{self.particlePolarPx}"
    
    @property
    def getParticlePolarPy(self):
        return f"{self.root_tree}/{self.particlePolarPy}"
    
    @property
    def getParticlePolarPz(self):
        return f"{self.root_tree}/{self.particlePolarPz}"
    
    @property
    def getParticlePolarE(self):
        return f"{self.root_tree}/{self.particlePolarE}"
    
    @property
    def getParticlePhi(self):
        return f"{self.root_tree}/{self.particlePhi}"
    
    @property
    def getParticleTheta(self):
        return f"{self.root_tree}/{self.particleTheta}"
    
    

class Dataset:
    """Interface to the dataset.
    """
    
    def __init__(self, fname: str):
        self.format = DataFormat()
        self.fname  = fname
        
    def getNumberFatJets(self):
        with uproot.open(self.fname) as f:
            return [x for x in f[self.format.getNumberFatJets].array().tolist() if x]
    
    def getJetID(self):
        with uproot.open(self.fname) as f:
            return [x for x in f[self.format.getJetID].array().tolist() if x]
        
    def getJetArea(self):
        with uproot.open(self.fname) as f:
            return [x for x in f[self.format.getJetArea].array().tolist() if x]
    
    def getJetVx(self):
        with uproot.open(self.fname) as f:
            return [x for x in f[self.format.getJetVx].array().tolist() if x]
        
    def getJetVy(self):
        with uproot.open(self.fname) as f:
            return [x for x in f[self.format.getJetVy].array().tolist() if x]
        
    def getJetVz(self):
        with uproot.open(self.fname) as f:
            return [x for x in f[self.format.getJetVz].array().tolist() if x]
        
    def getJetPx(self):
        with uproot.open(self.fname) as f:
            return [x for x in f[self.format.getJetPx].array().tolist() if x]
        
    def getJetPy(self):
        with uproot.open(self.fname) as f:
            return [x for x in f[self.format.getJetPy].array().tolist() if x]
        
    def getJetPz(self):
        with uproot.open(self.fname) as f:
            return [x for x in f[self.format.getJetPz].array().tolist() if x]
        
    def getJetE(self):
        with uproot.open(self.fname) as f:
            return [x for x in f[self.format.getJetE].array().tolist() if x]
        
    def getJetPolarPx(self):
        with uproot.open(self.fname) as f:
            return [x for x in f[self.format.getJetPolarPx].array().tolist() if x]
        
    def getJetPolarPy(self):
        with uproot.open(self.fname) as f:
            return [x for x in f[self.format.getJetPolarPy].array().tolist() if x]
        
    def getJetPolarPz(self):
        with uproot.open(self.fname) as f:
            return [x for x in f[self.format.getJetPolarPz].array().tolist() if x]
        
    def getJetPolarE(self):
        with uproot.open(self.fname) as f:
            return [x for x in f[self.format.getJetPolarE].array().tolist() if x]
        
    def getJetPhi(self):
        with uproot.open(self.fname) as f:
            return [x for x in f[self.format.getJetPhi].array().tolist() if x]
        
    def getJetTheta(self):
        with uproot.open(self.fname) as f:
            return [x for x in f[self.format.getJetTheta].array().tolist() if x]
        
    def getParticleType(self):
        with uproot.open(self.fname) as f:
            return [x for x in f[self.format.getParticleType].array().tolist() if x]
        
    def getParticleVx(self):
        with uproot.open(self.fname) as f:
            return [x for x in f[self.format.getParticleVx].array().tolist() if x]
        
    def getParticleVy(self):
        with uproot.open(self.fname) as f:
            return [x for x in f[self.format.getParticleVy].array().tolist() if x]
        
    def getParticleVz(self):
        with uproot.open(self.fname) as f:
            return [x for x in f[self.format.getParticleVz].array().tolist() if x]
        
    def getParticlePx(self):
        with uproot.open(self.fname) as f:
            return [x for x in f[self.format.getParticlePx].array().tolist() if x]
        
    def getParticlePy(self):
        with uproot.open(self.fname) as f:
            return [x for x in f[self.format.getParticlePy].array().tolist() if x]
        
    def getParticlePz(self):
        with uproot.open(self.fname) as f:
            return [x for x in f[self.format.getParticlePz].array().tolist() if x]
        
    def getParticleE(self):
        with uproot.open(self.fname) as f:
            return [x for x in f[self.format.getParticleE].array().tolist() if x]
        
    def getParticlePolarPx(self):
        with uproot.open(self.fname) as f:
            return [x for x in f[self.format.getParticlePolarPx].array().tolist() if x]
        
    def getParticlePolarPy(self):
        with uproot.open(self.fname) as f:
            return [x for x in f[self.format.getParticlePolarPy].array().tolist() if x]
        
    def getParticlePolarPz(self):
        with uproot.open(self.fname) as f:
            return [x for x in f[self.format.getParticlePolarPz].array().tolist() if x]
        
    def getParticlePolarE(self):
        with uproot.open(self.fname) as f:
            return [x for x in f[self.format.getParticlePolarE].array().tolist() if x]
        
    def getParticlePhi(self):
        with uproot.open(self.fname) as f:
            return [x for x in f[self.format.getParticlePhi].array().tolist() if x]
        
    def getParticleTheta(self):
        with uproot.open(self.fname) as f:
            return [x for x in f[self.format.getParticleTheta].array().tolist() if x]
        
