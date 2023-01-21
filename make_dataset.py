# Author: Nicolò Lai
# Date: 2023-21-01
#
# Description: 
# This script reads the ROOT file, extracts the features of the jets and particles, and saves them in a CSV file.
# The input file, output files, and data directory can be specified with the -in, -jet, -par, and -dir flags.
# The default values are extracted_data.root, jet_df.csv, particle_df.csv, and ./data/ respectively.
# The script can be run with the command: python make_dataset.py
# The script can be run with the command: python make_dataset.py -in extracted_data.root -jet jet_df.csv -par particle_df.csv -dir ./data/


import argparse
import numpy as np
import pandas as pd

from Dataset import Dataset



def argParser():
    """ 
    Summary:
        The input file, output file, and data directory can be specified with the -in, -jet, -par, and -dir flags.
        The default values are extracted_data.root, jet_df.csv, particle_df.csv, and ./data/ respectively.
        The function returns a parser object.

    Returns:
        parser: parser object
    """
    
    # Create a parser object
    parser = argparse.ArgumentParser() 
    
    # Add the arguments to the parser object
    parser.add_argument('-dir', '--directory', type=str, default='./data/',             help="data directory")
    parser.add_argument('-in',  '--input',     type=str, default='extracted_data.root', help="input file name")
    parser.add_argument('-jet', '--jet',       type=str, default='jet_df.csv',          help="jet output file name")
    parser.add_argument('-par', '--particle',  type=str, default='particle_df.csv',     help="particle output file name")
    
    # add verbose flag
    parser.add_argument('-v', '--verbose', action='store_true', default=False, help="verbose output")

    # Return the parser object
    return parser


def parseArgs(parser):
    """ 
    Summary:
        This function parses the arguments to the program.
        The function returns the input file, output file, and data directory.

    Returns:
        input_file  [str] : input file name
        output_file [str] : output file name
        data_dir    [str] : data directory
    """

    # Parse the arguments
    args = parser.parse_args()

    # Return the input file, output file, and data directory
    return args.input, args.jet, args.particle, args.directory, args.verbose


def getJetFeatures(dataset):
    """
    Summary:
        Extracts the features of the jets in the dataset.

    Args:
        dataset (Dataset): The dataset to get the features of.

    Returns:
        A dictionary of the jet features.
    """
    
    return {
        "jetID":        dataset.getJetID(),
        "jetArea":      dataset.getJetArea(),
        "jetPx":        dataset.getJetPx(),
        "jetPy":        dataset.getJetPy(),
        "jetPz":        dataset.getJetPz(),
        "jetE":         dataset.getJetE(),
        "jetPolarPx":   dataset.getJetPolarPx(),
        "jetPolarPy":   dataset.getJetPolarPy(),
        "jetPolarPz":   dataset.getJetPolarPz(),
        "jetPolarE":    dataset.getJetPolarE(),
        "jetPhi":       dataset.getJetPhi(),
        "jetTheta":     dataset.getJetTheta(),
    }
    

def getParticleFeatures(dataset):
    """
    Summary:
        Extracts the features of the particle in the dataset.
    
    Args:
        dataset (Dataset): The dataset to extract features for.
        
    Returns:
        A dictionary of the particle features.
    """
    
    return {
        "particleType":     dataset.getParticleType(),
        "particleVx":       dataset.getParticleVx(),
        "particleVy":       dataset.getParticleVy(),
        "particleVz":       dataset.getParticleVz(),
        "particlePx":       dataset.getParticlePx(),
        "particlePy":       dataset.getParticlePy(),
        "particlePz":       dataset.getParticlePz(),
        "particleE":        dataset.getParticleE(),
        "particlePolarPx":  dataset.getParticlePolarPx(),
        "particlePolarPy":  dataset.getParticlePolarPy(),
        "particlePolarPz":  dataset.getParticlePolarPz(),
        "particlePolarE":   dataset.getParticlePolarE(),
        "particlePhi":      dataset.getParticlePhi(),
        "particleTheta":    dataset.getParticleTheta(),
    }
    
    
def buildJetDataFrame(jet_features, par_features):
    """
    Summary:
        Builds a DataFrame of the jet features.

    Args:
        jet_features (dict): a dictionary of the jet features.
        par_features (dict): a dictionary of the particle features.

    Returns:
        pandas.DataFrame: A DataFrame of the jet features.
    """
    
    # Initialize the columns of the DataFrame and add the event ID
    ev_id = np.arange(0, len(jet_features["jetID"]), 1, dtype=np.int16)
    jet_columns = ["eventID"] + list(jet_features.keys())[:1] + ["nParticles"] + list(jet_features.keys())[1:]
    jet_df = pd.DataFrame(columns=jet_columns)

    # Loop over the events
    for ev in ev_id:
        # Get the unique jet IDs in the event
        jets = np.unique(jet_features["jetID"][ev])

        # Loop over the jets
        for i, jet in enumerate(jets):
            # Get the number of particles in the jet
            nParticles = len(np.array(par_features["particleType"][ev])[np.array(jet_features["jetID"][ev]) == jet])
            # Create a row of the DataFrame
            jet_row = np.array([
                ev, 
                jet, 
                nParticles, 
                jet_features["jetArea"][ev][i], 
                jet_features["jetPx"][ev][i], 
                jet_features["jetPy"][ev][i], 
                jet_features["jetPz"][ev][i], 
                jet_features["jetE"][ev][i], 
                jet_features["jetPolarPx"][ev][i], 
                jet_features["jetPolarPy"][ev][i], 
                jet_features["jetPolarPz"][ev][i], 
                jet_features["jetPolarE"][ev][i], 
                jet_features["jetPhi"][ev][i], 
                jet_features["jetTheta"][ev][i]
            ])
            # Add the row to the DataFrame
            jet_df = jet_df.append(pd.DataFrame(jet_row.reshape(1, -1), columns=jet_columns), ignore_index=True)
            
    # Set the correct data type for each column
    jet_df["eventID"]    = jet_df["eventID"].astype(np.int16)
    jet_df["jetID"]      = jet_df["jetID"].astype(np.int16)
    jet_df["nParticles"] = jet_df["nParticles"].astype(np.int16)
    
    return jet_df


def buildParticleDataFrame(jet_features, par_features):
    """
    Summary:
        Builds a DataFrame of the particle features.

    Args:
        jet_features (dict): a dictionary of the jet features.
        par_features (dict): a dictionary of the particle features.

    Returns:
        pandas.DataFrame: A DataFrame of the particle features.
    """
    
    # Initialize the columns of the DataFrame and add the event ID
    ev_id = np.arange(0, len(jet_features["jetID"]), 1, dtype=np.int16)
    particle_columns = ["eventID"] + ["jetID"] + list(par_features.keys())
    par_df = pd.DataFrame(columns=particle_columns)

    # Loop over the events
    for ev in ev_id:
        # Ge the number of particles in each jet
        nPartInJet   = list(np.unique(jet_features["jetID"][ev], return_counts=True)[1])
        # Loop over the jets
        for i, n in enumerate(nPartInJet):
            # Get the jet ID
            partJetID = np.unique(jet_features["jetID"][ev])[i]
            # Loop over the particles in the jet
            for p in range(n):
                # Get the particle index
                offset = sum(nPartInJet[:i]) if i > 0 else 0
                p = offset + p
                # Create a row of the DataFrame
                par_row = np.array([
                    ev, 
                    partJetID, 
                    par_features["particleType"][ev][p], 
                    par_features["particleVx"][ev][p], 
                    par_features["particleVy"][ev][p], 
                    par_features["particleVz"][ev][p], 
                    par_features["particlePx"][ev][p], 
                    par_features["particlePy"][ev][p], 
                    par_features["particlePz"][ev][p], 
                    par_features["particleE"][ev][p], 
                    par_features["particlePolarPx"][ev][p], 
                    par_features["particlePolarPy"][ev][p], 
                    par_features["particlePolarPz"][ev][p], 
                    par_features["particlePolarE"][ev][p], 
                    par_features["particlePhi"][ev][p], 
                    par_features["particleTheta"][ev][p]
                ])
                # Add the row to the DataFrame
                par_df = par_df.append(pd.DataFrame(par_row.reshape(1, -1), columns=particle_columns), ignore_index=True)
                offset += n

    # Set the correct data type for each column
    par_df["eventID"]       = par_df["eventID"].astype(np.int16)
    par_df["jetID"]         = par_df["jetID"].astype(np.int16)
    par_df["particleType"]  = par_df["particleType"].astype(np.int16)
    
    return par_df
    
 
def main():  
    
    # Load the input file
    input_file, jet_output, par_output, directory, verbose = parseArgs(argParser())
    # Append a slash to the directory if necessary
    directory  = directory + "/" if directory[-1] != "/" else directory
    # Construct the full path to the input file
    input_data = directory + input_file
    # Construct the full path to the jet output file
    jet_output = directory + jet_output
    # Construct the full path to the parameter output file
    par_output = directory + par_output

    # Load the data into a Dataset object
    dataset = Dataset(fname=input_data)

    if verbose:
        print()
        print("Loading data from file: " + input_data)
        print("Jet output file: "        + jet_output)
        print("Particle output file: "   + par_output)
        print()

    # get the jet features
    if verbose:
        print("Getting jet features...")
    jet_features = getJetFeatures(dataset)

    # get the particle features
    if verbose:
        print("Getting particle features...")
    par_features = getParticleFeatures(dataset)
              
          

    # build jet dataframe
    if verbose:
        print()
        print("Building jet DataFrame...")
    jet_df = buildJetDataFrame(jet_features, par_features)

    # save jet dataframe to a csv file
    if verbose:
        print("Saving jet DataFrame to file...")
    jet_df.to_csv(jet_output, index=False)


    # build particle dataframe
    if verbose:
        print()
        print("Building particle DataFrame...")
    par_df = buildParticleDataFrame(jet_features, par_features)

    # save jet dataframe to a csv file
    if verbose:
        print("Saving particle DataFrame to file...")
    par_df.to_csv(par_output, index=False)
 
 
 
if __name__ == '__main__' : 
    # execute only if run as a script
    main()