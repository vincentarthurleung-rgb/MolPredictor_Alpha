import os
import warnings
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import rdBase
from rdkit.Chem import Draw
from rdkit.Chem import AllChem

rdBase.DisableLog('rdApp.error')
rdBase.DisableLog('rdApp.warning')
warnings.filterwarnings("ignore", category=DeprecationWarning)

def show_molecule(df,idx,save_path="data/molecules/"):
    """
    Generate and save a molecular structure image from a SMILES string.

    Parameters:
        df (pd.DataFrame): DataFrame containing 'mol_id' and 'smiles' columns.
        idx (int): Row index of the molecule to visualize.
        save_path (str, optional): Directory to save the image. Defaults to "data/molecules/".

    Behavior:
        - Validates index bounds; prints 'error' and returns if out of range.
        - Retrieves mol_id and SMILES from the specified row.
        - Prints molecule ID and SMILES string.
        - Converts SMILES to an RDKit molecule object using Chem.MolFromSmiles.
        - Ensures the output directory exists (creates if missing).
        - Generates a PIL image of the molecule (size 400x400) with Draw.MolToImage.
        - Saves the image as a PNG file named {mol_id}_{idx}.png.
        - Prints a confirmation message with the saved file path.
    """
    # Check if the passed index is out of range.
    if idx < 0 or idx > len(df) - 1:
        print('error')
        return

    # Get the identifier and SMILES (mol_id and smiles of Tox21).
    mol_id = df['mol_id'].iloc[idx]
    smiles = df['smiles'].iloc[idx]

    # Print text information.
    print(f'\n[Generating Molecule Image for Index {idx}]')
    print(f'Molecule ID: {mol_id}')
    print(f'SMILES: {smiles}')

    # Convert the string to a molecule object.
    mol = Chem.MolFromSmiles(smiles)
    
    # Ensure the output directory exists.
    os.makedirs(save_path, exist_ok=True)

    filename = f"{save_path}{mol_id}_{idx}.png"

    # 1. First generate a PIL image object.
    img = Draw.MolToImage(mol, size=(400, 400))
    
    # 2. Call the image object's save method to save it to disk.
    img.save(filename)
    
    print(f'Successfully saved molecule image to: {filename}')

def extract_features(df):
    """
    Extract features (Morgan fingerprints) and labels from the Tox21 DataFrame.

    Parameters:
        df (pd.DataFrame): DataFrame containing SMILES and label columns.
                           - The last column (index -1) is assumed to be the SMILES strings.
                           - Label columns are from column 0 up to (but excluding) the second last column.

    Returns:
        X (np.ndarray): Feature matrix of shape (n_samples, 1024), each row is a 1024-bit Morgan fingerprint (binary vector).
        y (pd.DataFrame): Label DataFrame of shape (n_samples, n_labels), preserving original label values.

    Process:
        1. Split out the SMILES column (last column) as the feature source.
        2. Split out the label columns (columns 0 to second-last).
        3. For each SMILES:
           - Convert to an RDKit molecule object.
           - If valid, generate a Morgan fingerprint with radius=2 and nBits=1024 (bit vector).
           - If invalid, pad with a zero vector of length 1024.
        4. Stack all fingerprints into a numpy array X.
        5. Print shapes of X and y.
        6. Return X, y.
    
    (original function body)
    """
    # Split out the feature columns
    smiles = df.iloc[:,-1]

    # Split out the label columns.
    y = df.iloc[:,0:-2]

    # Convert each SMILES to a 1024-bit Morgan fingerprint (zero-filled if invalid) and store as feature matrix X
    fps = []
    print("Converting SMILES to Morgan Fingerprints...")
    for sm in smiles:
        mol = Chem.MolFromSmiles(sm)
        if mol:
            # Generate a 1024-bit binary vector (0/1).
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
            fps.append(np.array(fp))
        else:
            # If the SMILES is invalid, pad with a zero vector to maintain consistent dimensions.
            fps.append(np.zeros(1024))
            
    X = np.array(fps) # Convert to a numpy array.

    # print features and labels
    print(f"X shape (features): {X.shape}") # maybe is (7831, 1024)
    print(f"y shape (labels): {y.shape}")     # maybe is (7831, 12)

    return X,y



