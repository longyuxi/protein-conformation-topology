import pandas as pd
from Bio.PDB import PDBParser
from biopandas.mol2 import PandasMol2
import numpy as np

def load_pdbbind_data_index(index_filename):
    index = pd.read_csv(index_filename, delim_whitespace=True, skiprows=6, names=['PDB code', "resolution", "release year", "-logKd/Ki", "Kd/Ki", "slashes", "reference", "ligand name"])

    index.drop(columns='slashes', inplace=True)
    index['ligand name'] = index.apply(lambda row:  row['ligand name'][1:][:-1], axis=1)

    index['homology calculated?'] = [False] * len(index)

    return index

def get_pdb_coordinates(file):
    parser = PDBParser(PERMISSIVE=1, QUIET=True)
    structure = parser.get_structure(file, file)
    # Generate a list of the protein's atoms' R^3 coordinates
    coords = []
    for atom in structure.get_atoms():
        coords.append(list(atom.get_vector()))
    coords = np.array(coords)

    return coords

def get_mol2_coordinates(file):
    file = str(file)
    pmol = PandasMol2().read_mol2(file)
    return pmol.df[['x', 'y', 'z']].to_numpy()