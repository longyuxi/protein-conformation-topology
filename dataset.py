from pathlib import Path
import pickle

import pandas as pd
from Bio.PDB import PDBParser
from biopandas.mol2 import PandasMol2
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader, random_split

def load_pdbbind_data_index(index_filename: str) -> pd.DataFrame:
    index = pd.read_csv(index_filename, delim_whitespace=True, skiprows=6, names=['PDB code', "resolution", "release year", "-logKd/Ki", "Kd/Ki", "slashes", "reference", "ligand name"])

    index.drop(columns='slashes', inplace=True)
    index['ligand name'] = index.apply(lambda row:  row['ligand name'][1:][:-1], axis=1)

    return index

def get_pdb_coordinates(file) -> np.ndarray:
    parser = PDBParser(PERMISSIVE=1, QUIET=True)
    structure = parser.get_structure(file, file)
    # Generate a list of the protein's atoms' R^3 coordinates
    coords = []
    for atom in structure.get_atoms():
        coords.append(list(atom.get_vector()))
    coords = np.array(coords)

    return coords

def get_pdb_coordinates_heavy(file) -> np.ndarray:
    # Similar to get_pdb_coordinates, but only gets the C, N, O, S coordinates
    parser = PDBParser(PERMISSIVE=1, QUIET=True)
    structure = parser.get_structure(file, file)

    coords = []
    for atom in structure.get_atoms():
        if atom.element in ['C', 'N', 'O', 'S']:
            coords.append(list(atom.get_vector()))

    coords = np.array(coords)
    coords = np.round(coords, 5)

    return coords

def get_pdb_coordinates_by_element(file, element) -> np.ndarray:
    # Similar to get_pdb_coordinates, but only gets the C, N, O, S coordinates
    parser = PDBParser(PERMISSIVE=1, QUIET=True)
    structure = parser.get_structure(file, file)

    coords = []
    for atom in structure.get_atoms():
        if atom.element == element:
            coords.append(list(atom.get_vector()))

    if len(coords) == 0:
        return None

    coords = np.array(coords)
    coords = np.round(coords, 5)

    return coords

def get_mol2_coordinates(file):
    file = str(file)
    pmol = PandasMol2().read_mol2(file)
    return pmol.df[['x', 'y', 'z']].to_numpy()

def get_mol2_coordinates_heavy(file) -> np.ndarray:
    # Similar to get_mol2_coordinates, but only selects the following heavy atoms as detailed Cang and Wei: {C; N; O; S; P; F; Cl; Br}
    file = str(file)
    pmol = PandasMol2().read_mol2(file)
    coords = []

    for idx in range(len(pmol.df)):
        sybyl_atom_type = pmol.df.loc[idx, 'atom_type']
        if sybyl_atom_type.split('.')[0] in ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I']:
            coords.append(pmol.df.loc[idx, ['x', 'y', 'z']].to_numpy())

    if len(coords) == 0:
        raise Exception('No heavy element.')
    coords = np.array(coords)
    return coords

def get_mol2_coordinates_by_element(file, element) -> np.ndarray:
    # Similar to get_mol2_coordinates, but only selects the following heavy atoms as detailed Cang and Wei: {C; N; O; S; P; F; Cl; Br}
    file = str(file)
    pmol = PandasMol2().read_mol2(file)
    coords = []

    for idx in range(len(pmol.df)):
        sybyl_atom_type = pmol.df.loc[idx, 'atom_type']
        if sybyl_atom_type.split('.')[0] == element:
            coords.append(pmol.df.loc[idx, ['x', 'y', 'z']].to_numpy())

    if len(coords) == 0:
        return None
    coords = np.array(coords)
    return coords


class ProteinPersistenceHomologyDataset(Dataset):
    def __init__(self,
        index: pd.DataFrame,
        homology_base_folder : str = "computed_homologies"):

        pl.utilities.seed.seed_everything(123)
        self.index = index
        self.homology_base_folder = homology_base_folder

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        item_pdb_code = self.index.loc[idx, 'PDB code']
        item_base_folder = Path(self.homology_base_folder) / item_pdb_code

        # get persistence homology data
        # protein_ph = pickle.load(open(item_base_folder / 'protein.pckl', 'rb'))
        # pocket_ph = pickle.load(open(item_base_folder / 'pocket.pckl', 'rb'))
        # ligand_ph = pickle.load(open(item_base_folder / 'ligand.pckl', 'rb'))
        pw_opposition = pickle.load(open(item_base_folder / 'pairwise_opposition.pckl', 'rb'))
        pw_opposition = torch.from_numpy(pw_opposition)

        # get binding affinity
        binding_affinity = self.index.loc[idx, '-logKd/Ki']
        binding_affinity = torch.tensor([binding_affinity], dtype=torch.float32)

        return pw_opposition, binding_affinity

class CollapsedPairwiseOppositionDataset(ProteinPersistenceHomologyDataset):
    def __init__(self,
        index: pd.DataFrame,
        homology_base_folder : str = "computed_homologies",
        normalize=True):

        self.normalize = normalize

        super().__init__(index, homology_base_folder)

    def __getitem__(self, idx):
        pw_opposition, binding_affinity = super().__getitem__(idx)
        collapsed_pw_opposition = torch.zeros((pw_opposition.shape[0], pw_opposition.shape[1]))

        for i in range(len(collapsed_pw_opposition)):
            for j in range(len(collapsed_pw_opposition[0])):
                collapsed_pw_opposition[i, j] = torch.sum(pw_opposition[i, j, :, :])

        if self.normalize:
            collapsed_pw_opposition = (collapsed_pw_opposition - 900) / 2500

        return collapsed_pw_opposition, binding_affinity

class ProteinHomologyDataModule(pl.LightningDataModule):
    def __init__(self,
        index : pd.DataFrame,
        batch_size = 8,
        homology_base_folder : str = "computed_homologies",
        num_workers = 12) -> None:

        super().__init__()
        self.index = index
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset = CollapsedPairwiseOppositionDataset(index, homology_base_folder)

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        train_size = int(0.6 * len(self.dataset))
        val_size = int(0.2 * len(self.dataset))
        test_size = int(0.1 * len(self.dataset))
        predict_size = len(self.dataset) - train_size - val_size - test_size

        self.train_set, self.val_set, self.test_set, self.predict_set = random_split(self.dataset, [train_size, val_size, test_size, predict_size])

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.predict_set, batch_size=self.batch_size, num_workers=self.num_workers)