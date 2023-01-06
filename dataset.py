from pathlib import Path
import pickle

import pandas as pd
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader, random_split

def filter_out_nonexistent_entries(index, homology_base_folder = 'computed_homologies'):
    index.reset_index()
    rows_to_drop = []

    for idx in range(len(index)):
        drop_row = False

        item_pdb_code = index.loc[idx, 'PDB code']
        item_base_folder = Path(homology_base_folder) / item_pdb_code

        pw_opposition = (item_base_folder / 'pairwise_opposition.pckl')
        other_2345_homologies = (item_base_folder / '2345_homology.pckl')
        if not pw_opposition.is_file():
            drop_row = True
        if not other_2345_homologies.is_file():
            drop_row = True

        if drop_row:
            rows_to_drop.append(idx)

    index = index.drop(rows_to_drop)
    index = index.reset_index()

    return index

class ProteinPersistenceHomologyDataset(Dataset):
    def __init__(self,
        index: pd.DataFrame,
        homology_base_folder : str = "computed_homologies",
        use_only_existent_entries=True):

        if use_only_existent_entries:
            index = filter_out_nonexistent_entries(index)

        # pl.utilities.seed.seed_everything(123)
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
        other_2345_homologies = pickle.load(open(item_base_folder / '2345_homology.pckl', 'rb'))
        pw_opposition = torch.from_numpy(pw_opposition)

        # get binding affinity
        binding_affinity = self.index.loc[idx, '-logKd/Ki']
        binding_affinity = torch.tensor([binding_affinity], dtype=torch.float32)

        return [pw_opposition, other_2345_homologies], binding_affinity

class WeiDataset(ProteinPersistenceHomologyDataset):
    # Implemented almost exactly like the one described by Cang and Wei
    def __init__(self,
        index: pd.DataFrame,
        homology_base_folder : str = "computed_homologies",
        normalize=False,
        use_only_existent_entries=True,
        transpose=False):

        self.normalize = normalize
        self.transpose = transpose

        super().__init__(index, homology_base_folder, use_only_existent_entries)

    def __getitem__(self, idx):
        homologies, binding_affinity = super().__getitem__(idx)

        # For pairwise, we only keep Betti-0 and birth (TODO: check that Betti-0's die at the same time)
        pw_opposition = homologies[0] # shape torch.Size([36, 201, 3, 3])
        pw_opposition = pw_opposition[:, :, 0, 0] # now should be of shape torch.Size([36, 201])

        # For 2345, we only wish to keep the Betti-1 and Betti-2. For each of them, flatten the last dimension.
        raw_2345_homologies = homologies[1]
        final_2345_homologies = [] # used as output
        for hom in raw_2345_homologies:
            # iterate through 2345 and doing the same thing

            # hom is of shape ndarray(201, 3, 3)
            hom = hom[:, 1:3, :] # shape ndarray(201, 2, 3)
            hom = hom.reshape(201, 6)

            final_2345_homologies.append(hom)

        # A lot of reshaping to get it into shape (24, 201)
        final_2345_homologies = np.array(final_2345_homologies) # (4, 201, 6)
        pl_heavy_diff = final_2345_homologies[0] - final_2345_homologies[1]
        pl_carbon_diff = final_2345_homologies[2] - final_2345_homologies[3]
        final_2345_homologies = np.concatenate((final_2345_homologies, pl_heavy_diff[None, :, :], pl_carbon_diff[None, :, :]), 0) # (6, 201, 6)
        final_2345_homologies = np.swapaxes(final_2345_homologies, 0, 1) # (201, 6, 6)
        final_2345_homologies = final_2345_homologies.reshape(201, 36)
        final_2345_homologies = np.swapaxes(final_2345_homologies, 0, 1)

        final_2345_homologies = torch.tensor(final_2345_homologies, dtype=torch.float)

        # Finally, concatenate the pairwise homologies and the 2345 homologies to end up with a tensor of shape (60, 201)
        out = torch.concat((pw_opposition, final_2345_homologies), dim=0)
        if self.transpose:
            out = out.T
        if self.normalize:
            out = (out - 1.5) / 30

        return out, binding_affinity

class CollapsedPairwiseOppositionDataset(ProteinPersistenceHomologyDataset):
    def __init__(self,
        index: pd.DataFrame,
        homology_base_folder : str = "computed_homologies",
        normalize=True):

        self.normalize = normalize

        super().__init__(index, homology_base_folder)

    def __getitem__(self, idx):
        homologies, binding_affinity = super().__getitem__(idx)
        pw_opposition = homologies[0]
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
        num_workers = 12,
        use_only_existent_entries = True,
        transpose = False) -> None:

        super().__init__()
        self.index = index
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset = WeiDataset(index, homology_base_folder, use_only_existent_entries=use_only_existent_entries, transpose=transpose)
        print(f'num datapoints: {len(self.dataset)}')

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
        return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=self.num_workers)

if __name__ == '__main__':
    from preprocessing import load_pdbbind_data_index
    import sys

    if sys.platform == 'linux':
        index_location = '/home/longyuxi/Documents/mount/pdbbind-dataset/index/INDEX_refined_data.2020'
    else:
        raise Exception

    index = load_pdbbind_data_index(index_location)
    wd = WeiDataset(index)
    for i in range(len(wd)):
        print(torch.std_mean(wd[i][0]))
