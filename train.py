import sys

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from models import SusNet, WeiTopoNet, MLPTopoNet, AttentionTopoNet
from pathlib import Path


from preprocessing import load_pdbbind_data_index
from dataset import ProteinPersistenceHomologyDataset, CollapsedPairwiseOppositionDataset, ProteinHomologyDataModule, WeiDataset


if sys.platform == 'linux':
    index_location = '/home/longyuxi/Documents/mount/pdbbind-dataset/index/INDEX_refined_data.2020'
else:
    raise Exception

TRANSPOSE_DATASET = True
print('transpose dataset:', TRANSPOSE_DATASET)

index = load_pdbbind_data_index(index_location)
net = WeiTopoNet(transpose=TRANSPOSE_DATASET)
datamodule = ProteinHomologyDataModule(index, transpose=TRANSPOSE_DATASET, batch_size=16)

if sys.platform == 'linux':
    trainer = pl.Trainer(max_epochs=2000, accelerator='gpu', devices=1)
    # trainer = pl.Trainer(max_epochs=100, devices=1)
else:
    raise NotImplementedException

# For training
trainer.fit(net, datamodule=datamodule)

