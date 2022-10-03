import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

class SusNet(pl.LightningModule):
    # Takes input size [36, 201]
    # Not sure if this is correct? Or is it [201, 36]
    def __init__(self):
        super().__init__()
        self.c1 = self.conv(in_channels=36)
        self.c2 = self.conv()
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout1d(0.25)

        self.c3 = self.conv(in_channels=144, out_channels=288)
        self.c4 = self.conv(in_channels=288, out_channels=288)

        self.d1 = self.dense(14400, 2048)
        self.d2 = self.dense(2048, 2048)
        self.d3 = self.dense(2048, 2048, activation=nn.ReLU())
        self.dropout2 = nn.Dropout1d(0.25)

        self.fc = nn.Linear(2048, 1)
        self.criterion = nn.MSELoss()


    def conv(self, in_channels=144, out_channels=144, kernel_size=3, padding='same'):
        return nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU()
            )

    def dense(self, in_features, out_features, activation=nn.Tanh()):
        return nn.Sequential(
            nn.Linear(in_features, out_features),
            activation
            )


    def forward(self, x):

        x = self.c1(x)
        x = self.c2(x)
        x = self.pool(x)
        x = self.dropout(x)

        x = self.c3(x)
        x = self.c4(x)
        x = self.pool(x)
        x = self.dropout(x)

        x = nn.Flatten()(x)

        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        x = self.dropout2(x)

        x = self.fc(x)

        return x

    def on_fit_start(self):
        self.tensorboard = self.logger.experiment
        self.tensorboard.add_text('Model', str(self))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.002)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss)