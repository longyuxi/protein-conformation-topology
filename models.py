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

class WeiTopoNet(pl.LightningModule):
    # Implemented exactly like described in the 2017 paper:
    #     Step size 0.001 with Glorot uniform initialization
    #     Padding mode for each pair of conv layers are first 'same' and then 'valid'
    #
    # Takes input size [n, 72, 201] or input size [n, 201, 72] (unclear in paper)
    # If first case (transpose), then nn.Linear(12288, 2048) after convolutional layers
    # If second case (no transpose), then nn.Linear(4096, 2048) after convolutional layers
    def __init__(self, transpose=False):
        super().__init__()
        if transpose:
            self.c1 = self.conv(in_channels=201, padding='same')
        else:
            self.c1 = self.conv(in_channels=72, padding='same')
        self.c2 = self.conv(padding='valid')
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout1d(0.25)

        self.c3 = self.conv(in_channels=128, out_channels=256, padding='same')
        self.c4 = self.conv(in_channels=256, out_channels=256, padding='valid')

        if transpose:
            self.d1 = self.dense(4096, 2048)
        else:
            self.d1 = self.dense(12288, 2048)

        self.d2 = self.dense(2048, 2048)
        self.d3 = self.dense(2048, 2048, activation=nn.ReLU())
        self.dropout2 = nn.Dropout1d(0.25)

        self.fc = nn.Linear(2048, 1)
        self.criterion = nn.MSELoss()
        self.glorot_init()


    def glorot_init(self):
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)

        self.apply(init_weights)

    def conv(self, in_channels=128, out_channels=128, kernel_size=3, padding='same'):
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
        return torch.optim.Adam(self.parameters(), lr=0.0001)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss)

class MLPTopoNet(pl.LightningModule):
    # Takes input size [72, 201]
    def __init__(self):
        super().__init__()
        self.d1 = self.dense(14472, 14472)
        self.d2 = self.dense(14472, 4608)
        self.d3 = self.dense(4608, 2048)
        self.dropout1 = nn.Dropout1d(0.25)

        self.d4 = self.dense(2048, 2048)
        self.d5 = self.dense(2048, 2048)
        self.d6 = self.dense(2048, 2048)

        self.dropout2 = nn.Dropout1d(0.25)
        self.fc = nn.Linear(2048, 1)

        self.criterion = nn.MSELoss()


    def conv(self, in_channels=128, out_channels=128, kernel_size=3, padding='same'):
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
        x = nn.Flatten()(x)

        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        x = self.dropout1(x)

        x = self.d4(x)
        x = self.d5(x)
        x = self.d6(x)
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

class AttentionBlock(nn.Module):
    def __init__(self, input_dim, embed_dim=1, num_heads=1):
        super().__init__()
        # input is a batched 1D array
        self.ma1 = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.norm1 = nn.LayerNorm((3600, 1))
        self.ma2 = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.norm2 = nn.LayerNorm((3600, 1))

        self.ff = nn.Linear(input_dim[1], input_dim[1])

    def forward(self, x):
        x = torch.reshape(x, (*x.shape, 1))
        x = x + self.ma1(x, x, x, need_weights=False)[0]
        x = self.norm1(x)
        x = x + self.ma2(x, x, x, need_weights=False)[0]
        x = self.norm2(x)
        x = torch.flatten(x, start_dim=-2)
        x = self.ff(x)

        return x

class AttentionTopoNet(pl.LightningModule):
    # Takes input size [n_batch, 72, 201]
    # We will trim it down to [n_batch, 72, 50] just to keep the runtime reasonable
    def __init__(self, batch_size):
        super().__init__()
        self.ab1 = AttentionBlock((batch_size, 3600))
        self.ab2 = AttentionBlock((batch_size, 3600))
        self.ab3 = AttentionBlock((batch_size, 3600))
        self.ab4 = AttentionBlock((batch_size, 3600))
        self.ab5 = AttentionBlock((batch_size, 3600))
        self.ab6 = AttentionBlock((batch_size, 3600))
        self.fc = nn.Linear(3600, 1)

        self.criterion = nn.MSELoss()

    def forward(self, x):
        x = x[:, :, :50]
        x = torch.flatten(x, start_dim=1) # [n_batch, 3600]

        x = self.ab1(x)
        x = self.ab2(x)
        x = self.ab3(x)
        x = self.ab4(x)
        x = self.ab5(x)
        x = self.ab6(x)

        x = self.fc(x)
        return x

    def dense(self, in_features, out_features, activation=nn.Tanh()):
        return nn.Sequential(
            nn.Linear(in_features, out_features),
            activation
            )


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