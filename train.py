import torch
from torch.utils.data import DataLoader
import lightning

from .train_cfg import *
from learning.model.rnn_model import RNNModel
from learning.dataset.csv_dataset import CSVDataset

# Create DataLoader
dataset_cfg = RadarDatasetCfg()
dataset = CSVDataset(dataset_cfg)
train_loader = DataLoader(dataset, batch_size=12, shuffle=True)

# Initialize model
model_cfg = RadarModelCfg()
optimizer_cfg = RadarOptimizerCfg()
model = RNNModel(model_cfg, optimizer_cfg) 

# Train model
trainer = lightning.Trainer(max_epochs=200)
trainer.fit(model, train_loader)