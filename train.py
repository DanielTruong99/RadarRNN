import torch
import lightning
import os

from torch.utils.data import DataLoader
from train_cfg import *
from learning.model.rnn_model import RNNModel
from learning.dataset.csv_dataset import CSVDataset
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

# Create DataLoader
dataset_cfg = RadarDatasetsCfg()
train_dataset = CSVDataset(os.path.join(dataset_cfg.dataset_path, 'train'), dataset_cfg.train)
train_loader = DataLoader(train_dataset, batch_size=dataset_cfg.train.batch_size, shuffle=dataset_cfg.train.is_shuffle, num_workers=dataset_cfg.train.num_workers)

val_dataset = CSVDataset(os.path.join(dataset_cfg.dataset_path, 'val'), dataset_cfg.val)
val_loader = DataLoader(val_dataset, batch_size=dataset_cfg.val.batch_size, shuffle=False, num_workers=dataset_cfg.val.num_workers)

test_dataset = CSVDataset(os.path.join(dataset_cfg.dataset_path, 'test'), dataset_cfg.test)
test_loader = DataLoader(test_dataset, batch_size=dataset_cfg.test.batch_size, shuffle=False, num_workers=dataset_cfg.test.num_workers)

# Initialize model
model_cfg = RadarModelCfg()
optimizer_cfg = RadarOptimizerCfg()
model = RNNModel(model_cfg, optimizer_cfg) 

# Train model
early_stop_callback = EarlyStopping(monitor="val_loss", mode="min")
trainer = lightning.Trainer(max_epochs=optimizer_cfg.epochs)
trainer.fit(model, train_loader, val_loader, callbacks=[early_stop_callback])