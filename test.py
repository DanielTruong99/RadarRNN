import torch
import lightning
import os
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from train_cfg import *
from learning.model.rnn_model import RNNModel
from learning.dataset.csv_dataset import CSVDataset

# Create DataLoader
dataset_cfg = RadarDatasetsCfg()
test_dataset = CSVDataset(os.path.join(dataset_cfg.dataset_path, 'test'), dataset_cfg.test)
test_loader = DataLoader(test_dataset, batch_size=dataset_cfg.test.batch_size, shuffle=False, num_workers=dataset_cfg.test.num_workers)

'''
    lightning_logs
    ├── version_0
    │   ├── checkpoints
    ├── version_1
    ├── version_2
    ├── version_3
'''
# Load the model from the checkpoint from latest version
# Find latest version folder
latest_version = max([int(version.split("_")[1]) for version in os.listdir("lightning_logs")])
ckpt_path = f"lightning_logs/version_{latest_version}/checkpoints"
ckpt_files = os.listdir(ckpt_path)
latest_ckpt_abs_path = os.path.join(ckpt_path, ckpt_files[0])

# Load the model
model_cfg = RadarModelCfg()
optimizer_cfg = RadarOptimizerCfg()
model = RNNModel.load_from_checkpoint(latest_ckpt_abs_path, model_cfg=model_cfg, optimizer_cfg=optimizer_cfg)
model.eval()
model.to("cpu")

# Predict with trained model
predictions = []
labels = []
with torch.no_grad():
    for batch in test_loader:
        input, label = batch
        predictions.append(model(input))
        labels.append(label)

predictions = torch.cat(predictions, dim=0)
labels = torch.cat(labels, dim=0)

# Plot the predictions
fig, axs = plt.subplots(4, 1, figsize=(10, 25))
for i in range(4):
    axs[i].plot(predictions[1:1000, i], label=f'is_on_r{i+1}')
    axs[i].plot(labels[1:1000, i], label=f'r{i+1}')
    axs[i].legend()
    axs[i].grid(True)

plt.tight_layout()
plt.show()