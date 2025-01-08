import torch

from learning.model.cfg import ModelCfg, OptimizerCfg
from learning.dataset.dataset_cfg import DatasetCfg

class RadarDatasetCfg(DatasetCfg):
    input_name_list = ["X", "Y", "Z"]
    label_name_list = ["Vx", "Vy"]
    sequence_length = 5
    data_type = torch.float32

class RadarModelCfg(ModelCfg):
    input_dim = 3
    hidden_dim = 128
    num_layers = 2
    rnn_type = 'LSTM'
    hidden_architecture = [
        {
            'hidden_dimension': 64,
            'activation': 'ReLU'
        },
        {
            'hidden_dimension': 32,
            'activation': 'ReLU'
        }
    ]

class RadarOptimizerCfg(OptimizerCfg):
    learning_rate = 1e-3
    epochs = 1000
    optimizer = 'Adam'
    criterion = 'MSE'

