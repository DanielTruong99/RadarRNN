import torch

from learning.model.cfg import ModelCfg, OptimizerCfg
from learning.dataset.dataset_cfg import DatasetCfg

class RadarDatasetCfg(DatasetCfg):
    input_name_list = ["s_q4", "c_q4", "s_q5", "c_q5", "s_q6", "c_q6", "r1", "r2", "r3", "r4"]
    label_name_list = ["r1", "r2", "r3", "r4"]
    sequence_length = 10
    data_type = torch.float32
    batch_size = 256

class RadarDatasetsCfg:
    dataset_path: str = 'dataset/radar'
    train: RadarDatasetCfg = RadarDatasetCfg()
    val: RadarDatasetCfg = RadarDatasetCfg()
    test: RadarDatasetCfg = RadarDatasetCfg()


class RadarModelCfg(ModelCfg):
    input_dim = 10
    output_dim = 4
    hidden_dim = 64
    num_layers = 2
    rnn_type = 'GRU'
    hidden_architecture = [
        {
            'hidden_dimension': 128,
            'activation': 'relu'
        },
        {
            'hidden_dimension': 128,
            'activation': 'relu'
        },
        {
            'hidden_dimension': 128,
            'activation': 'relu'
        }
    ]

class RadarOptimizerCfg(OptimizerCfg):
    learning_rate = 1e-3
    epochs = 1000
    optimizer = 'Adam'
    criterion = 'MSE'

