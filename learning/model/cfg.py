from typing import List, Dict

class ModelCfg:
    input_dim: int = 3
    output_dim: int = 2

    # Decoder architecture
    architecture: List[Dict] = [
        {"hidden_dimension": 256, "activation": "elu", "pre_process": "BatchNorm1d"}, 
        {"hidden_dimension": 256, "activation": "elu", "pre_process": "BatchNorm1d"}, 
        {"hidden_dimension": 256, "activation": "elu", "pre_process": "BatchNorm1d"}, 
    ]

    # Encoder architecture
    rnn_type: str = "LSTM"
    hidden_dim: int = 150
    num_layers: int = 1

class OptimizerCfg:
    learning_rate: float = 1e-3
    epochs: int = 1000
    optimizer: str = "Adam"
    criterion: str = "MSE"