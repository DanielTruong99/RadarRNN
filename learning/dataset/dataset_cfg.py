import torch
from typing import List, Optional, Dict, String

class DatasetCfg:
    sequence_length: int = 10
    is_shuffle: bool = True
    num_workers: int = 4
    input_name_list: List[str] = ["I1", "I2", "I3"]
    label_name_list: List[str] = ["L1", "L2"]

    # Case the input have the history of label
    # input_name_list: List[str] = ["I1", "I2", "I3", "L1", "L2"]
    # label_name_list: List[str] = ["L1", "L2"]   

    data_type: torch.dtype = torch.float32