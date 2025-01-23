import torch 
import pandas as pd
import os

from torch.utils.data import Dataset
from .dataset_cfg import DatasetCfg

class CSVDataset(Dataset):
    def __init__(self, csv_file, dataset_cfg: DatasetCfg):
        file_name = os.listdir(csv_file)
        if len(file_name) > 1:
            raise ValueError('The directory should contain only one csv file')
        
        #! Assume the data was stored based on time series

        # Load the csv file and cache the dataset config
        csv_file_path = os.path.join(csv_file, file_name[0])
        self.data_frame = pd.read_csv(csv_file_path)
        self.cfg = dataset_cfg
        self.num_features = len(self.cfg.input_name_list)
        self.num_labels = len(self.cfg.label_name_list)

        # Get the input and label data and convert to numpy
        self.input = self.data_frame[self.cfg.input_name_list].values
        self.label = self.data_frame[self.cfg.label_name_list].values

        
    def __len__(self):
        return len(self.input) - self.cfg.sequence_length

    def __getitem__(self, index):
        '''
            Example: index = 2; sequence_length = 3
            sample = (input, label)
            input = [
                [I1, I2, I3],  row_index = 2
                [I1, I2, I3],  row_index = 3
                [I1, I2, I3]   row_index = 4
            ]
            label = [
                [L1, L2], row_index = 5
            ]
        '''

        sample = (
            torch.tensor(self.input[index], dtype=self.cfg.data_type),
            torch.tensor(self.label[index], dtype=self.cfg.data_type),
        )
        return sample