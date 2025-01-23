import torch
import lightning

from .cfg import ModelCfg, OptimizerCfg

class RNNModel(lightning.LightningModule):
    def __init__(self, model_cfg: ModelCfg, optimizer_cfg: OptimizerCfg):
        super().__init__()
        # Cache the model config
        self.cfg = model_cfg

        '''
            Initialize the Decoder
        '''
        hidden_architecture = self.cfg.hidden_architecture
        layers = []

        # Add input layer 
        layers.append(torch.nn.Linear(self.cfg.hidden_dim, hidden_architecture[0]['hidden_dimension']))
        layers.append(self._get_activation(hidden_architecture[0]['activation']))
        
        # Add hidden layers
        for index, layer in enumerate(hidden_architecture):
            if index == len(hidden_architecture) - 1:
                break
            
            if 'pre_process' in layer:
                pre_process = getattr(torch.nn, layer['pre_process']) if 'pre_process' in layer or index == 0 else None
                if pre_process is not None:
                    layers.append(pre_process(hidden_architecture[index]['hidden_dimension']))

            dense_layer = torch.nn.Linear(layer['hidden_dimension'], hidden_architecture[index + 1]['hidden_dimension'])
            layers.append(dense_layer)
            
            post_process = self._get_activation(layer['activation']) if 'activation' in layer else None
            if post_process is not None:
                layers.append(post_process) 

        # Add output layer
        # layers.append(torch.nn.BatchNorm1d(hidden_architecture[-1]['hidden_dimension']))
        layers.append(torch.nn.Linear(hidden_architecture[-1]['hidden_dimension'], self.cfg.output_dim))

        # Initialize the decoder
        self.decoder = torch.nn.Sequential(*layers)

        '''
            Send the model to the device
        '''
        self.to(self.device)

        # Get optimizer
        self.optimizer = self._get_optimizer(optimizer_cfg)

        # Get criterion
        self.criterion = self._get_criterion(optimizer_cfg)

    def forward(self, x):
        return self.decoder(x)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return self.optimizer
    
    def _get_optimizer(self, optimizer_cfg):
        if optimizer_cfg.optimizer == "Adam":
            return torch.optim.Adam(self.parameters(), lr=optimizer_cfg.learning_rate)
        elif optimizer_cfg.optimizer == "SGD":
            return torch.optim.SGD(self.parameters(), lr=optimizer_cfg.learning_rate)
        else:
            raise NotImplementedError

    def _get_criterion(self, optimizer_cfg):
        loss_fcn = None
        
        if optimizer_cfg.criterion == 'BCE':
            loss_fcn = torch.nn.BCELoss()
        elif optimizer_cfg.criterion == 'MSE':
            loss_fcn = torch.nn.MSELoss()
        elif optimizer_cfg.criterion == 'BCEWithLogitsLoss':
            loss_fcn = torch.nn.BCEWithLogitsLoss()
        else:
            raise NotImplementedError('Criterion not implemented, Currently supported: BCE, MSE, BCEWithLogitsLoss') 
        
        return loss_fcn

    def _get_activation(self, act_name):
        if act_name == "elu":
            return torch.nn.ELU()
        elif act_name == "selu":
            return torch.nn.SELU()
        elif act_name == "relu":
            return torch.nn.ReLU()
        elif act_name == "crelu":
            return torch.nn.ReLU()
        elif act_name == "lrelu":
            return torch.nn.LeakyReLU()
        elif act_name == "tanh":
            return torch.nn.Tanh()
        elif act_name == "sigmoid":
            return torch.nn.Sigmoid()
        else:
            raise NotImplementedError