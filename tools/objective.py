import csv
import os
import torch
import torch.nn as nn
import yaml

from datetime import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau

from tools.custom_loss import me_tv_loss, blocky_loss
from tools.models.autoencoder_models import AutoEncoder, LatentDeconvolution
from tools.models.cnn_models import CNN_1l, CNN_2l, CNN_3l, CNN_4l
from tools.models.lstm_models import LSTM_nl, RNNDeconvolutionRNN
from tools.training_functions import train, train_autoencoder
from tools.utils import get_number_of_parameters, MODEL_DICT

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CSV_FOLDER = 'csv_logs/'
INPUT_SIZE = 1

with open("configs/config_objective.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)

TRAINING_PARAMS = CONFIG['training_params']

class BaseObjective:
    def __init__(self, study_name, train_loader, val_loader, num_epochs, blocky, log_file='cnn_logs.csv', test_loader=None):

        self.study_name = study_name
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.blocky = blocky
        self.log_file = log_file
        self.test_loader = test_loader

        self.initial_lr, self.patience, self.tolerance, self.scheduler_pr = TRAINING_PARAMS.values()
    
    def __call__(self, trial):

        # Get trial parameters
        params = self.suggest_hyperparameters(trial)
        criterion, criterion_name = self.suggest_criterion(trial)

        # Instantiate model, optimizer, scheduler
        model = self.model_class(**params).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.initial_lr)
        scheduler = ReduceLROnPlateau(optimizer, **self.scheduler_pr)

        # String for output files
        model_name = self.get_model_name(params, criterion_name)

        # Run training loop    
        output = train(
            model, self.train_loader, self.val_loader, criterion, optimizer,
            scheduler, DEVICE, self.num_epochs, model_name, self.study_name,
            patience=self.patience, tolerance=self.tolerance, test_loader=self.test_loader
        )

        # Gather trial information to be written in master_log file
        csv_data = self.get_csv_data(model, model_name, output)
        self.write_training_results(csv_data, os.path.join(CSV_FOLDER, self.log_file))

        min_mse_val = min(output['mse_val'])
        
        return min_mse_val
    
    def suggest_hyperparameters(self, trial):

        params = {
            hp: trial.suggest_int(**kwargs) for hp, kwargs in self.trial_pr.items()
        }
        
        return params
    
    def suggest_criterion(self, trial):
        '''
        Helper to suggest criterion based on whether the search is done
        for blocky or non-blocky loss. 
        MAE loss is removed from this new version, and the alpha parameter is re-adjusted to higher values.
        '''
        
        me_criterion = nn.MSELoss()
        criterion_name = 'MSE_TV'
            
        if self.blocky:
            # alpha = trial.suggest_float('alpha', 10, 100)
            # lambda_tv = trial.suggest_float('lambda_tv', 1e-7, 1e-1, log=True)

            #### Oversampling correction ####
            alpha = trial.suggest_float('alpha', 30, 30)
            lambda_tv = trial.suggest_float('lambda_tv', 1e-4, 1e-4, log=True)
            criterion_name += f'_(a={alpha}, l={lambda_tv})'
            criterion = blocky_loss(alpha, lambda_tv, mean_error_loss=me_criterion)
        else:
            lambda_tv = trial.suggest_float('lambda_tv', 1e-5, 1e2, log=True)
            criterion_name += f'_(l={lambda_tv})'
            criterion = me_tv_loss(lambda_tv, me_criterion)
        
        return criterion, criterion_name
    
    def get_csv_data(self, model, model_name, output):

        csv_data = {
            'study_name': self.study_name,
            'timecode': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'model_name': model_name,
            'n_params': get_number_of_parameters(model),
            'train_size': len(self.train_loader.dataset),
            'validation_size': len(self.val_loader.dataset),
            'test_size': len(self.test_loader.dataset),
            'min_mse_val': min(output['mse_val']),
            'test_mse': output['mse_test'],
            'test_mse_var': output['mse_var_test'],
            'lr': output['lr'],
            'batch_size': self.train_loader.batch_size,
            'patience': self.patience,
            'tolerance': self.tolerance,
            'architecture': str(model)
        }

        return csv_data

    def write_training_results(self, results_dict, path):
        with open(path, mode="a") as file:
            writer = csv.DictWriter(file, fieldnames=list(results_dict.keys()))
            writer.writerows([results_dict])
    
    def get_model_name(self, params, criterion_name):
        raise NotImplementedError("Override this method to define hyperparameter search space.")


class CNN_1l_Objective(BaseObjective):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model_class = CNN_1l
        self.trial_pr = CONFIG["trial_params_CNN_1l"]
    
    def get_model_name(self, params, criterion_name):

        model_name = 'CNN_1l_'
        k = params['kernel_size']
        params_name = f'kernel={k}_criterion={criterion_name}_'
        timecode = str(datetime.now().strftime('%Y%m%d_%H%M%S'))
        model_name += params_name + timecode
        
        return model_name
    

class CNN_2l_Objective(BaseObjective):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model_class = CNN_2l
        self.trial_pr = CONFIG["trial_params_CNN_2l"]

    def get_model_name(self, params, criterion_name):

        model_name = 'CNN_2l_'
        k1, k2, h = params.values()
        params_name = f'kernel1={k1}_kernel2={k2}_hidden={h}_criterion={criterion_name}_'
        timecode = str(datetime.now().strftime('%Y%m%d_%H%M%S'))
        model_name += params_name + timecode

        return model_name
    

class CNN_3l_Objective(BaseObjective):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model_class = CNN_3l
        self.trial_pr = CONFIG["trial_params_CNN_3l"]
    
    def get_model_name(self, params, criterion_name):

        model_name = 'CNN_3l_'
        k1, k2, k3, h1, h2 = params.values()
        params_name = f'kernel1={k1}_kernel2={k2}_kernel3={k3}_hidden1={h1}_hidden2={h2}_criterion={criterion_name}_'
        timecode = str(datetime.now().strftime('%Y%m%d_%H%M%S'))
        model_name += params_name + timecode
        
        return model_name
    
    
class CNN_4l_Objective(BaseObjective):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model_class = CNN_4l
        self.trial_pr = CONFIG["trial_params_CNN_4l"]
    
    def get_model_name(self, params, criterion_name):

        model_name = 'CNN_4l_'
        k1, k2, k3, k4, h1, h2, h3 = params.values()
        params_name = f'kernel1={k1}_kernel2={k2}_kernel3={k3}_kernel4={k4}_hidden1={h1}_hidden2={h2}_hidden3={h3}_criterion={criterion_name}_'
        timecode = str(datetime.now().strftime('%Y%m%d_%H%M%S'))
        model_name += params_name + timecode
        
        return model_name
    

class LSTM_Objective(BaseObjective):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model_class = LSTM_nl
        self.trial_pr = CONFIG["trial_params_LSTM"]
    
    def get_model_name(self, params, criterion_name):

        model_name = f'LSTM_'
        num_layers, hidden_size = params.values()
        params_name = f'{num_layers}l_hidden={hidden_size}_criterion={criterion_name}_'
        timecode = str(datetime.now().strftime('%Y%m%d_%H%M%S'))
        model_name += params_name + timecode

        return model_name
        

class RNNDeconvolutionRNN_Objective(BaseObjective):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model_class = RNNDeconvolutionRNN
        self.trial_pr = CONFIG["trial_params_RNNDeconvolutionRNN"]
    
    def get_model_name(self, params, criterion_name):

        model_name = f'LSTMDeconvLSTM_'
        num_layers_kernel_rnn, hidden_size_kernel_rnn, num_layers_output_rnn, hidden_size_output_rnn, kernel_size = params.values()
        params_name = f'{num_layers_kernel_rnn}lK_hiddenK={hidden_size_kernel_rnn}_{num_layers_output_rnn}lO_hiddenO={hidden_size_output_rnn}_kernel={kernel_size}_criterion={criterion_name}_'
        timecode = str(datetime.now().strftime('%Y%m%d_%H%M%S'))
        model_name += params_name + timecode

        return model_name

class AutoEncoder_Objective(BaseObjective):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.ae_objective_pr = CONFIG['ae_objective_params']

        self.trial_pr = self.ae_objective_pr['trial_params_AutoEncoder']
        self.ldm_type = MODEL_DICT[self.ae_objective_pr['ldm_type']]
        self.ldm_pr = self.ae_objective_pr['ldm_params']
        self.criterion_ldm_pr = self.ae_objective_pr['criterion_ldm_pr']

        self.autoencoder = AutoEncoder
        self.latent_deconv_model = LatentDeconvolution
    
    def __call__(self, trial):

        # Get trial parameters (i.e. for AutoEncoder)
        params_ae = self.suggest_hyperparameters(trial)
        compression = params_ae['compression']

        # Instantiate AutoEncoder along with its criterion and optimizer
        signal_length = self.train_loader.dataset[0][0].shape[0]
        autoencoder = self.autoencoder(signal_length, **params_ae).to(DEVICE)
        criterion_ae = nn.MSELoss()
        optimizer_ae = torch.optim.Adam(autoencoder.parameters(), lr=self.initial_lr)
        scheduler_ae = ReduceLROnPlateau(optimizer_ae, **self.scheduler_pr)

        # Instantiate LatentDeconvolution model along with its criterion and optimizer
        latent_deconv_model = self.latent_deconv_model(
            signal_length,
            compression,
            self.ldm_type(**self.ldm_pr)
        ).to(DEVICE)
        criterion_ldm = blocky_loss(**self.criterion_ldm_pr)
        optimizer_ldm = torch.optim.Adam(latent_deconv_model.parameters(), lr=self.initial_lr)
        scheduler_ldm = ReduceLROnPlateau(optimizer_ldm, **self.scheduler_pr)

        # String for output files
        model_name = self.get_model_name(params_ae)

        # Run training loop
        final_model, output = train_autoencoder(
            autoencoder, latent_deconv_model, self.train_loader, self.val_loader,
            criterion_ae, criterion_ldm, optimizer_ae, optimizer_ldm, scheduler_ae,
            scheduler_ldm, DEVICE, self.num_epochs, model_name, self.study_name,
            patience=self.patience, tolerance=self.tolerance, test_loader=self.test_loader
            )
        
        # Gather trial information to be written in master_log file
        csv_data = self.get_csv_data(final_model, model_name, output)
        self.write_training_results(csv_data, os.path.join(CSV_FOLDER, self.log_file))

        min_mse_val = min(output['mse_val']['ldm']) # We use the validation loss of the LatentDeconvolution model
        
        return min_mse_val
    
    def suggest_hyperparameters(self, trial):

        params = {
            hp: trial.suggest_int(**kwargs) if hp != 'dropout' else trial.suggest_float(**kwargs)
            for hp, kwargs in self.trial_pr.items()
        }
        
        return params
    
    def get_csv_data(self, model, model_name, output):

        csv_data = {
            'study_name': self.study_name,
            'timecode': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'model_name': model_name,
            'n_params': get_number_of_parameters(model),
            'train_size': len(self.train_loader.dataset),
            'validation_size': len(self.val_loader.dataset),
            'test_size': len(self.test_loader.dataset),
            'min_mse_val': min(output['mse_val']['ldm']),
            'test_mse': output['mse_test']['ldm'],
            'test_mse_var': output['mse_var_test']['ldm'],
            'lr': output['lr'],
            'batch_size': self.train_loader.batch_size,
            'patience': self.patience,
            'tolerance': self.tolerance,
            'architecture': str(model)
        }

        return csv_data
    
    def get_model_name(self, params_ae):

        model_name = f'AutoEncoder_'

        params_ae_name = '_'.join(str.upper(key[0]) + '=' + str(value) for key, value in params_ae.items())

        params_ldm_name = '_' + self.ae_objective_pr['ldm_type'] + '('
        params_ldm_name += '_'.join(str.upper(key[0]) + key[-1] + '=' + str(value) for key, value in self.ldm_pr.items())
        params_ldm_name += '_criterion=MSE_TV_('
        params_ldm_name += '_'.join(str(key[0]) + '=' + str(value) for key, value in self.criterion_ldm_pr.items())
        params_ldm_name += '))_'

        timecode = str(datetime.now().strftime('%Y%m%d_%H%M%S'))

        model_name += params_ae_name + params_ldm_name + timecode

        return model_name

