import optuna
import os
import torch
import yaml

from datetime import datetime
from torch.utils.data import DataLoader

from tools.objective import *
from tools.utils import pickle_dump

with open('configs/config_hp_search.yaml', 'r') as file:
    CONFIG = yaml.safe_load(file)

OBJECTIVE_FN_DICT = {
   'CNN_1L': CNN_1l_Objective,
   'CNN_2L': CNN_2l_Objective,
   'CNN_3L': CNN_3l_Objective,
   'CNN_4L': CNN_4l_Objective,
   'LSTM': LSTM_Objective,
   'LSTMDeconvolutionLSTM': RNNDeconvolutionRNN_Objective,
   'AutoEncoder': AutoEncoder_Objective
}

MODEL_TYPES = CONFIG['model_types']
NUM_TRIALS = CONFIG['num_trials']

N_EPOCHS, BATCH_SIZE = CONFIG['num_epochs'], CONFIG['batch_size']

REAL = CONFIG['real']
REGIMES = CONFIG['regimes']
DATA_SUF = CONFIG['data_suf']

BLOCKY = CONFIG['blocky']

LOG_FILE = CONFIG['output_file']

def main():

    print(f'''
        Starting hyper-paramter search for {MODEL_TYPES} models with {NUM_TRIALS} trials
        Dataset suffix: {DATA_SUF}
        Regimes of the datasets: {REGIMES}
        Number of epochs per training: {N_EPOCHS} | Batch size: {BATCH_SIZE}
        Output file: {LOG_FILE}
        Blocky: {BLOCKY}

        ''')
        
    for model_type, n_trials in zip(MODEL_TYPES, NUM_TRIALS):
        for regime in REGIMES:
            for blocky in BLOCKY:

                print(f'\n##### STARTING HP SEARCH ON {str.upper(regime)} DATASET | BLOCKY = {blocky} #####\n')

                # Load and preprocess data
                if REAL:
                    dataset = torch.load(f'datasets/real_data_{regime}_{DATA_SUF}.pt')
                else:
                     dataset = torch.load(f'datasets/simulated_data_{regime}_{DATA_SUF}.pt')

                    
                train_loader = DataLoader(dataset['train'], batch_size=BATCH_SIZE, shuffle=True)
                val_loader = DataLoader(dataset['validation'], batch_size=BATCH_SIZE, shuffle=True)
                test_loader = DataLoader(dataset['test'], batch_size=BATCH_SIZE, shuffle=True)

                # Create study path name
                if REAL:
                    study_name = f"{model_type}_RealData{str.upper(regime)}_blocky={blocky}"
                else:
                    study_name = f"{model_type}_SimData{str.upper(regime)}_blocky={blocky}"
                timecode = "_" + str(datetime.now().strftime('%Y%m%d_%H%M%S'))
                study_name += timecode

                # Create the directories where models and outputs will be stored
                os.makedirs(f"model_logs/{study_name}") 
                os.makedirs(f"training_outputs/{study_name}") 

                # Hyperparameter search
                study = optuna.create_study(direction='minimize', study_name=study_name)
                objective_fn = OBJECTIVE_FN_DICT[model_type](study_name, train_loader, val_loader, N_EPOCHS, blocky, LOG_FILE, test_loader)
                study.optimize(lambda trial: objective_fn(trial), n_trials=n_trials)

                # Save the Study object
                pickle_dump(study, f"optuna_logs/{study_name}")
    
if __name__ == '__main__':
    
    main()
