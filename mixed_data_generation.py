import numpy as np
import torch
import yaml

from torch.utils.data import ConcatDataset, random_split
from tqdm import tqdm

from tools.simulated_data import generate_dataset
from tools.utils import FMRIDataset

from tools.preprocessing_script import get_real_data_split

with open('configs/config_mixed_data.yaml', 'r') as file:
    CONFIG = yaml.safe_load(file)

REAL_DATA_PARAMS = CONFIG['real']
SIM_DATA_PARAMS = CONFIG['simulated']
N_SAMPLES_PER_SUBSET = CONFIG['n_samples']

SIGNAL_LENGTH_DICT = {
    'motor': 284,
    'emotion': 176,
    'language': 316,
    'wm': 405,
    'gambling': 253
}

def main():

    prop_real = [0.0, 0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.97, 0.99, 1.0]
    print(f'\n#### prop_real = {prop_real} ####')


    n_tasks = len(REAL_DATA_PARAMS['tasks'])
    final_length = max([SIGNAL_LENGTH_DICT[task] for task in REAL_DATA_PARAMS['tasks']])
    print(f'\n#### Signal lengths = {final_length} ####')

    # Get real data
    print(f'\n#### Load real data ####')
    real_data_list = [
        get_real_data_split(task, final_length=final_length, augmentation_ratios=REAL_DATA_PARAMS['augmentation_ratios'])
        for task in REAL_DATA_PARAMS['tasks']
    ]

    # Get simulated data
    print('\n#### Load simulated data ####')
    y, _, X = generate_dataset(final_length, SIM_DATA_PARAMS['n_voxels'], SIM_DATA_PARAMS['regime'], SIM_DATA_PARAMS['method'])
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)
    sim_data = FMRIDataset(X, y)
    total_size = len(sim_data)
    train_len = int(0.7 * total_size)
    val_len = int(0.15 * total_size)
    test_len = total_size - train_len - val_len
    train_set, val_set, test_set = random_split(sim_data, [train_len, val_len, test_len])
    sim_data = {'train': train_set, 'validation': val_set, 'test': test_set}

    # Mix the datasets
    print('\n#### Mix the datasets ####')
    dataset_dict = {}
    for p in tqdm(prop_real):

        mixed_data = {}
        for key in ['train', 'validation', 'test']:
            subset_list = []

            if p != 1:
                n_simulated_samples = len(sim_data[key])
                idx_simulated_samples = np.random.choice(n_simulated_samples, size=int((1-p)*N_SAMPLES_PER_SUBSET[key]), replace=False).tolist()
                subset_list.append(sim_data[key][idx_simulated_samples])
            if p != 0:
                for real_data in real_data_list:
                    n_real_samples = len(real_data[key])
                    idx_real_samples = np.random.choice(n_real_samples, size=int(p*N_SAMPLES_PER_SUBSET[key]//n_tasks), replace=False).tolist()
                    subset_list.append(real_data[key][idx_real_samples])

            mixed_data[key] = ConcatDataset(subset_list)
        dataset_dict[f"{p:.1f}"] = mixed_data

    # Save all the datasets in a dictionnary
    torch.save(dataset_dict, 'datasets/mixed_datasets_900m.pt')


if __name__ == '__main__':
    
    main()