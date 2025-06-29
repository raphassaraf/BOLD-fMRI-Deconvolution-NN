import torch
import yaml

from torch.utils.data import random_split

from tools.simulated_data import generate_dataset
from tools.utils import FMRIDataset


with open('configs/config_sim_data.yaml', 'r') as file:
    CONFIG = yaml.safe_load(file)

REGIME_NAME = CONFIG['simulated_regime']['name']
REGIME = CONFIG['simulated_regime']['regime']
N_POINTS, N_VOXELS = CONFIG['n_points'], CONFIG['n_voxels']
METHOD = CONFIG['simulated_regime']['method']

n_data_suf = N_VOXELS // 1000
if n_data_suf >= 1000:
    n_data_suf = str(n_data_suf // 1000) + 'M'
else:
    n_data_suf = str(n_data_suf) + 'm'


def main():
    
    print(f'''
        Generating simulated dataset.
        Regime name = {REGIME_NAME}
        Regime parameters: {REGIME}
        HRF generation method: {METHOD}
        Size of the dataset (number of samples, number of time points): ({N_VOXELS}, {N_POINTS})
        ''')
        
    # Generate simulated data
    y, _, X = generate_dataset(N_POINTS, N_VOXELS, REGIME, METHOD)
        
    # Convert to torch tensors
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)
        
    # Instantiate FMRIDataset objects
    dataset = FMRIDataset(X, y)
        
    # Split the datasets into train/val/test sets
    total_size = len(dataset)
    train_len = int(0.7 * total_size)
    val_len = int(0.15 * total_size)
    test_len = total_size - train_len - val_len
    train_set, val_set, test_set = random_split(dataset, [train_len, val_len, test_len])
        
    # Store the split in a dictionnary
    dataset = {'train': train_set, 'validation': val_set, 'test': test_set}
        
    # Save the dataset
    torch.save(dataset, f'datasets/simulated_data_{REGIME_NAME}_{n_data_suf}.pt')

    print('Dataset saved !')

if __name__ == '__main__':
    
    main()