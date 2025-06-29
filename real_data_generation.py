import torch
import yaml

from torch.utils.data import ConcatDataset

from tools.preprocessing_script import get_real_data_split

with open('configs/config_real_data.yaml', 'r') as file:
    CONFIG = yaml.safe_load(file)

TASKS = CONFIG['tasks']
AUGMENTATION_RATIOS = CONFIG['augmentation_ratios']
SIGNAL_LENGTH_DICT = {
    'motor': 284,
    'emotion': 176,
    'language': 316,
    'wm': 405,
    'gambling': 253
}

def main():

    final_length = max([SIGNAL_LENGTH_DICT[task] for task in TASKS])
    regime_name = ''.join([str.upper(task[0]) for task in TASKS])
    n_augm_suf = str(AUGMENTATION_RATIOS).translate({ord(el): None for el in '[],. '})

    print(f'''
        Generating real dataset.
        Tasks = {TASKS} ({regime_name})
        Augmentation ratios: {AUGMENTATION_RATIOS} ({n_augm_suf})
        ''')

    datasets = [
        get_real_data_split(task, augmentation=True, augmentation_ratios=AUGMENTATION_RATIOS, final_length=final_length)
        for task in TASKS
    ]

    real_data = {}
    for key in ['train', 'validation', 'test']:
        dataset_split = [ds[key] for ds in datasets]
        real_data[key] = ConcatDataset(dataset_split)

    torch.save(real_data, f'datasets/real_data_{regime_name}_{n_augm_suf}.pt')
    
    print('Dataset saved !')
    
if __name__ == '__main__':
    
    main()