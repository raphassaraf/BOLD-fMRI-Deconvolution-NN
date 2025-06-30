import random
import torch
import xarray as xr

from tools.preprocessing_functions import get_data_tensors, concat_task_datasets
from tools.utils import FMRIDataset

# The following are the directories to the fMRI datasets. 
DATA_DIR = '/media/RCPNAS/Data2/Flavia/CS-433-ML4S/dataset/'
TASK_FILES = {
    'motor': DATA_DIR + 'dataset_MOTOR_100_subjects_smoothing5mm_98thquantile_not_normalized_remove_mean_regressor_min_duration_10_rsquared.nc',
    'emotion': DATA_DIR + 'dataset_EMOTION_100_subjects_smoothing5mm_98thquantile_not_normalized_remove_mean_regressor_min_duration_10_rsquared.nc',
    'gambling': DATA_DIR + 'dataset_GAMBLING_100_subjects_smoothing5mm_98thquantile_not_normalized_remove_mean_regressor_min_duration_10_rsquared.nc',
    'wm': DATA_DIR + 'dataset_WM_30_subjects_smoothing5mm_98thquantile_not_normalized_merged_shortseq.nc',
    'language': DATA_DIR + 'dataset_LANGUAGE_100_subjects_smoothing5mm_98thquantile_not_normalized_remove_mean_regressor_min_duration_10_rsquared.nc'
}

def get_real_data_split(task, augmentation=True, augmentation_ratios=(0.3, 0.3, 0.3), for_test=False, final_length=None, seed=0):
    '''
    Wraps a real fMRI dataset into a FMRIDataset object with train/val/test split.

    task: 'motor', 'emotion', 'gambling', 'wm', 'language'.
    augmentation_ratios: tuble of floats, proportions of the dataset to apply the each individual augmentation to (shift, amplitude, noise).
    for_test: bool, True if the whole dataset should be used for testing (no split would thus be applied).
    final_length: int, the final length to pad the signals to.
    '''
    
    random.seed(seed)

    path = TASK_FILES[task]
    dataset = xr.open_dataset(path)
    
    data_splits = get_data_tensors(
        dataset,
        augmentation=augmentation,
        ratios=augmentation_ratios,
        for_test=for_test
    )

    if for_test:
        X_test, y_test = data_splits
        dataset = {'test': FMRIDataset(X_test, y_test, final_length=final_length)}
    else:
        X_train, y_train, X_val, y_val, X_test, y_test = data_splits
        dataset = {
            'train': FMRIDataset(X_train, y_train, final_length=final_length),
            'validation': FMRIDataset(X_val, y_val, final_length=final_length),
            'test': FMRIDataset(X_test, y_test, final_length=final_length)
        }

    return dataset

def preprocessing(seed=0):
    '''
    The full preprocessing code.
    '''

    random.seed(seed)
    ## Import all datasets as xarrays.
    dataset_list = [xr.open_dataset(path) for path in TASK_FILES.values()]
    dataset_motor, dataset_emotion, dataset_gambling, dataset_wm, dataset_language = dataset_list

    ## Preprocess the datasets for each individual task.
    # motor task
    print('\nPreprocessing motor task with train/val/test split:')

    (
        X_train_motor, y_train_motor, X_val_motor, y_val_motor, X_test_motor, y_test_motor
    ) = get_data_tensors(dataset_motor, augmentation=True)
    print('Train shape (input, target): ', X_train_motor.shape, y_train_motor.shape)
    print('Validation shape (input, target): ', X_val_motor.shape, y_val_motor.shape)
    print('Test shape (input, target): ', X_test_motor.shape, y_test_motor.shape)

    # emotion task
    print('\nPreprocessing emotion task with train/val/test split:')
    (
        X_train_emotion, y_train_emotion, X_val_emotion, y_val_emotion, X_test_emotion, y_test_emotion
    ) = get_data_tensors(dataset_emotion, augmentation=True)
    print('Train shape (input, target): ', X_train_emotion.shape, y_train_emotion.shape)
    print('Validation shape (input, target): ', X_val_emotion.shape, y_val_emotion.shape)
    print('Test shape (input, target): ', X_test_emotion.shape, y_test_emotion.shape)

    # working memory task
    print('\nPreprocessing working memory task with train/val/test split:')
    (
        X_train_wm, y_train_wm, X_val_wm, y_val_wm, X_test_wm, y_test_wm
    ) = get_data_tensors(dataset_wm, augmentation=True)
    print('Train shape (input, target): ', X_train_wm.shape, y_train_wm.shape)
    print('Validation shape (input, target): ', X_val_wm.shape, y_val_wm.shape)
    print('Test shape (input, target): ', X_test_wm.shape, y_test_wm.shape)

    # gambling task, taken as our OOD (Out-of-Distribution) data
    print('\nPreprocessing gambling task only as a test set')
    X_test_gambling, y_test_gambling = get_data_tensors(dataset_gambling, for_test=True)
    print('Test shape (input, target): ', X_test_gambling.shape, y_test_gambling.shape)

    # language task, taken as our OOD data
    print('\nPreprocessing language task only as a test set')
    X_test_language, y_test_language = get_data_tensors(dataset_language, for_test=True)
    print('Test shape (input, target): ', X_test_language.shape, y_test_language.shape)


    ## Concatenate the task-corresponding train, validation and test sets.
    X_train = concat_task_datasets([X_train_motor, X_train_emotion, X_train_wm])
    y_train = concat_task_datasets([y_train_motor, y_train_emotion, y_train_wm])

    X_val = concat_task_datasets([X_val_motor, X_val_emotion, X_val_wm])
    y_val = concat_task_datasets([y_val_motor, y_val_emotion, y_val_wm])

    X_test = concat_task_datasets([X_test_motor, X_test_emotion, X_test_wm])
    y_test = concat_task_datasets([y_test_motor, y_test_emotion, y_test_wm])

    X_ood = concat_task_datasets([X_test_gambling, X_test_language])
    y_ood = concat_task_datasets([y_test_gambling, y_test_language])

    print('\nConcatenating task data by using 0-padding on different lenght signals:')
    print('Train shape (input, target): ', X_train.shape, y_train.shape)
    print('Validation shape (input, target): ', X_val.shape, y_val.shape)
    print('Test shape (input, target): ', X_test.shape, y_test.shape)
    print('Test OOD shape (input, target): ', X_ood.shape, y_ood.shape)
    
    return X_train, y_train, X_val, y_val, X_test, y_test, X_ood, y_ood