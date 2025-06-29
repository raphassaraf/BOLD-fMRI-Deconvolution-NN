import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import torch.nn.functional as F
import xarray as xr

from tqdm import tqdm

def preprocess_subject_data(
    subject_data, normalize=True
):  # converts data to tensor for a single subject

    X_data, y_data = (
        subject_data["X"].values,
        subject_data["Y"].values,
    )  # Shape: [voxels, time]

    no_nan_mask = ~np.isnan(X_data).any(axis=1)
    X_data, y_data = X_data[no_nan_mask], y_data[no_nan_mask]

    # Normalization
    if normalize:
        X_data = (X_data - np.mean(X_data, axis=1, keepdims=True)) / (
            np.std(X_data, axis=1, keepdims=True) + 1e-8
        )

    # Conversion to torch tensor
    X_tensor, y_tensor = torch.tensor(X_data, dtype=torch.float32), torch.tensor(
        y_data, dtype=torch.float32
    )

    return X_tensor, y_tensor


def get_data_tensors(
    dataset,
    subject_ids=None,
    normalize=True,
    augmentation=True,
    shift_range=(-20, 20),
    amplitude_range=(0.75, 1.25),
    noise_snr_db_range=(-1, 5),
    ratios=(0.3, 0.3, 0.3),
    for_test=False
):

    # If no subjects are specified, use all of them
    if subject_ids is None:
        subject_ids = dataset.subject.values

    if for_test: # No splitting, all the data is preprocessed as a test set
        print("Preprocessing data as test set:")
        X_test, y_test = [], []
        for subject in tqdm(subject_ids):

            subject_data = dataset.sel(subject=subject)
            X, y = preprocess_subject_data(subject_data, normalize)
            X_test.append(X)
            y_test.append(y)

        X_test = torch.cat(X_test, dim=0).unsqueeze(-1)
        y_test = torch.cat(y_test, dim=0).unsqueeze(-1)
        
        return X_test, y_test
            
    else: # Data is split into train / validation / test set
        # Shuffle subject IDs
        shuffled_subjects = subject_ids.copy()
        random.shuffle(shuffled_subjects)
        #print(shuffled_subjects)

        # Train, validation, test split
        total_subjects = len(shuffled_subjects)
        train_end, val_end = int(0.7 * total_subjects), int(0.85 * total_subjects)
        train_subjects = shuffled_subjects[:train_end]
        val_subjects = shuffled_subjects[train_end:val_end]
        test_subjects = shuffled_subjects[val_end:]

        # Preprocess train set
        print(f"Preprocessing train set with augmentation set to {augmentation}:")
        X_train, y_train = [], []
        for subject in tqdm(train_subjects):

            subject_data = dataset.sel(subject=subject)
            X, y = preprocess_subject_data(subject_data, normalize)
            X_train.append(X)
            y_train.append(y)

        X_train = torch.cat(X_train, dim=0).unsqueeze(-1)
        y_train = torch.cat(y_train, dim=0).unsqueeze(-1)

        if augmentation:
            X_augmented, y_augmented = augment(
                X_train, y_train, shift_range, amplitude_range, noise_snr_db_range, ratios
            )
            X_train = torch.cat([X_train, X_augmented], dim=0)
            y_train = torch.cat([y_train, y_augmented], dim=0)

        # Preprocess validation set
        print("Preprocessing validation set:")
        X_val, y_val = [], []
        for subject in tqdm(val_subjects):

            subject_data = dataset.sel(subject=subject)
            X, y = preprocess_subject_data(subject_data, normalize)
            X_val.append(X)
            y_val.append(y)

        X_val = torch.cat(X_val, dim=0).unsqueeze(-1)
        y_val = torch.cat(y_val, dim=0).unsqueeze(-1)

        # Preprocess test set
        print("Preprocessing test set:")
        X_test, y_test = [], []
        for subject in tqdm(test_subjects):

            subject_data = dataset.sel(subject=subject)
            X, y = preprocess_subject_data(subject_data, normalize)
            X_test.append(X)
            y_test.append(y)

        X_test = torch.cat(X_test, dim=0).unsqueeze(-1)
        y_test = torch.cat(y_test, dim=0).unsqueeze(-1)

        return X_train, y_train, X_val, y_val, X_test, y_test

def concat_task_datasets(task_datasets):
    
    output_tensor = []
    
    max_signal_lenght = max([dataset.shape[1] for dataset in task_datasets])
    for dataset in task_datasets:
        
        total_padding = max_signal_lenght - dataset.shape[1]
        left_padding = (max_signal_lenght - dataset.shape[1]) // 2
        right_padding = total_padding - left_padding
        
        output_tensor.append(F.pad(dataset, (0, 0, left_padding, right_padding), mode='constant'))
        
    output_tensor = torch.cat(output_tensor, dim=0)
    
    return output_tensor


### DATA AUGMENTATION FUNCTIONS ###
def shift(X, y, shift_range=(-20, 20), ratio=0.3):

    dataset_size, signal_length, _ = X.shape
    n_shifted_samples = int(dataset_size * ratio)

    # Select n_shifted_samples random indices
    shift_idx = torch.tensor(
        np.random.choice(dataset_size, size=n_shifted_samples, replace=False)
    )

    # Generate random shifts for each selected index
    shift_list = np.random.randint(
        shift_range[0], shift_range[1] + 1, (n_shifted_samples,)
    )

    # Apply generated shifts to selected indices
    shifted_X, shifted_y = X.clone(), y.clone()
    for idx, shift in zip(shift_idx, shift_list):
        shifted_X[idx] = torch.roll(shifted_X[idx], shift, dims=0)
        shifted_y[idx] = torch.roll(shifted_y[idx], shift, dims=0)

    return shifted_X, shifted_y, shift_idx


def scale_amplitude(X, y, amplitude_range=(0.75, 1.25), ratio=0.3):

    dataset_size = X.shape[0]
    n_scaled_samples = int(dataset_size * ratio)

    # Select n_scaled_samples random indices
    scaled_idx = torch.tensor(
        np.random.choice(dataset_size, size=n_scaled_samples, replace=False)
    )

    # Generate random scalings for each selected index
    scaling_list = np.random.uniform(
        amplitude_range[0], amplitude_range[1], (n_scaled_samples,)
    )

    # Apply generated scalings to selected indices
    scaled_X, scaled_y = X.clone(), y.clone()
    for idx, scale_factor in zip(scaled_idx, scaling_list):
        scaled_X[idx] = scaled_X[idx] * scale_factor
        scaled_y[idx] = scaled_y[idx] * scale_factor

    return scaled_X, scaled_y, scaled_idx


def add_gaussian_noise(X, y, noise_snr_db_range=(-1, 5), ratio=0.3):

    dataset_size = X.shape[0]
    n_noisy_samples = int(dataset_size * ratio)

    # Select n_noisy_samples random indices
    noisy_idx = torch.tensor(
        np.random.choice(dataset_size, size=n_noisy_samples, replace=False)
    )

    # Generate random desired SNR for the selected index
    noise_snr_db_list = torch.tensor(
        np.random.uniform(
            noise_snr_db_range[0], noise_snr_db_range[1], (n_noisy_samples, 1)
        )
    ).to(torch.float32)

    # Calculate gaussian noise
    noisy_X = X.clone()
    std_signal = noisy_X[noisy_idx].std(dim=1)
    std_noise = std_signal / np.sqrt(10 ** (noise_snr_db_list / 10))
    noise = (
        torch.randn(size=(noisy_X[noisy_idx].shape[0], noisy_X[noisy_idx].shape[1]))
        * std_noise
    )

    # Add noise to the selected samples
    noisy_X[noisy_idx] = noisy_X[noisy_idx] + noise.unsqueeze(-1)

    return noisy_X, y, noisy_idx


def augment(
    X,
    y,
    shift_range=(-20, 20),
    amplitude_range=(0.75, 1.25),
    noise_snr_db_range=(-1, 5),
    ratios=(0.3, 0.3, 0.3),
):

    augmented_X, augmented_y, shift_idx = shift(X, y, shift_range, ratios[0])
    augmented_X, augmented_y, scaled_idx = scale_amplitude(
        augmented_X, augmented_y, amplitude_range, ratios[1]
    )
    augmented_X, augmented_y, noisy_idx = add_gaussian_noise(
        augmented_X, augmented_y, noise_snr_db_range, ratios[2]
    )

    augmented_idx = torch.concat([shift_idx, scaled_idx, noisy_idx]).unique()

    # Only return augmented samples
    return augmented_X[augmented_idx], augmented_y[augmented_idx]
