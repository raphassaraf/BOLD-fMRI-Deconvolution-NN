import numpy as np
import torch

from nilearn.glm.first_level import spm_hrf
from nipy.modalities.fmri.hrf import spm_hrf_compat
from scipy.signal import convolve
from skimage.morphology import opening, closing
from tqdm import tqdm


def generate_timeseries(n_points, regime):
    
    min_onset_interval = regime['min_onset_interval']
    duration_mean, duration_scale = regime['duration'][0], regime['duration'][1]
    amplitude_mean, amplitude_scale = regime['amplitude'][0], regime['amplitude'][1]
    
    time_series = np.zeros(n_points)
    current_time = 0
    
    while current_time < n_points:
        # onset_interval = max(min_onset_interval, np.random.exponential(scale=1/lambda_onset))
        # current_time += int(onset_interval)
       
        if current_time >= n_points:
            break
        
        duration = int(max(5, np.round(np.random.normal(loc=duration_mean, scale=duration_scale))))
        amplitude = np.clip(np.random.normal(loc=amplitude_mean, scale=amplitude_scale), -2.6, 2.6)
        end_time = min(current_time + duration, n_points)
        
        time_series[current_time:end_time] = amplitude
        current_time = max(current_time, current_time + duration - min_onset_interval)
    
    morphology_struct = np.ones((9,))

    return closing(opening(time_series, footprint=morphology_struct), footprint=morphology_struct)

def get_hrf(method, tr=0.72):
    '''
    method: 'random', 'spm'
    '''

    if method == 'random':
        t = np.arange(0, 32, tr)
        hrf = spm_hrf_compat(
            t,
            peak_delay=np.random.uniform(5.5, 6.5),
            under_delay=np.random.uniform(15.5, 16.5),
            peak_disp=np.random.uniform(0.5, 1),
            under_disp=np.random.uniform(1.5, 1.8),
            p_u_ratio=max(6.5, np.random.normal(6, 0.5)),
            normalize=True
        )
    elif method == 'spm':
        hrf = spm_hrf(tr, oversampling=1, time_length=32.0, onset=0.) 
    
    return hrf


def convolve_multivariate_timeseries(dataset, method):

    print('Convolving signals with random HRFs with oversampling = 1.')

    n_points = dataset.shape[1]
    tr = .72

    convolved_dataset = []
    for i in tqdm(range(len(dataset))):
        hrf = get_hrf(method, tr)
        convolved_dataset.append(
            convolve(dataset[i], hrf)[:n_points]
            )
        
    convolved_dataset = np.stack(convolved_dataset)
    
    return convolved_dataset

def add_noise(dataset, snr):

    snr_mean, snr_scale = snr
    
    noise = np.random.normal(size=dataset.shape)
    power_signal = np.mean(dataset**2, axis=1)
    power_noise = np.mean(noise**2, axis=1)
    snr = np.clip(np.random.normal(loc=snr_mean, scale=snr_scale, size=power_signal.shape), a_min=-6, a_max=None)
    noise_std = np.sqrt(power_signal / (power_noise * (10**(snr/10))))

    dataset_noisy = dataset + np.expand_dims(noise_std, 1) * noise
    
    return dataset_noisy

def generate_dataset(n_points, n_voxels, regime, method='random'):
    """
    Generate a multivariate time series with different regimes.

    Parameters:


    Returns:
    - np.ndarray: A 2D array of shape (n_voxels, n_points).
    """
        
    simulated_signals = np.zeros((n_voxels, n_points))

    for voxel in tqdm(range(n_voxels)):
        simulated_signals[voxel] = generate_timeseries(n_points, regime)
    
    convolved_signals = convolve_multivariate_timeseries(simulated_signals, method)
    
    snr = regime['snr']
    dataset = add_noise(convolved_signals, snr)
        
    return torch.tensor(simulated_signals), torch.tensor(convolved_signals), torch.tensor(dataset)