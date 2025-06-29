import numpy as np
import pickle
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from tools.models.autoencoder_models import *
from tools.models.cnn_models import *
from tools.models.lstm_models import *


MODEL_DICT = {
   'CNN_1L': CNN_1l,
   'CNN_2L': CNN_2l,
   'CNN_3L': CNN_3l,
   'CNN_4L': CNN_4l,
   'LSTM': LSTM_nl,
   'LSTMDeconvLSTM': RNNDeconvolutionRNN,
   'AutoEncoder': AutoEncoderDeconvolution
}


class FMRIDataset(Dataset):
    def __init__(self, X, y, convolved=None, final_length=None):
        self.X = X
        self.y = y
        self.convolved = convolved
        self.final_length = final_length

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if isinstance(idx, (list, np.ndarray, torch.Tensor)):
            return [self._get_single_item(i) for i in idx]
        else:
            return self._get_single_item(idx)

    def _get_single_item(self, idx):
        X_signal = self.X[idx].squeeze()
        y_signal = self.y[idx].squeeze()

        if self.final_length:
            padding = (self.final_length - self.X.shape[1]) / 2
            X_signal = F.pad(X_signal, (int(np.ceil(padding)), int(np.floor(padding))), mode='constant', value=0)
            y_signal = F.pad(y_signal, (int(np.ceil(padding)), int(np.floor(padding))), mode='constant', value=0)

        if self.convolved is not None:
            y_convolved = self.convolved[idx].squeeze()
            if self.final_length:
                padding = (self.final_length - self.convolved.shape[1]) / 2
                y_convolved = F.pad(y_convolved, (int(np.ceil(padding)), int(np.floor(padding))), mode='constant', value=0)
            return X_signal, y_signal, y_convolved
        else:
            return X_signal, y_signal
        
        
class PaddedDatasetWrapper(Dataset):
    def __init__(self, dataset, final_length):
        self.dataset = dataset
        self.final_length = final_length
        self.initial_length = len(dataset[0][0])
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        padding = (self.final_length - self.initial_length) / 2
        X, y = self.dataset[idx]
        X = F.pad(X, (int(np.ceil(padding)), int(np.floor(padding))), mode='constant', value=0)
        y = F.pad(y, (int(np.ceil(padding)), int(np.floor(padding))), mode='constant', value=0)
        return (X, y)

    
def get_number_of_parameters(model):
    return sum(p.numel() for p in model.parameters())


def pickle_dump(result_object, path):
    with open(path, 'wb') as file:
        pickle.dump(result_object, file)
    pass


def pickle_load(path):
    with open(path, 'rb') as file:
        results_object = pickle.load(file)
    return results_object


def model_save(model, path):
    torch.save(model.state_dict(), path)
    pass


def get_latent_dataloader(dataloader, autoencoder, device):
    '''
    Convert DataLoader from (X, y) to (latent, y). Used for autoencoder training
    '''

    latent_data, y_data = [], []
    for X, y in dataloader:
        X = X.unsqueeze(dim=2).to(device)
        _, latent = autoencoder(X)
        latent_data.append(latent)
        y_data.append(y)
    latent_data, y_data = torch.cat(latent_data).detach(), torch.cat(y_data).detach()

    dataset = FMRIDataset(latent_data, y_data)
    dataloader = DataLoader(dataset, batch_size=dataloader.batch_size, shuffle=True)
    
    return dataloader 
