import torch
import torch.nn as nn
import torch.nn.functional as F

from tools.models.base_models import BaseModel

class LatentDeconvolution(BaseModel):
    def __init__(self, signal_length, compression, model):
        super(LatentDeconvolution, self).__init__()
        '''
        Model: can be CNN, LSTM, ... idea is to pick one of the winning models of the previous HP searches
        '''
        
        self.fc = nn.Linear(signal_length//compression, signal_length)
        self.model = model

        self.relu = nn.ReLU()
    
    def forward(self, x):
        
        # FC layer puts input back to initial's data shape (B, L, 1)
        x = self.fc(x.squeeze())
        x = x.reshape((-1, x.shape[-1], 1))

        # Pass the output of the FC to the predefined model
        x = self.model(x)

        return x

class Encoder(nn.Module):
    def __init__(self, signal_length, compression, filter_number, kernel_size, dropout=0.2):
        '''
        Compression determines by what factor is the signal compressed. Starting with 8, might have to use lower value....
        '''
        super(Encoder, self).__init__()

        self.n_pooling = 1

        self.conv1 = nn.Conv1d(1, filter_number, kernel_size, padding=kernel_size//2)        
        self.maxpool = nn.MaxPool1d(2, return_indices=True)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(filter_number*signal_length//2, signal_length//compression)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        
        x = x.permute(0, 2, 1)

        x = self.relu(self.conv1(x))
        x, switches = self.maxpool(x)
        x = x.flatten(start_dim=1) # (B, L*F//2, 1)
        x = self.fc(x)
        x = self.dropout(x)
        
        return x, switches
    
class Decoder(nn.Module):
    def __init__(self, signal_length, compression, filter_number, kernel_size, dropout=0.2):
        super(Decoder, self).__init__()

        self.filter_number = filter_number
        self.n_pool = 1

        self.conv1 = nn.Conv1d(filter_number, 1, kernel_size, padding=kernel_size//2)
        self.maxunpool = nn.MaxUnpool1d(2)
        self.fc = nn.Linear(signal_length//compression, filter_number*signal_length//2)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, switches):
        
        x = self.fc(x)
        x = self.dropout(x)
        x = x.reshape((x.shape[0], self.filter_number*self.n_pool, -1))
        x = self.maxunpool(x, switches)
        x = self.conv1(x)

        x = x.permute(0, 2, 1)
         
        return x
    
class AutoEncoder(nn.Module):
    def __init__(self, signal_length, compression, filter_number=4, kernel_size=5, dropout=0.2):
        super(AutoEncoder, self).__init__()

        self.signal_length = signal_length

        self.encoder = Encoder(signal_length, compression, filter_number, kernel_size, dropout)
        self.decoder = Decoder(signal_length, compression, filter_number, kernel_size, dropout)
        
    def forward(self, x):

        latent, switches = self.encoder(x) # (B, L*F//2, 1)
        reconstructed = self.decoder(latent, switches) # (B, L, 1)

        return reconstructed, latent
    
    def train_step(self, optimizer, criterion, X):
        optimizer.zero_grad()
        reconstructed, latent = self(X)
        loss = criterion(reconstructed, X)
        loss.backward()
        optimizer.step()
        return loss.item(), reconstructed, latent.detach()
    
class AutoEncoderDeconvolution(nn.Module):
    def __init__(self, autoencoder, latent_deconv_model):
        super(AutoEncoderDeconvolution, self).__init__()

        self.signal_length = autoencoder.signal_length

        self.autoencoder = autoencoder
        self.latent_deconv_model = latent_deconv_model
    
    def forward(self, x):

        reconstructed, latent = self.autoencoder(x)
        prediction = self.latent_deconv_model(latent)

        return reconstructed, latent, prediction