import torch
import torch.nn as nn
import torch.nn.functional as F

from tools.models.base_models import BaseModel

'''
Those architectures might be deprecated. 
They were dropped from the studied models as they were underperforming compared to the CNNs and AEs.
'''

class LSTM_nl(BaseModel):
    def __init__(self, num_layers, hidden_size, input_size=1):
        super(LSTM_nl, self).__init__()
        
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        
        x, _ = self.rnn(x)
        x = self.fc(x)
        
        return x
    

class KernelRNN(nn.Module):
    def __init__(self, num_layers, hidden_size, kernel_size, input_size=1, dropout=0.2):
        super(KernelRNN, self).__init__()
        
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True, num_layers=num_layers, dropout=dropout)
        self.fc = nn.Linear(hidden_size, kernel_size)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        
        output_rnn, _ = self.rnn(x) # [batch_size, time_steps, hidden_size]
        output_rnn = self.layer_norm(output_rnn) # Normalize activations
        output_last = output_rnn[:, -1, :] # [batch_size, hidden_size]
        kernel = self.fc(output_last) # [batch_size, kernel_size]
        
        return kernel

class Deconvolution(nn.Module):
    def __init__(self):
        super(Deconvolution, self).__init__()
    
    def forward(self, x, kernel):
        # x: [batch_size, signal_length, 1]
        # kernel: [batch_size, signal_length]
        
        x = x.permute(0, 2, 1) # [batch_size, 1, signal_length]
        kernel = kernel.unsqueeze(dim=1) # [batch_size, 1, kernel_size]
        
        total_padding = kernel.shape[2] - 1
        left_padding = total_padding // 2
        right_padding = total_padding - left_padding
        x = F.pad(x, (left_padding, right_padding), mode='reflect') # [batch_size, 1, padded_signal_length]
        x = x.permute(1, 0, 2) # --> [1, batch_size, padded_signal_length]
        x = F.conv1d(x, kernel, groups=kernel.size(0))
        x.permute(1, 2, 0) # [batch_size, padded_signal_length, 1]
        
        return x
    
class OutputRNN(nn.Module):
    def __init__(self, num_layers, hidden_size, input_size=1, dropout=0.2):
        super(OutputRNN, self).__init__()
        
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True, num_layers=num_layers, dropout=dropout)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x):

        x = x.permute(1, 2, 0)

        x, _ = self.rnn(x)
        x = self.fc(x)
        
        return x
    
class RNNDeconvolutionRNN(nn.Module):
    '''
    Architecture from the ML4S students: input signal seeds a LSTM model that outputs a kernel (KernelRNN).
    The input is convolved with that kernel and then passed in a second LSTM model (OutputRNN) that outputs the deconvolved signal.
    '''
    def __init__(
        self, num_layers_kernel_rnn, hidden_size_kernel_rnn, num_layers_output_rnn, 
        hidden_size_output_rnn, kernel_size, input_size=1, dropout=0.2
    ):
        """
        num_layers_kernel_rnn: int, number of LSTM layers in the KernelRNN.
        hidden_size_kernel_rnn: int, hidden size for the KernelRNN.
        num_layers_output_rnn: int, number of LSTM layers in the OutputRNN.
        hidden_size_output_rnn: int, hidden size for the OutputRNN.
        kernel_size: int, kernel size for the KernelRNN's output.
        input_size: int, number of channels in the input (should always be 1).
        dropout: int, dropout to add in the both KernelRNN and OutputRNN.
        """
        super(RNNDeconvolutionRNN, self).__init__()
        
        self.kernel_rnn = KernelRNN(num_layers_kernel_rnn, hidden_size_kernel_rnn, kernel_size, input_size, dropout)
        self.deconvolution = Deconvolution()
        self.output_rnn = OutputRNN(num_layers_output_rnn, hidden_size_output_rnn, input_size, dropout)
        
    def forward(self, x):
        
        kernel = self.kernel_rnn(x)
        x = self.deconvolution(x, kernel)
        x = self.output_rnn(x)
        
        return x
    
    def train_step(self, optimizer, criterion, X, y):
        optimizer.zero_grad()
        predictions = self(X)
        loss = criterion(predictions, y)
        loss.backward()
        
        # Apply gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        
        optimizer.step()
        return loss.item(), predictions