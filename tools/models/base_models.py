import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
    
    def train_step(self, optimizer, criterion, X, y):
        optimizer.zero_grad()
        predictions = self(X)
        loss = criterion(predictions, y)
        loss.backward()
        optimizer.step()
        return loss.item(), predictions