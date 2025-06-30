import math
import torch
import torch.nn as nn


def me_tv_loss(lambda_tv=1., mean_error_loss=nn.MSELoss()):
    '''
    MSE or MAE + TV regularization loss.
    '''
    
    def loss_fn(predictions, targets):
        me_loss = mean_error_loss(predictions, targets)
        tv = torch.mean(
            torch.abs(predictions[:, 1:] - predictions[:, :-1])
        )
        return me_loss + lambda_tv * tv
    
    return loss_fn


def blocky_loss(alpha=0.0005, lambda_tv=1., lambda_const=0., mean_error_loss=nn.MSELoss()):
    '''
    MSE or MAE + gaussian TV regularization loss.
    '''

    def gaussian_tv_penalty(predictions, alpha):
        diff = torch.abs(predictions[:, 1:] - predictions[:, :-1])
        gaussian_tv = torch.mean(
            diff * torch.exp(-alpha * diff**2)
        )
        return gaussian_tv
    
    
    def loss_fn(predictions, targets):
        me_loss = mean_error_loss(predictions, targets)
        gaussian_tv_term = gaussian_tv_penalty(predictions, alpha)
        return me_loss + lambda_tv * gaussian_tv_term
    
    return loss_fn


        