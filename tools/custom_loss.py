import math
import torch
import torch.nn as nn


def me_tv_loss(lambda_tv=1., mean_error_loss=nn.MSELoss()):
    
    def loss_fn(predictions, targets):
        me_loss = mean_error_loss(predictions, targets)
        tv = torch.mean(
            torch.abs(predictions[:, 1:] - predictions[:, :-1])
        )
        return me_loss + lambda_tv * tv
    
    return loss_fn


def blocky_loss(alpha=0.0005, lambda_tv=1., lambda_const=0., mean_error_loss=nn.MSELoss()):
    
    # def gaussian_tv_penalty(predictions, alpha):
    #     diffs = torch.mean(
    #         torch.abs(predictions[:, 1:] - predictions[:, :-1])
    #     )
    #     #weights = math.sqrt(2 * alpha * math.e) * torch.exp(-alpha * diffs**2)
    #     weights = torch.exp(-alpha * diffs**2)
    #     return torch.mean(torch.abs(diffs) * weights)
    def gaussian_tv_penalty(predictions, alpha):
        diff = torch.abs(predictions[:, 1:] - predictions[:, :-1])
        gaussian_tv = torch.mean(
            diff * torch.exp(-alpha * diff**2)
        )
        return gaussian_tv
    
    #def anti_constant_penalty(predictions):
     #   varmean = torch.mean(torch.var(predictions, dim=1))
     #   return torch.pow(varmean - 1, 2)
    
    def loss_fn(predictions, targets):
        me_loss = mean_error_loss(predictions, targets)
        gaussian_tv_term = gaussian_tv_penalty(predictions, alpha)
        #anti_const_term = anti_constant_penalty(predictions)
        #return mae_loss + lambda_tv * gaussian_tv_term + lambda_const * anti_const_term
        return me_loss + lambda_tv * gaussian_tv_term
    
    return loss_fn


        