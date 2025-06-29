import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import torch
import torch.nn.functional as F

from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter
from torch.utils.data import DataLoader

from tools.models.autoencoder_models import LatentDeconvolution, AutoEncoder, AutoEncoderDeconvolution
from tools.utils import pickle_load, MODEL_DICT


def get_best_models_df(logs, idx=None, top_k=1, sort=True):
    '''
    Pandas query to get the best or selected models
    '''
    if idx is not None:
        best_models = logs.iloc[idx]
    else:
        logs = logs.dropna(subset=['test_mse'])
        best_models = logs.loc[logs.groupby('study_name').apply(lambda x: x.nsmallest(top_k, 'min_mse_val')).index.get_level_values(1)]

    if sort:
        best_models = best_models.sort_values(by='min_mse_val').reset_index()

    return best_models    

def add_formatted_model_name(df):
    '''
    Add model_type, model_name_formatted and architecture_dict in the logs dataframe
    '''

    def get_loss_name(str_split, idx):
        loss_name = f'{str_split[idx]}_{str_split[idx+1]} ({str_split[idx+2]}={float(str_split[idx+3]):.4f}'
        if str_split[idx+2] == 'a':
            loss_name += f', {str_split[idx+4]}={float(str_split[idx+5]):.4f})'
        else:
            loss_name += ')'
        return loss_name    
    
    def get_regime(study_name):
        return '(' + study_name[study_name.find('Data')+4:study_name.find('_', study_name.find('Data'))] + ')'
    
    def get_formatted_model_info(row):
        study_name, model_path, n_params = row.study_name, row.model_name, row.n_params
        n_params = f'{n_params} params'
        str_split = re.split(r'[_=(),\s]+', model_path)
        
        if model_path[:3] == 'CNN':
            regime = get_regime(study_name)
            model_type = str.upper(model_path[:6])
            if model_type == 'CNN_1L':
                k = int(str_split[3])
                loss_name = get_loss_name(str_split, 5)
                architecture = f'(K={k})'
                architecture_dict = {'k': k}

            elif model_type == 'CNN_2L':
                k1, k2 = [int(str_split[i + 1]) for i in range(len(str_split) - 1) if str_split[i].startswith('kernel') and str_split[i + 1].isdigit()]
                f = int(str_split[7])
                loss_name = get_loss_name(str_split, 9)
                architecture = f'(K1={k1}, K2={k2}, F={f})' # F = number of filters (out_channels argument in torch.nn.Conv1d)
                architecture_dict = {'k1': k1, 'k2': k2, 'f':f}

            elif model_type == 'CNN_3L':
                k1, k2, k3 = [int(str_split[i + 1]) for i in range(len(str_split) - 1) if str_split[i].startswith('kernel') and str_split[i + 1].isdigit()]
                f1, f2 = [int(str_split[i + 1]) for i in range(len(str_split) - 1) if str_split[i].startswith('hidden') and str_split[i + 1].isdigit()]
                loss_name = get_loss_name(str_split, 13)
                architecture = f'(K1={k1}, K2={k2}, K3={k3}, F1={f1}, F2={f2})'
                architecture_dict = {'k1': k1, 'k2': k2, 'k3': k3, 'f1': f1, 'f2': f2}

            elif model_type == 'CNN_4L':
                k1, k2, k3, k4 = [int(str_split[i + 1]) for i in range(len(str_split) - 1) if str_split[i].startswith('kernel') and str_split[i + 1].isdigit()]
                f1, f2, f3 = [int(str_split[i + 1]) for i in range(len(str_split) - 1) if str_split[i].startswith('hidden') and str_split[i + 1].isdigit()]
                loss_name = get_loss_name(str_split, 17)
                architecture = f'(K1={k1}, K2={k2}, K3={k3}, K4={k4}, F1={f1}, F2={f2}, F3={f3})'
                architecture_dict = {'k1': k1, 'k2': k2, 'k3': k3, 'k4': k4, 'f1': f1, 'f2': f2, 'f3': f3}

        elif model_path[:4] == 'LSTM':
            regime = get_regime(study_name)
            if model_path[:5] == 'LSTM_':
                num_layers, hidden_size = int(str_split[1][0]), int(str_split[3])
                loss_name = get_loss_name(str_split, 5)
                model_type = 'LSTM'
                architecture = f'({num_layers}L, H={hidden_size})'
                architecture_dict = {'l': num_layers, 'h': hidden_size}
 
            elif model_path[:10] == 'LSTMDeconv':
                num_layers_K, hidden_size_K = int(str_split[1][0]), int(str_split[3])
                num_layers_O, hidden_size_O = int(str_split[4][0]), int(str_split[6])
                kernel_size = int(str_split[8])
                loss_name = get_loss_name(str_split, 10)
                model_type = 'LSTMDeconvLSTM'
                architecture = f'({num_layers_K}L(K), {hidden_size_K}H(K) | {num_layers_O}L(O), {hidden_size_O}H(O) | kernel={kernel_size})'
                architecture_dict = {'lk': num_layers_K, 'hk': hidden_size_K, 'lo': num_layers_O, 'ho': hidden_size_O, 'k': kernel_size}
        
        elif model_path[:11] == 'AutoEncoder':
            regime = get_regime(study_name)
            c, f, k = [int(str_split[i]) for i in [2, 4, 6]]
            d = float(str_split[8])
            # loss_name = get_loss_name(str_split, 22)
            loss_name = 'MSE'
            model_type = 'AutoEncoder(' + str_split[9] + '_' + str.upper(str_split[10]) + ')'
            architecture = f'(C={c}, F={f}, K={k}, D={d:.1f})'
            architecture_dict = {
                'ae': {'c': c, 'f': f, 'k': k, 'd': d},
                'ldm': {str_split[i]: int(str_split[i+1]) for i in range(11, str_split.index('criterion'), 2)}
            }
            
        model_name = f'{model_type} {regime}\n{n_params}\n{architecture}\n{loss_name}'

        return pd.Series([model_type, model_name, architecture_dict])

    df[['model_type', 'model_name_formatted', 'architecture_dict']] = df.apply(get_formatted_model_info, axis=1)

    return df


def get_best_models_dict(logs, idx=None, top_k=1):
    '''
    Get the best models dictionnary needed to creates the results visualistion plots
    '''

    best_models = get_best_models_df(logs, idx, top_k)
    best_models = add_formatted_model_name(best_models)

    best_models_dict = {}
    for i, row in best_models.iterrows():

        study_name = row.study_name
        model_path, model_type, model_name = row.model_name, row.model_type, row.model_name_formatted

        if 'AutoEncoder' not in model_type:
            params = row.architecture_dict.values()
            model = MODEL_DICT[model_type](*params)
        else:
            params = row.architecture_dict
            latent_deconv_model_type = model_type[12:18]
            compression = params['ae']['c']
            signal_length = row.signal_length

            autoencoder = AutoEncoder(signal_length, *params['ae'].values())
            latent_deconv_model = LatentDeconvolution(
                signal_length,
                compression,
                MODEL_DICT[latent_deconv_model_type](*params['ldm'].values())
            )

            model = AutoEncoderDeconvolution(autoencoder, latent_deconv_model)
        
        model.load_state_dict(torch.load(f'model_logs/{study_name}/{model_path}'))
        training_output = pickle_load(f'training_outputs/{study_name}/{model_path}')
    
        if idx is not None:
            key = i
        elif top_k > 1:
            key = f'{study_name}_{i%top_k+1}'
        else:
            key = study_name
        best_models_dict[key] = [model_name, model, training_output]
        
    return best_models_dict

def plot_losses_per_model(best_models_dict):
    
    n_models = len(best_models_dict.keys())
    columns = 4 if len(best_models_dict.keys()) >=4 else len(best_models_dict.keys())
    rows = int(np.ceil(n_models / columns))
    
    f = plt.figure(figsize=(7*columns, 5*rows))
    gs = GridSpec(2*rows, columns, figure=f)  # twice more rows to split

    handles, labels = [], []

    for idx, (model_info) in enumerate(best_models_dict.values()):
        col = idx % columns
        row = idx // columns

        name, training_outputs = model_info[0], model_info[2]

        if 'AutoEncoder' not in name:
            ax = f.add_subplot(gs[2*row:2*row+2, col])  # take 2 rows for normal plot
            h1, = ax.plot(training_outputs['mse_train'], label='train', c='tab:blue', linewidth=3)
            h2, = ax.plot(training_outputs['mse_val'], label='validation', c='tab:orange', linewidth=3)
            handles.extend([h1, h2])

            if 'lr' in training_outputs.keys():
                ax2 = ax.twinx()
                h3, = ax2.plot(training_outputs['lr'], c='tab:green', label='learning rate', linewidth=3)
                ax2.tick_params(axis='y', labelcolor='tab:green')
                ax2.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
                handles.append(h3)

            ax.set_title(name, fontsize=16)

        else:
            ax_top = f.add_subplot(gs[2*row, col])
            ax_bottom = f.add_subplot(gs[2*row+1, col])

            ax_top2 = ax_top.twinx()
            ax_bottom2 = ax_bottom.twinx()

            h1, = ax_top.plot(training_outputs['mse_train']['ae'], label='train', c='tab:blue', linewidth=3)
            h2, = ax_top.plot(training_outputs['mse_val']['ae'], label='validation', c='tab:orange', linewidth=3)
            h3, = ax_top2.plot(training_outputs['lr']['ae'], c='tab:green', label='learning rate', linewidth=3)

            ax_bottom.plot(training_outputs['mse_train']['ldm'], label='train', c='tab:blue', linewidth=3)
            ax_bottom.plot(training_outputs['mse_val']['ldm'], label='validation', c='tab:orange', linewidth=3)
            ax_bottom2.plot(training_outputs['lr']['ldm'], c='tab:green', label='learning rate', linewidth=3)

            for ax_ in [ax_top2, ax_bottom2]:
                ax_.tick_params(axis='y', labelcolor='tab:green')
                ax_.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

            if col == 0:   # only first column
                ax_top.set_ylabel('AE', fontsize=16)
                ax_bottom.set_ylabel('LDM', fontsize=16)

            ax_top.set_title(name, fontsize=16)

            handles.extend([h1, h2, h3])

    f.legend(handles, ['train', 'validation', 'learning rate'], loc='lower center', bbox_to_anchor=(0.5, -0.08), ncol=3, fontsize=14)
    f.tight_layout()
    
    return f


def plot_predictions(best_models_dict, dataset):
    
    device = 'cpu'
        
    n_models = len(best_models_dict.keys())

    fig = plt.figure(figsize=(20, 5*n_models), constrained_layout=True)
    subfigs = fig.subfigures(nrows=n_models, ncols=1)
    if n_models == 1: subfigs = [subfigs]

    random_voxels = np.random.randint(0, len(dataset), size=(4,))

    for row, (model_info, subfig) in enumerate(zip(best_models_dict.values(), subfigs)):
        name, model = model_info[0], model_info[1].to(device)
        subfig.suptitle(name, fontsize=16)

        ax = subfig.subplots(1, 4)
        for col, a in enumerate(ax):
            voxel = random_voxels[col]
            input_sample = dataset[voxel][0]
            target_sample = dataset[voxel][1]

            if type(model) == AutoEncoderDeconvolution:
                padding = (model.signal_length - input_sample.shape[0]) / 2
                input_sample = F.pad(input_sample, (int(np.ceil(padding)), int(np.floor(padding))), mode='constant', value=0)
                target_sample = F.pad(target_sample, (int(np.ceil(padding)), int(np.floor(padding))), mode='constant', value=0)
                reconstructed, _, prediction = model(input_sample.unsqueeze(dim=-1).unsqueeze(dim=0))
                reconstructed = reconstructed.detach().squeeze()
                a.plot(reconstructed, linewidth=1, alpha=0.7, label='reconstructed', c='tab:purple')
            else:
                prediction = model(input_sample.unsqueeze(dim=-1).unsqueeze(dim=0))
            prediction = prediction.detach().squeeze()

            a.plot(input_sample, linewidth=0.5, alpha=0.7, label='input', c='tab:blue')
            a.plot(target_sample, linewidth=2, label='target', c='tab:orange')

            a.plot(prediction, label='prediction', c='tab:green', linewidth=3)
            # a.grid()
            # a.legend(fontsize=14)
            a.set_ylim(-3, 3)
    
        handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.08), fontsize=16, ncol=len(handles))

    return fig

def test_model(test_loader, model, device="cuda", metric_fn=torch.nn.MSELoss()):
    
    model = model.to(device)
    
    metric = []
    for X, y in test_loader:
        X, y = X.unsqueeze(dim=2).to(device), y.unsqueeze(dim=2).to(device)
        if type(model) == AutoEncoderDeconvolution:
            padding = (model.signal_length - X.shape[1]) / 2
            X = F.pad(X, (0, 0, int(np.ceil(padding)), int(np.floor(padding))), mode='constant', value=0)
            y = F.pad(y, (0, 0, int(np.ceil(padding)), int(np.floor(padding))), mode='constant', value=0)
            _, _, predictions = model(X)
        else:
            predictions = model(X)
        metric.append(metric_fn(predictions, y).item())
    metric = torch.tensor(metric)

    mean, var = torch.mean(metric).item(), torch.var(metric).item()
    
    return mean, var

def compute_test_metric(best_models_dict, dataset, device="cuda", metric_fn=torch.nn.MSELoss(), batch_size=64):
    
    dataloader = DataLoader(dataset, batch_size)
    
    test_results = {'model': [], 'metric': [], 'var': []}
    for model_info in best_models_dict.values():
        name, model = model_info[0], model_info[1]
        test_results['model'].append(name)
        
        mean, var = test_model(dataloader, model, device, metric_fn)
        test_results['metric'].append(mean)
        test_results['var'].append(var)
    
    test_results = pd.DataFrame(test_results)
    
    return test_results


def analyse_hp_search(results_df, dataset, idx=None, top_k=1):

    best_models_dict = get_best_models_dict(results_df, idx, top_k)

    loss_plot = plot_losses_per_model(best_models_dict)
    inference_plot = plot_predictions(best_models_dict, dataset)

    df_test_mse = compute_test_metric(best_models_dict, dataset)

    return loss_plot, inference_plot, df_test_mse

def get_study_path(study_name):
    optuna_folder = 'optuna_logs'
    return os.path.join(optuna_folder, study_name)