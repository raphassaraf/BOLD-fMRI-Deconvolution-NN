import torch
import torch.nn as nn
import yaml

from torch.utils.data import DataLoader

from tools.custom_loss import blocky_loss
from tools.models.autoencoder_models import AutoEncoderDeconvolution, AutoEncoder, LatentDeconvolution
from tools.models.cnn_models import CNN_4l, CNN_3l
from tools.training_functions import train, train_autoencoder
from tools.utils import MODEL_DICT, pickle_dump

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


with open('configs/config_mixed_data_training.yaml', 'r') as file:
    CONFIG = yaml.safe_load(file)
    
model_type = CONFIG['model_type']
lr = CONFIG['lr']
batch_size = CONFIG['batch_size']
stopping_params = CONFIG['stopping_params']
criterion_params = CONFIG['criterion_params']
scheduler_params = CONFIG['scheduler_params']
output_filename = f'mixed_models_{model_type}.pkl'

def training_pipeline(model, dataset):

    train_loader = DataLoader(dataset['train'], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset['validation'], batch_size=batch_size, shuffle=True)

    criterion = blocky_loss(**criterion_params)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_params)

    if type(model) == AutoEncoderDeconvolution:
        criterion_ae = nn.MSELoss()
        optimizer_ae = torch.optim.Adam(model.autoencoder.parameters(), lr=lr)
        scheduler_ae = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_ae, **scheduler_params)

        _, output = train_autoencoder(
            autoencoder=model.autoencoder,
            latent_deconv_model=model.latent_deconv_model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion_ae=criterion_ae,
            criterion_ldm=criterion,
            optimizer_ae=optimizer_ae,
            optimizer_ldm=optimizer,
            scheduler_ae=scheduler_ae,
            scheduler_ldm=scheduler,
            device=device,
            **stopping_params
        )
    
    else:
        output = train(
            model=model, 
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            **stopping_params
        )
    
    return output

def main():

    print(f'### Launching mixed data training with model_type: {model_type} ###')

    # Import dictionnary containing each mixed dataset
    print('\n\n#### Importing datasets. ####\n\n')
    dataset_dict = torch.load('datasets/mixed_datasets_900m.pt')
    signal_length = len(dataset_dict['0.0']['train'][0][0])

    # Train the models
    output_dict = {}
    models_dict = {}
    for prop_real, dataset in dataset_dict.items():
        print(f'\n\n#### Train {model_type} model on prop_real = {prop_real} dataset. ####\n\n')

        if model_type == 'CNN_small':
            model = CNN_3l(13, 13, 13, 4, 8).to(device)
        if model_type == 'CNN_4L':
            model = CNN_4l(70, 30, 35, 53, 9, 10, 10).to(device)
        elif model_type == 'AutoEncoderDeconvolution':
            compression = 4
            autoencoder = AutoEncoder(signal_length, compression, 8, 13, 0.1)
            ldm = LatentDeconvolution(signal_length, compression, CNN_4l(25, 25, 25, 25, 10, 10, 10))
            model = AutoEncoderDeconvolution(autoencoder, ldm).to(device)

        output_dict[prop_real] = training_pipeline(model, dataset)
        models_dict[prop_real] = model
    
    # Gather the resutls and save
    mixed_data_output = {
        'output_dict': output_dict,
        'models_dict': models_dict
    }
    pickle_dump(mixed_data_output, f'final_models/{output_filename}')

if __name__ == '__main__':
    
    main()