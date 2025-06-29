import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from tools.analysis import get_best_models_dict
from tools.custom_loss import blocky_loss
from tools.training_functions import train_autoencoder
from tools.utils import pickle_dump, pickle_load, PaddedDatasetWrapper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lr = 0.001
batch_size = 64
stopping_params = {
    'num_epochs': 50,
    'patience': 12,
    'tolerance': 1e-4
}
criterion_params = {
    'alpha': 85.0,
    'lambda_tv': 0.0001
}
scheduler_params = {
    'patience': 5,
    'threshold': 1e-3,
    'factor': 0.5
}
checkpoint_filename = 'ae_finetuned.pkl'

def load_winning_ae_models():
    results_df = pd.read_csv('csv_logs/ae_logs_W9.csv')
    results_df = results_df[
        results_df['study_name'].apply(lambda s: int(s[-15:-7]) >= 20250424)
    ]
    sim_length, real_length = 300, 316
    results_df['signal_length'] = [sim_length if 'ALICE' in row.study_name else real_length for _, row in results_df.iterrows()]
    results_df = results_df.loc[results_df.study_name.str.contains('ALICE_')==False] # Correction to get the simulated model trained on corrected sim_data (HRF oversampling error)
    best_models_dict = get_best_models_dict(results_df)

    # Print winning models' names
    best_model_keys = list(best_models_dict.keys())
    print('### WINNING AE MODELS ###\n')
    print(best_models_dict[best_model_keys[0]][0])
    print('\n')
    print(best_models_dict[best_model_keys[1]][0])
    print('\n')

    # Gather winning models
    winning_ae_models = {
        "real": ['AE (real)'] + best_models_dict[best_model_keys[0]][1:],
        "simulated": ['AE (simulated)'] + best_models_dict[best_model_keys[1]][1:]
    }

    return winning_ae_models

def finetuning_pipeline(model, dataset):

    train_loader = DataLoader(dataset['train'], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset['validation'], batch_size=batch_size, shuffle=True)

    criterion = blocky_loss(**criterion_params)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_params)

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
    
    return output

def main():

    # Datasets
    real_data = torch.load('datasets/real_data_MGL_000000.pt')
    sim_data = torch.load('datasets/simulated_data_aliceO2_300m.pt') # corrected simulated dataset (HRF oversampling)

    # Wrap datasets with paddin transform to match models' input sizes
    real_data = {
        key: PaddedDatasetWrapper(subset, 300) for key, subset in real_data.items()
    }
    sim_data = {
        key: PaddedDatasetWrapper(subset, 316) for key, subset in sim_data.items()
    }

    ##################################
    ## FT - E/D + LAST LAYER OF LDM ##
    ##################################
    # Load models
    winning_ae_models = load_winning_ae_models()
    pickle_dump(winning_ae_models, f'final_models/{checkpoint_filename}')
    model_real, model_sim = winning_ae_models['real'][1].to(device), winning_ae_models['simulated'][1].to(device)

    # Finetune
    print("\n#### Finetuning AE (real) on simulated data - E/D + LDM last layer finetuning ####")
    for param in model_real.parameters():
        param.requires_grad = False
    for param in model_real.autoencoder.encoder.fc.parameters():
        param.requires_grad = True
    for param in model_real.autoencoder.decoder.fc.parameters():
        param.requires_grad = True
    for param in model_real.latent_deconv_model.model.conv4.parameters():
        param.requires_grad = True
    output_real_ft_edldm = finetuning_pipeline(model_real, sim_data)
    print("\n#### Finetuning AE (simulated) on real data - E/D + LDM last layer finetuning ####")
    for param in model_sim.parameters():
        param.requires_grad = False
    for param in model_sim.autoencoder.encoder.fc.parameters():
        param.requires_grad = True
    for param in model_sim.autoencoder.decoder.fc.parameters():
        param.requires_grad = True
    for param in model_sim.latent_deconv_model.model.conv4.parameters():
        param.requires_grad = True
    output_sim_ft_edldm = finetuning_pipeline(model_sim, real_data)

    # Save finetuned models
    winning_ae_models = pickle_load(f'final_models/{checkpoint_filename}')
    winning_ae_models['real_ft_ED_LDM'] = ['AE (real)\nFT ED_LDM', model_real, output_real_ft_edldm]
    winning_ae_models['simulated_ft_ED_LDM'] = ['AE (simulated)\nFT ED_LDM', model_sim, output_sim_ft_edldm]
    pickle_dump(winning_ae_models, f'final_models/{checkpoint_filename}')

    ##################################
    ######### FT - WHOLE LDM #########
    ##################################
    # Load models
    winning_ae_models = pickle_load(f'final_models/{checkpoint_filename}')
    model_real, model_sim = winning_ae_models['real'][1].to(device), winning_ae_models['simulated'][1].to(device)

    # Finetune
    print("\n#### Finetuning AE (real) on simulated data - whole LDM finetuning ####")
    for param in model_real.parameters():
        param.requires_grad = False
    for param in model_real.latent_deconv_model.model.parameters():
        param.requires_grad = True
    output_real_ft_ldm = finetuning_pipeline(model_real, sim_data)
    print("\n#### Finetuning AE (simulated) on real data - whole LDM finetuning ####")
    for param in model_sim.parameters():
        param.requires_grad = False
    for param in model_sim.latent_deconv_model.model.parameters():
        param.requires_grad = True
    output_sim_ft_ldm = finetuning_pipeline(model_sim, real_data)

    # Save finetuned models
    winning_ae_models = pickle_load(f'final_models/{checkpoint_filename}')
    winning_ae_models['real_ft_LDM'] = ['AE (real)\nFT LDM', model_real, output_real_ft_ldm]
    winning_ae_models['simulated_ft_LDM'] = ['AE (simulated)\nFT LDM', model_sim, output_sim_ft_ldm]
    pickle_dump(winning_ae_models, f'final_models/{checkpoint_filename}')

if __name__ == '__main__':
    
    main()