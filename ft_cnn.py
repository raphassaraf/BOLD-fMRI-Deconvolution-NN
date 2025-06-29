import pandas as pd
import torch

from torch.utils.data import DataLoader

from tools.analysis import get_best_models_dict
from tools.custom_loss import blocky_loss
from tools.models.cnn_models import CNN_4l_Adapter, CNN_4l_Adapter2
from tools.training_functions import train
from tools.utils import pickle_dump, pickle_load

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
checkpoint_filename = 'cnn_finetuned.pkl'

# Datasets
real_data = torch.load('datasets/real_data_MGL_000000.pt')
sim_data = torch.load('datasets/simulated_data_aliceO2_300m.pt') # corrected simulated dataset (HRF oversampling)

def load_winning_cnn_models():
    
    results_df = pd.read_csv(f'csv_logs/cnn_logs_W9.csv')
    results_df.loc[results_df.min_mse_val == results_df.min_mse_val.min()]
    best_models = pd.DataFrame([
        results_df.loc[results_df.study_name.str.contains('ALICEO2')].loc[lambda x: x['min_mse_val'].idxmin()], # Corrected to get the model with correct HRF oversampling
        results_df.loc[results_df.study_name.str.contains('MGL')].loc[lambda x: x['min_mse_val'].idxmin()],
    ])
    best_models_dict = get_best_models_dict(best_models)

    # Print winning models' names
    best_model_keys = list(best_models_dict.keys())
    print('### WINNING CNN MODELS ###\n')
    print(best_models_dict[best_model_keys[0]][0])
    print('\n')
    print(best_models_dict[best_model_keys[1]][0])
    print('\n')

    # Gather winning models
    winning_cnn_models = {
        "real": ['CNN (real)'] + best_models_dict[best_model_keys[1]][1:], # Real model key is located at idx=1 of best_model_keys
        "simulated": ['CNN (simulated)'] + best_models_dict[best_model_keys[0]][1:] # Simulated model key is located at idx=0 of best_model_keys
    }

    return winning_cnn_models

def finetuning_pipeline(model, dataset):

    train_loader = DataLoader(dataset['train'], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset['validation'], batch_size=batch_size, shuffle=True)

    criterion = blocky_loss(**criterion_params)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_params)
  
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

    #######################
    ### FT - LAST LAYER ###
    #######################
    # Load models
    winning_cnn_models = load_winning_cnn_models()
    pickle_dump(winning_cnn_models, f'final_models/{checkpoint_filename}')
    model_real, model_sim = winning_cnn_models['real'][1].to(device), winning_cnn_models['simulated'][1].to(device)

    # Finetune
    print("\n#### Finetuning CNN (simulated) on real data - last layer finetuning ####")
    for param in model_sim.parameters():
        param.requires_grad = False
    for param in model_sim.conv4.parameters():
        param.requires_grad = True
    output_sim_ft_lastlayer = finetuning_pipeline(model_sim, real_data)
    print("\n#### Finetuning CNN (real) on simulated data - last layer finetuning ####")
    for param in model_real.parameters():
        param.requires_grad = False
    for param in model_real.conv4.parameters():
        param.requires_grad = True
    output_real_ft_lastlayer = finetuning_pipeline(model_real, sim_data)

    # Save finetuned models
    winning_cnn_models = pickle_load(f'final_models/{checkpoint_filename}')
    winning_cnn_models['real_ft_lastlayer'] = ['CNN (real)\nFT last_layer', model_real, output_real_ft_lastlayer]
    winning_cnn_models['simulated_ft_lastlayer'] = ['CNN (simulated)\nFT last_layer', model_sim, output_sim_ft_lastlayer]
    pickle_dump(winning_cnn_models, f'final_models/{checkpoint_filename}')

    #######################
    #### FT - ADAPTER #####
    #######################
    # Load models
    winning_cnn_models = pickle_load(f'final_models/{checkpoint_filename}')
    model_real, model_sim = winning_cnn_models['real'][1], winning_cnn_models['simulated'][1]
    model_real_adapter = CNN_4l_Adapter(model_real, 21).to(device)
    model_sim_adapter = CNN_4l_Adapter(model_sim, 21).to(device)

    # Finetune
    print("\n#### Finetuning CNN (simulated) on real data - adapter finetuning ####")
    output_sim_ft_adapter = finetuning_pipeline(model_sim_adapter, real_data)
    print("\n#### Finetuning CNN (real) on simulated data - adapter finetuning ####")
    output_real_ft_adapter = finetuning_pipeline(model_real_adapter, sim_data)

    # Save finetuned models
    winning_cnn_models = pickle_load(f'final_models/{checkpoint_filename}')
    winning_cnn_models['real_ft_adapter'] = ['CNN (real)\nFT adapter', model_real_adapter, output_real_ft_adapter]
    winning_cnn_models['simulated_ft_adapter'] = ['CNN (simulated)\nFT adapter', model_sim_adapter, output_sim_ft_adapter]
    pickle_dump(winning_cnn_models, f'final_models/{checkpoint_filename}')

    #######################
    #### FT - ADAPTER2 ####
    #######################
    # Load models
    winning_cnn_models = pickle_load(f'final_models/{checkpoint_filename}')
    model_real, model_sim = winning_cnn_models['real'][1], winning_cnn_models['simulated'][1]
    model_real_adapter2 = CNN_4l_Adapter2(model_real, 21, 21).to(device)
    model_sim_adapter2 = CNN_4l_Adapter2(model_sim, 21, 21).to(device)

    # Finetune
    print("\n#### Finetuning CNN (simulated) on real data - adapter2 finetuning ####")
    output_sim_ft_adapter2 = finetuning_pipeline(model_sim_adapter2, real_data)
    print("\n#### Finetuning CNN (real) on simulated data - adapter2 finetuning ####")
    output_real_ft_adapter2 = finetuning_pipeline(model_real_adapter2, sim_data)

    # Save finetuned models
    winning_cnn_models = pickle_load(f'final_models/{checkpoint_filename}')
    winning_cnn_models['real_ft_adapter2'] = ['CNN (real)\nFT adapter2', model_real_adapter2, output_real_ft_adapter2]
    winning_cnn_models['simulated_ft_adapter2'] = ['CNN (simulated)\nFT adapter2', model_sim_adapter2, output_sim_ft_adapter2]
    pickle_dump(winning_cnn_models, f'final_models/{checkpoint_filename}')
    
if __name__ == '__main__':
    
    main()
