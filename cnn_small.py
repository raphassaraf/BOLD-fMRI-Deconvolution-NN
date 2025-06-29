import torch
from torch.utils.data import DataLoader

from tools.utils import pickle_dump, pickle_load
from tools.models.cnn_models import CNN_3l
from tools.custom_loss import blocky_loss
from tools.training_functions import train

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

def training_pipeline(model, dataset):

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

real_data = torch.load('datasets/real_data_MGL_080808.pt')
sim_data = torch.load('datasets/simulated_data_aliceO2_1M.pt')

def main():

    # Pretrain base models
    model_sim = CNN_3l(13, 13, 13, 4, 8).to(device)
    model_real = CNN_3l(13, 13, 13, 4, 8).to(device)

    output_real = training_pipeline(model_real, real_data)
    output_sim = training_pipeline(model_sim, sim_data)

    small_models_dict = {}
    small_models_dict['real'] = ['CNN-small (real)', model_real, output_real]
    small_models_dict['simulated'] = ['CNN-small (simulated)', model_sim, output_sim]
    pickle_dump(small_models_dict, 'final_models/cnn_small.pkl')

    # Finetune using last-layer strategy
    print("\n#### Finetuning CNN (simulated) on real data - last layer finetuning ####")
    for param in model_sim.parameters():
        param.requires_grad = False
    for param in model_sim.conv3.parameters():
        param.requires_grad = True
    output_sim_ft_lastlayer = training_pipeline(model_sim, real_data)
    print("\n#### Finetuning CNN (real) on simulated data - last layer finetuning ####")
    for param in model_real.parameters():
        param.requires_grad = False
    for param in model_real.conv3.parameters():
        param.requires_grad = True
    output_real_ft_lastlayer = training_pipeline(model_real, sim_data)

    # Save finetuned models
    small_models_dict = pickle_load('final_models/cnn_small.pkl')
    small_models_dict['real_ft_lastlayer'] = ['CNN-small FT (real)', model_real, output_real_ft_lastlayer]
    small_models_dict['simulated_ft_lastlayer'] = ['CNN-small FT (simulated)', model_sim, output_sim_ft_lastlayer]
    pickle_dump(small_models_dict, 'final_models/cnn_small.pkl')

if __name__ == '__main__':
    
    main()



