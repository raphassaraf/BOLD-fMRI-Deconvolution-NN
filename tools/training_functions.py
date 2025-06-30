import torch
import torch.nn as nn

from tools.models.autoencoder_models import AutoEncoderDeconvolution, AutoEncoder
from tools.utils import model_save, pickle_dump, get_latent_dataloader


def train(
    model, train_loader, val_loader, criterion, optimizer,
    scheduler, device, num_epochs, model_name=None, study_name=None,
    patience=5, tolerance=1e-4, test_loader=None
):
    '''
    Base training loop.

    train_loader: DataLoader, training data.
    val_loader: DataLoader, validation data.
    criterion: loss function.
    optimizer: torch.optim optimizer.
    device: 'cuda' or 'cpu'.
    num_epochs: int, max number of epochs.
    model_name: model name to use for logging.
    study_name: name of the Optuna study within which the model is trained.
    patience: int, patience parameter for early stopping.
    tolerance: int, tolerance for early stopping.
    test_loader: DataLoader, testing data.
    '''
    
    # train_loss and val_loss correspond to the loss calculated with the criterion at each epoch
    # mse_train, mse_val correspond to the MSE loss calculation at each epoch. Those values are 
    # used as metrics for comparing different models with each other.
    mse_metric = nn.MSELoss() 
    output = {
        'train_loss': [],
        'val_loss': [],
        'mse_train': [],
        'mse_val': [],
        'mse_test': None,
        'mse_var_test': None,
        'lr': [optimizer.param_groups[0]['lr']]
    }
    
    best_model_state = None
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):

        train_loss, mse_train, val_loss, mse_val = train_val_epoch(
            model, train_loader, val_loader, criterion, optimizer, device, mse_metric
            )
        
        # Losses used for scheduler and training progression tracking
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        # Losses used for model comparisons
        avg_mse_train = mse_train / len(train_loader)
        avg_mse_val = mse_val / len(val_loader)
        
        # Logging
        output['train_loss'].append(avg_train_loss)
        output['val_loss'].append(avg_val_loss)
        output['mse_train'].append(avg_mse_train)
        output['mse_val'].append(avg_mse_val)
        
        scheduler.step(avg_val_loss)
        current_lr = scheduler._last_lr[0]
        output['lr'].append(current_lr)
        
        print(f"Epoch {epoch}: train_loss = {avg_train_loss} | val_loss = {avg_val_loss} | lr = {current_lr}")
        
        # Early stopping
        if torch.isnan(torch.tensor(output['val_loss'][-1])):
            print(f"Early stopping at epoch {epoch} due to nan value")
            break
        elif output['val_loss'][-1] < best_val_loss - tolerance:
            best_val_loss = output['val_loss'][-1]
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
        
        # Early stopping if no improvements are made for 'patience' epochs
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch} with loss {best_val_loss}")
            break
        
    if best_model_state:
        model.load_state_dict(best_model_state)
        
    if test_loader:
        output['mse_test'], output['mse_var_test'] = test_model(test_loader, model, mse_metric)
    
    if study_name and model_name:
        save_training_logs(model, output, study_name, model_name)
    
    return output


def train_autoencoder(
    autoencoder, latent_deconv_model, train_loader, val_loader,
    criterion_ae, criterion_ldm, optimizer_ae, optimizer_ldm,
    scheduler_ae, scheduler_ldm, device, num_epochs, model_name=None, study_name=None,
    patience=5, tolerance=1e-4, test_loader=None
    ):
    '''
    AutoEncoder training loop. Trains the Encoder+Decoder blocks and LatentDeconvolution block sequentially.

    autoencoder: AutoEncoder object.
    latent_deconv_model: LatentDeconvolution object.
    train_loader: DataLoader, training data.
    val_loader: DataLoader, validation data.
    criterion_ae: loss function for the autoencoder.
    criterion_ldm: loss function for the latent deconvolution model.
    optimizer_ae: optimizer for the autoencoder.
    optimizer_ldm: optimizer for the latent deconvolution model.
    scheduler_ae: learning rate scheduler for the autoencoder.
    scheduler_ldm: learning rate scheduler for the latent deconvolution model.
    device: 'cuda' or 'cpu'.
    num_epochs: int, max number of epochs.
    model_name: model name to use for logging.
    study_name: name of the Optuna study within which the model is trained.
    patience: int, patience parameter for early stopping.
    tolerance: int, tolerance for early stopping.
    test_loader: DataLoader, testing data.
    '''

    # Train AutoEncoder
    try:
        output_ae = train(
            autoencoder, train_loader, val_loader, criterion_ae,
            optimizer_ae, scheduler_ae, device, num_epochs, 
            patience=patience, tolerance=tolerance, test_loader=test_loader
            )
    except:
        output_ae = {
            'train_loss': None,
            'val_loss': None,
            'mse_train': None,
            'mse_val': None,
            'mse_test': None,
            'mse_var_test': None,
            'lr': None
        }
    
    # Freeze the AutoEncoder once it's done training
    autoencoder.eval()
    for param in autoencoder.parameters():
        param.requires_grad = False

    # Convert DataLoaders from (X, y) to (latent, y)
    train_loader_latent = get_latent_dataloader(train_loader, autoencoder, device)
    val_loader_latent = get_latent_dataloader(val_loader, autoencoder, device)
    if test_loader:
        test_loader_latent = get_latent_dataloader(test_loader, autoencoder, device)
    else:
        test_loader_latent = None
    
    # Train LatentDeconvolution model
    output_ldm = train(
        latent_deconv_model, train_loader_latent, val_loader_latent, criterion_ldm,
        optimizer_ldm, scheduler_ldm, device, num_epochs,
        patience=patience, tolerance=tolerance, test_loader=test_loader_latent
        )
    
    # Merge AutoEncoder and LatentDeconvolution model in one final model
    final_model = AutoEncoderDeconvolution(autoencoder, latent_deconv_model)

    # Gather the outputs from the AutoEncoder and LatentDeconvolution trainings
    output = {
        key: {'ae': value_ae, 'ldm': value_ldm} 
        for (key, value_ae), value_ldm in zip(output_ae.items(), output_ldm.values())
    }

    # Save the logs (final_model and output containing information on AutoEncoder and LatentDeconvolution trainings)
    if study_name and model_name:
        save_training_logs(final_model, output, study_name, model_name)

    return final_model, output


def train_val_epoch(model, train_loader, val_loader, criterion, optimizer, device, mse_metric=nn.MSELoss()):
    '''
    Perform training and validation for 1 epoch - differentiates between AutoEncoder and non-AutoEncoder
    '''
    train_loss, val_loss = 0, 0
    mse_train, mse_val = 0, 0

    model.train()

    if type(model) == AutoEncoder:
        for X, _, in train_loader:
            X = X.unsqueeze(dim=-1).to(device)
            loss, tr_predictions, _ = model.train_step(optimizer, criterion, X)
            train_loss += loss
            mse_train += mse_metric(tr_predictions, X).item()
        model.eval()
        with torch.no_grad():
            for X, _ in val_loader:
                X = X.unsqueeze(dim=-1).to(device)
                v_predictions, _ = model(X)
                val_loss += criterion(v_predictions, X).item()
                mse_val += mse_metric(v_predictions, X).item()

    else:
        for X, y in train_loader:
            X, y = X.unsqueeze(dim=-1).to(device), y.unsqueeze(dim=-1).to(device)
            loss, tr_predictions = model.train_step(optimizer, criterion, X, y)
            train_loss += loss
            mse_train += mse_metric(tr_predictions, y).item()
        model.eval()
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.unsqueeze(dim=-1).to(device), y.unsqueeze(dim=-1).to(device)
                v_predictions = model(X)
                val_loss += criterion(v_predictions, y).item()
                mse_val += mse_metric(v_predictions, y).item()
 
    return train_loss, mse_train, val_loss, mse_val


def test_model(test_loader, model, metric_fn):
    '''
    Test model. Differentiates between AutoEncoder and non-AutoEncoder models.

    test_loader: DataLoader, testing data.
    model: model to evaluate.
    metric_fn: the evaluation metric.
    '''
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    metric = []
    model.eval()

    if type(model) == AutoEncoder:
        for X, _, in test_loader:
            X = X.unsqueeze(dim=-1).to(device)
            reconstructed, _ = model(X)
            metric.append(metric_fn(reconstructed, X).item())

    else:
        for X, y in test_loader:
            X, y = X.unsqueeze(dim=2).to(device), y.unsqueeze(dim=2).to(device)
            predictions = model(X)
            metric.append(metric_fn(predictions, y).item())

    metric = torch.tensor(metric)
    mean, var = torch.mean(metric).item(), torch.var(metric).item()
    
    return mean, var


def save_training_logs(model, output, study_name, model_name):
        
    # Save the model's state dict
    model_save(model, f'model_logs/{study_name}/{model_name}')
    # Save the output
    pickle_dump(output, f'training_outputs/{study_name}/{model_name}')

    pass


