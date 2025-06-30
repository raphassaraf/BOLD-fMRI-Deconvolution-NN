# BOLD fMRI deconvolution: a neural network based appraoch
This repository contains the work undertaken at EPFL's Medical Image Processing in the scope of a semester project. The aim was to develop neural networks to perform task-based fMRI deconvolution using CNN, RNN and AutoEncoder models.

## Contributors
* Raphael Assaraf - raphael.assaraf@epfl.ch

## Introduction
This project contains all the code used to generate the results and plots of the _BOLD fMRI deconvolution: a neural network based appraoch_ semester project report.
The work was done in the scope of a semester project at the Medical Image Processing laboratory at the Swiss Federal Institute of Technology of Lausanne (EPFL).

## Project Structure
* `/configs/*` - Contains config files for the .py scripts.
* `/data_generation/` - Contains scripts used to generate real and simulated datasets
    - `mixed_data_generation.py` - Generate mixed (real + simulated) datasets of various proportions of real data.
    - `real_data_generation.py` - Generate real dataset.
    - `simulated_data_generation.py` - Generate simulated dataset.
* `/notebooks/` - Contains notebooks used for analyis.
    - `hp_search_results_analysis.ipynb`
    - `midterm_pres_plots.ipynb`
    - `report_plots.ipynb`
* `/tools/`
    - `./models/*` - Contains all neural network model classes.
    - `analysis.py` - Contains all functions used for plots and results analysis.
    - `custom_loss.py` - Loss functions.
    - `objective.py` - Optuna objectives for hyperparameter searches.
    - `preprocessing_functions.py` - Base preprocessing functions (including data augmentation functions).
    - `preprocessing_script.py` - Wrapper functions to preprocess real datasets.
    - `simulated_data.py` -  Functions to generate simulated datasets.
    - `training_functions.py` - Contains the main training loops.
    - `utils.py` - Utility functions.
* `cnn_small.py` - Train & finetune CNN-small model.
* `ft_ae.py` - Finetune AE model using different strategies.
* `hp_search.py` - Hyperparameter search script.
* `mixed_data_training.py` - Script for training models on mixed datasets with varying proportions of real data.
* `requirements.txt`


TODO: ADD requirements.txt