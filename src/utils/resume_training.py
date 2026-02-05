# Chargement du modèle sauvegardé
import torch
import yaml
import os
from src.models.models import UNet

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_previous_model(path, model=None, in_channels=3, out_channels=1, device="cpu"):
    if model is None:
        model = UNet(in_channels, out_channels).to(device)
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Le fichier modèle n'existe pas: {path}")
    
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    print(f"Modèle chargé depuis: {path}")
    return model


def get_model_from_config(config, device="cpu"):
    in_channels = config['model']['in_channels']
    out_channels = config['model']['out_channels']
    
    model = UNet(in_channels, out_channels).to(device)
    
    if config['training'].get('resume_training', False):
        resume_path = config['training'].get('resume_model_path')
        if resume_path and os.path.exists(resume_path):
            model = load_previous_model(resume_path, model=model, device=device)
            print("Reprise de l'entraînement depuis le checkpoint précédent.")
        else:
            print(f"Attention: resume_training est True mais le chemin '{resume_path}' n'existe pas.")
            print("Démarrage d'un nouvel entraînement.")
    else:
        print("Démarrage d'un nouvel entraînement.")
    
    return model
