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


def save_checkpoint(model, optimizer, train_losses, val_losses, checkpoint_path):
    """Sauvegarde le modèle, l'optimizer et les historiques de pertes."""
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint sauvegardé à: {checkpoint_path}")


def load_checkpoint(checkpoint_path, model, optimizer, device="cpu"):
    """Charge le modèle, l'optimizer et les historiques de pertes."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Le checkpoint n'existe pas: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    train_losses = checkpoint['train_losses']
    val_losses = checkpoint['val_losses']
    
    print(f"Checkpoint chargé depuis: {checkpoint_path}")
    print(f"Epochs déjà entraînés: {len(train_losses)}")
    return model, optimizer, train_losses, val_losses


def get_model_from_config(config, device="cpu"):
    in_channels = config['model']['in_channels']
    out_channels = config['model']['out_channels']
    
    model = UNet(in_channels, out_channels).to(device)
    return model


def list_available_checkpoints(save_dir="../saved_models/"):
    """Liste tous les checkpoints disponibles."""
    if not os.path.exists(save_dir):
        print(f"Le dossier {save_dir} n'existe pas.")
        return []
    
    checkpoints = []
    for dirname in sorted(os.listdir(save_dir)):
        dirpath = os.path.join(save_dir, dirname)
        if os.path.isdir(dirpath):
            checkpoint_path = os.path.join(dirpath, 'checkpoint.pth')
            if os.path.exists(checkpoint_path):
                try:
                    checkpoint = torch.load(checkpoint_path, map_location='cpu')
                    epochs = len(checkpoint['train_losses'])
                    checkpoints.append({
                        'name': dirname,
                        'path': checkpoint_path,
                        'epochs': epochs
                    })
                except Exception as e:
                    print(f"Erreur lors de la lecture de {dirname}: {e}")
    
    return checkpoints
