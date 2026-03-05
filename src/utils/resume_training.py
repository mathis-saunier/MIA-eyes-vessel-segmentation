# Chargement du modèle sauvegardé
import torch
import yaml
import os
from src.models.models import UNet

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_loss_name(loss):
    """Retourne un identifiant stable pour une loss (instance ou chaîne)."""
    if loss is None:
        return None
    if isinstance(loss, str):
        return loss
    return loss.__class__.__name__


def load_previous_model(path, model=None, in_channels=3, out_channels=1, device="cpu"):
    if model is None:
        model = UNet(in_channels, out_channels).to(device)
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Le fichier modèle n'existe pas: {path}")
    
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    print(f"Modèle chargé depuis: {path}")
    return model


def save_checkpoint(model, optimizer, train_losses, val_losses, checkpoint_path, loss_name=None):
    """Sauvegarde le modèle, l'optimizer et les historiques de pertes."""
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'loss_name': get_loss_name(loss_name),
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint sauvegardé à: {checkpoint_path}")


def load_checkpoint(checkpoint_path, model, optimizer, device="cpu", loss_name=None, enforce_loss_check=True):
    """Charge le modèle, l'optimizer et les historiques de pertes."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Le checkpoint n'existe pas: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    saved_loss_name = checkpoint.get('loss_name')
    current_loss_name = get_loss_name(loss_name)

    print(f"Ancienne loss: {saved_loss_name}, Loss actuelle: {current_loss_name}")

    if enforce_loss_check:
        if saved_loss_name is None:
            raise ValueError(
                "Le checkpoint ne contient pas d'information de loss. "
                "Impossible de vérifier la compatibilité de la loss utilisée."
            )
        if current_loss_name is None:
            raise ValueError(
                "Aucune loss fournie pour la reprise. "
                "Passez loss_name=... à load_checkpoint pour vérifier la compatibilité."
            )
        if saved_loss_name != current_loss_name:
            raise ValueError(
                f"Incompatibilité de loss: checkpoint='{saved_loss_name}', "
                f"actuelle='{current_loss_name}'"
            )

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    train_losses = checkpoint['train_losses']
    val_losses = checkpoint['val_losses']
    
    print(f"Checkpoint chargé depuis: {checkpoint_path}")
    if saved_loss_name is not None:
        print(f"Loss du checkpoint: {saved_loss_name}")
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
                        'epochs': epochs,
                        'loss_name': checkpoint.get('loss_name')
                    })
                except Exception as e:
                    print(f"Erreur lors de la lecture de {dirname}: {e}")
    
    return checkpoints
