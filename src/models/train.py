import os
import time
import torch
from tqdm import tqdm

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for images, labels, _ in tqdm(loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        # librer le gpu apres chaque batch
        del outputs, loss
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
    return total_loss / len(loader)

def train(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    epochs,
    patience,
    epsilon,
    dir_save=None,
    initial_train_losses=None,
    initial_val_losses=None,
    model_dir=None,
    ):
    from .evaluate import evaluate

    if dir_save is None:
        raise Warning("Il n'y a pas de chemin spécifié pour sauvegarder le modèle. Pas de sauvegarde effectuée.")
    else:
        # Vérifier que le dossier existe
        if not os.path.exists(os.path.dirname(dir_save)):
            raise FileNotFoundError(f"Le dossier pour sauvegarder le modèle n'existe pas : {os.path.dirname(dir_save)}")
        else:
            # Si model_dir n'est pas fourni, créer un nouveau dossier
            if model_dir is None:
                dirname = f"train_{time.strftime('%Y%m%d-%H%M%S')}"
                model_dir = os.path.join(dir_save, dirname)
            
            path_save = os.path.join(model_dir, 'best_model.pth')
            path_checkpoint = os.path.join(model_dir, 'checkpoint.pth')
            os.makedirs(model_dir, exist_ok=True)

    # Initialiser les historiques à partir des valeurs précédentes ou listes vides
    train_losses = initial_train_losses if initial_train_losses is not None else []
    val_losses = initial_val_losses if initial_val_losses is not None else []
    
    # Déterminer le min_val_loss à partir de l'historique
    max_val_loss = min(val_losses) if val_losses else float('inf')
    start_epoch = len(train_losses)

    for epoch in range(start_epoch, epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Save the best model
        if val_loss < max_val_loss:
            max_val_loss = val_loss
            if path_save is not None:
                torch.save(model.state_dict(), path_save)
        
        # Sauvegarder le checkpoint complet
        if path_checkpoint is not None:
            from ..utils.resume_training import save_checkpoint
            save_checkpoint(
                model,
                optimizer,
                train_losses,
                val_losses,
                path_checkpoint,
                loss_name=criterion,
            )
        
        # Early stopping patience
        if epoch > patience:
            if val_losses[-1] > min(val_losses[-(patience+1):-1]):
                print("Early stopping (patience)")
                break

        # Early stopping augmentation
        if epoch > 1:
            if val_losses[-1] > val_losses[-2] + epsilon:
                print("Early stopping (augmentation)")
                break

        print(f"Epoch {epoch+1}/{epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

    return train_losses, val_losses