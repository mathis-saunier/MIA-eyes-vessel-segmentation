import torch
from tqdm import tqdm

def evaluate(model, loader, criterion, device):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, labels, _ in tqdm(loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            # Libérer la mémoire GPU
            del outputs, loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
    return val_loss / len(loader)