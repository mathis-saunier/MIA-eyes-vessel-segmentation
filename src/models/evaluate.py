import torch
from tqdm import tqdm

def evaluate(model, loader, criterion, device):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, labels, _ in tqdm(loader):
            images = images.to(device)
            labels = labels.to(device)

            for image_idx in range(images.size(0)):
                image = images[image_idx:image_idx + 1]
                label = labels[image_idx:image_idx + 1]

                outputs = model(image)
                loss = criterion(outputs, label)
                val_loss += loss.item()

                del image, label, outputs, loss

            # Libérer la mémoire GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
    return val_loss / len(loader.dataset)