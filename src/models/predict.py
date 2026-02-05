import torch
from tqdm import tqdm

def predict(model, loader, device="cpu"):
    model.eval()
    preds = []

    with torch.no_grad():
        for images, labels, _ in tqdm(loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs = torch.sigmoid(outputs)
            masks = (probs > 0.5).float()
            preds.append(masks)

            # Libérer la mémoire GPU
            del outputs, probs, masks
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        return torch.cat(preds, dim=0)