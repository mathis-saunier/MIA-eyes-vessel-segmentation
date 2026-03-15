import torch
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score

def predict(model, loader, device="cpu"):
    model.eval()
    preds = []

    f1_scores = []
    precision_scores = []
    recall_scores = []

    with torch.no_grad():
        for images, labels, _ in tqdm(loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs = torch.sigmoid(outputs)
            masks = (probs > 0.5)
            preds.append(masks)

            # Calcul du F1-score pour ce batch
            labels_binary = (labels > 0.5).cpu().numpy().flatten()
            masks_binary = masks.cpu().numpy().flatten()
            f1 = f1_score(labels_binary, masks_binary)
            f1_scores.append(f1)

            # Calcul du score de précision pour ce batch
            precision = precision_score(labels_binary, masks_binary)
            precision_scores.append(precision)

            # Calcul du score de rappel pour ce batch
            recall = recall_score(labels_binary, masks_binary)
            recall_scores.append(recall)

            # Libérer la mémoire GPU
            del outputs, probs, masks
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        return torch.cat(preds, dim=0), f1_scores, precision_scores, recall_scores