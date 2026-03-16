import torch
from tqdm import tqdm

def predict(model, loader, device="cpu", verbose="False"):
    model.eval()
    preds = []

    f1_scores = []
    precisions = []
    recalls = []

    with torch.no_grad():
        for images, labels, desease, quality in tqdm(loader):
            
            if verbose:
                for i in range(images.size(0)):
                    ic = quality["IC"][i].item()
                    blur = quality["Blur"][i].item()
                    lc = quality["LC"][i].item()
                    d = desease[i]
                    tqdm.write(f"  Image {len(preds)+i+1} | Maladie: {d} | IC: {ic} | Blur: {blur} | LC: {lc} | Somme: {ic+blur+lc}")

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs = torch.sigmoid(outputs)
            masks = (probs > 0.5)
            preds.append(masks)

            labels_binary = (labels > 0.5).cpu().numpy().flatten()
            masks_binary = masks.cpu().numpy().flatten()
            
            true_positives = (labels_binary == 1) & (masks_binary == 1)
            false_positives = (labels_binary == 0) & (masks_binary == 1)
            false_negatives = (labels_binary == 1) & (masks_binary == 0)
            true_negatives = (labels_binary == 0) & (masks_binary == 0)
            
            f1 = 2 * true_positives.sum() / (2 * true_positives.sum() + false_positives.sum() + false_negatives.sum() + 1e-8)
            precision = true_positives.sum() / (true_positives.sum() + false_positives.sum() + 1e-8)
            recall = true_positives.sum() / (true_positives.sum() + false_negatives.sum() + 1e-8)
            
            f1_scores.append(f1)
            precisions.append(precision)
            recalls.append(recall)
            
            # Libérer la mémoire GPU
            del outputs, probs, masks
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        return torch.cat(preds, dim=0), f1_scores, precisions, recalls