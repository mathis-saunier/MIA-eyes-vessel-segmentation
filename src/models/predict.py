import torch
from tqdm import tqdm
from sklearn.metrics import f1_score


def _split_batch_metadata(batch_metadata, batch_size):
    if isinstance(batch_metadata, list):
        return batch_metadata

    if isinstance(batch_metadata, dict):
        metadata_list = []
        for sample_idx in range(batch_size):
            metadata = {}
            for key, value in batch_metadata.items():
                if isinstance(value, torch.Tensor):
                    item = value[sample_idx]
                    if item.dim() == 0:
                        metadata[key] = item.item()
                    else:
                        metadata[key] = tuple(int(v.item()) for v in item)
                elif isinstance(value, (list, tuple)):
                    if len(value) > 0 and all(isinstance(v, torch.Tensor) for v in value):
                        if len(value) == batch_size:
                            item = value[sample_idx]
                            if item.dim() == 0:
                                metadata[key] = item.item()
                            else:
                                metadata[key] = tuple(int(v.item()) for v in item)
                        else:
                            metadata[key] = tuple(int(v[sample_idx].item()) for v in value)
                    else:
                        item = value[sample_idx]
                        if isinstance(item, (list, tuple)):
                            metadata[key] = tuple(int(v) for v in item)
                        else:
                            metadata[key] = item
                else:
                    metadata[key] = value

            metadata_list.append(metadata)

        return metadata_list

    return [None for _ in range(batch_size)]

def predict(model, loader, device="cpu"):
    model.eval()
    patch_preds = []
    stitched_predictions = {}
    stitched_labels = {}
    original_sizes = {}
    image_order = []

    f1_scores = []

    with torch.no_grad():
        for images, labels, metadata_batch in tqdm(loader):
            images = images.to(device)
            labels = labels.to(device)

            batch_size = images.size(0)
            metadata_list = _split_batch_metadata(metadata_batch, batch_size)

            for image_idx in range(batch_size):
                image = images[image_idx:image_idx + 1]
                label = labels[image_idx:image_idx + 1]
                metadata = metadata_list[image_idx]

                outputs = model(image)
                probs = torch.sigmoid(outputs)
                mask = (probs > 0.5).to("cpu")
                patch_preds.append(mask)

                # Calcul du F1-score pour cette imagette
                labels_binary = (label > 0.5).to("cpu").numpy().flatten()
                masks_binary = mask.numpy().flatten()
                f1 = f1_score(labels_binary, masks_binary)
                f1_scores.append(f1)

                if isinstance(metadata, dict) and "top_left" in metadata and "padded_size" in metadata and "original_size" in metadata:
                    image_path = metadata["image_path"]
                    left, top = metadata["top_left"]
                    padded_width, padded_height = metadata["padded_size"]
                    original_width, original_height = metadata["original_size"]
                    patch_height, patch_width = mask.shape[-2], mask.shape[-1]

                    if image_path not in stitched_predictions:
                        stitched_predictions[image_path] = torch.zeros((padded_height, padded_width), dtype=torch.bool)
                        stitched_labels[image_path] = torch.zeros((padded_height, padded_width), dtype=torch.bool)
                        original_sizes[image_path] = (original_width, original_height)
                        image_order.append(image_path)

                    stitched_predictions[image_path][top:top + patch_height, left:left + patch_width] = mask.squeeze(0).squeeze(0)
                    stitched_labels[image_path][top:top + patch_height, left:left + patch_width] = (label > 0.5).to("cpu").squeeze(0).squeeze(0)

                del image, label, outputs, probs, mask

            # Libérer la mémoire GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if stitched_predictions:
        reconstructed_preds = []
        image_level_f1_scores = []

        for image_path in image_order:
            original_width, original_height = original_sizes[image_path]
            prediction_2d = stitched_predictions[image_path][:original_height, :original_width]
            label_2d = stitched_labels[image_path][:original_height, :original_width]

            prediction = prediction_2d.unsqueeze(0)
            reconstructed_preds.append(prediction)

            y_true = label_2d.numpy().flatten()
            y_pred = prediction_2d.numpy().flatten()
            image_level_f1_scores.append(f1_score(y_true, y_pred))

        return reconstructed_preds, image_level_f1_scores

    return torch.cat(patch_preds, dim=0), f1_scores