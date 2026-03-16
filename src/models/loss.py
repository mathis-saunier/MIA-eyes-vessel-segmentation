import torch
import torch.nn as nn

class DICE_BCE_Loss(nn.Module):

    def __init__(self, smooth=1):
        super().__init__()
        self.smooth = smooth # Paramètre de lissage pour éviter la division par zéro

    def forward(self, logits, targets):
        # Appliquer sigmoid pour le Dice seulement
        probs = torch.sigmoid(logits)
        targets = targets.float()

        intersection = 2 * (probs * targets).sum() + self.smooth
        union = probs.sum() + targets.sum() + self.smooth
        dice_loss = 1. - intersection / union

        loss = nn.BCEWithLogitsLoss()
        bce_loss = loss(logits, targets)

        return dice_loss + bce_loss