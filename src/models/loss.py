import torch
import torch.nn as nn

class DICE_BCE_Loss(nn.Module):
    """Définit une fonction de perte combinée DICE et BCE pour la segmentation sémantique."""

    def __init__(self, smooth=1):
        """Initialise la fonction de perte avec un paramètre de lissage."""
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """Calcule la perte DICE et BCE entre les logits et les cibles.
        
        Args:
            logits (torch.Tensor): Les prédictions du modèle (logits).
            targets (torch.Tensor): Les cibles réelles (masque).
        Returns:
            torch.Tensor: La perte combinée DICE et BCE."""
        # Appliquer sigmoid pour le Dice seulement
        probs = torch.sigmoid(logits)
        targets = targets.float()

        intersection = 2 * (probs * targets).sum() + self.smooth
        union = probs.sum() + targets.sum() + self.smooth
        dice_loss = 1. - intersection / union

        loss = nn.BCEWithLogitsLoss()
        bce_loss = loss(logits, targets)

        return dice_loss + bce_loss