import torch.nn as nn
from entities.utils import calculate_f1_tensor

class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        dice_loss = 1 - calculate_f1_tensor(inputs, targets)
        bce_loss = self.bce(inputs, targets)
        return bce_loss + dice_loss