import torch

def calculate_f1_tensor(outputs, masks, epsilon=1e-8):
    outputs = torch.sigmoid(outputs)
    outputs = outputs.view(-1)
    masks = masks.view(-1)

    intersection = (outputs * masks).sum()
    dice = (2. * intersection + epsilon) / (outputs.sum() + masks.sum() + epsilon)

    return dice
