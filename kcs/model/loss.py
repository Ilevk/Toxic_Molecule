import torch
from torch import nn
from torch.nn import functional as F

def F1_BCE_Loss(y_pred, y_true):
    assert y_pred.ndim == 2
    assert y_true.ndim == 1
    epsilon = 1e-7
    y_true = F.one_hot(y_true, 2).to(torch.float32)
    y_pred = F.softmax(y_pred, dim=1)

    tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    f1 = f1.clamp(min=epsilon, max=1 - epsilon)

    bce = F.binary_cross_entropy(y_pred, y_true)
    return 1 - f1.mean() + bce