import torch
import torch.nn as nn
import torch.nn.functional as F

class OrdinalLoss(nn.Module):
    """
    Ordinal Regression Loss based on SISMO paper [cite: 407]
    Cost function: phi(r_t, r_i) = alpha * |r_t - r_i|
    """
    def __init__(self, alpha=1.5, num_classes=4, device='cpu'):
        super(OrdinalLoss, self).__init__()
        self.alpha = alpha
        self.num_classes = num_classes
        self.device = device
        
        # Precompute the Cost Matrix (distance matrix)
        # This avoids recalculating during training, speeding up computation
        self.cost_matrix = torch.zeros((num_classes, num_classes)).to(device)
        for i in range(num_classes):
            for j in range(num_classes):
                self.cost_matrix[i, j] = abs(i - j) # |r_t - r_i| 

    def forward(self, logits, targets):
        """
        logits: model raw outputs (batch_size, num_classes)
        targets: true labels (batch_size)
        """
        # TODO: Implement the ordinal loss calculation using the cost matrix
        # Main logic: compute Soft Labels -> Cross Entropy
        pass