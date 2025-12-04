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

        if torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using MPS (Apple Silicon GPU)")
        
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
        logits: (B, C)
        targets: (B,)
        """
        # Convert logits to probabilities
        probs = F.softmax(logits, dim=1)  # (B, C)

        # Cost vectors per sample: φ(target, i) = |target - i|
        # shape: (B, C)
        cost_vec = self.cost_matrix[targets]  

        # Soft labels: exp(-alpha * |t - i|)
        soft_labels = torch.exp(-self.alpha * cost_vec)
        soft_labels = soft_labels / soft_labels.sum(dim=1, keepdim=True)

        # Cross entropy with soft labels
        loss = torch.sum(-soft_labels * torch.log(probs + 1e-12), dim=1)

        return loss.mean()