import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import torch

def get_efficientnet_b0(num_classes):
    """
    Initializes and returns an EfficientNet-B0 model with pretrained weights 
    and a modified classifier for the given number of classes.
    
    Args:
        num_classes (int): Number of output classes for classification.

    Returns:
        nn.Module: Modified EfficientNet-B0 model.
    """
    # Load pretrained EfficientNet-B0
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    
    # Replace the classifier
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    
    return model


def get_criterion_and_optimizer(model, class_counts, lr=0.0001, gamma=2):
    """
    Initializes the loss function (Focal Loss with weights) and optimizer (AdamW).
    
    Args:
        model (nn.Module): The model whose parameters are to be optimized.
        class_counts (list): List of sample counts for each class to compute weights.
        lr (float): Learning rate for the optimizer.
        gamma (int): Gamma parameter for Focal Loss.

    Returns:
        tuple: (criterion, optimizer)
    """
    # Calculate class weights for imbalance handling
    total_count = sum(class_counts)
    weights = [total_count / c for c in class_counts]
    weights = torch.tensor(weights, dtype=torch.float32)
    weights /= weights.sum()  # Normalize weights
    
    # Define Focal Loss
    class FocalLoss(nn.Module):
        def __init__(self, gamma=2, alpha=None):
            super(FocalLoss, self).__init__()
            self.gamma = gamma
            self.alpha = alpha

        def forward(self, inputs, targets):
            ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
            pt = torch.exp(-ce_loss)
            focal_loss = (1 - pt) ** self.gamma * ce_loss
            if self.alpha is not None:
                alpha_t = self.alpha[targets]
                focal_loss = alpha_t * focal_loss
            return focal_loss.mean()
    
    criterion = FocalLoss(gamma=gamma, alpha=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    return criterion, optimizer
