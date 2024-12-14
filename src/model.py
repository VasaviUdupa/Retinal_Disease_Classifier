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


def get_criterion_and_optimizer(model, class_counts, lr=0.0001):
    """
    Initializes the loss function (CrossEntropyLoss with weights) and optimizer (AdamW).
    
    Args:
        model (nn.Module): The model whose parameters are to be optimized.
        class_counts (list): List of sample counts for each class to compute weights.
        lr (float): Learning rate for the optimizer.

    Returns:
        tuple: (criterion, optimizer)
    """
    # Calculate class weights for imbalance handling
    total_count = sum(class_counts)
    weights = [total_count / c for c in class_counts]
    weights = torch.tensor(weights, dtype=torch.float32)
    weights /= weights.sum()  # Normalize weights

    # Define CrossEntropyLoss with weights
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    return criterion, optimizer