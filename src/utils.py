import torch

def calculate_accuracy(outputs, labels):
    """
    Calculate accuracy of predictions.
    """
    _, preds = torch.max(outputs, 1)
    correct = (preds == labels).sum().item()
    return correct

def save_model_checkpoint(model, optimizer, epoch, loss, path):
    """
    Save the model checkpoint.
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)

def load_model_checkpoint(model, optimizer, path, device):
    """
    Load the model checkpoint.
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']
