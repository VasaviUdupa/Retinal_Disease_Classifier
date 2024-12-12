import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from src.model import get_efficientnet_b0, get_criterion_and_optimizer
from src.utils import calculate_accuracy, save_model_checkpoint

# Paths
train_path = "/content/drive/My Drive/RetinalFundusImages/Retinal Fundus Images/train"
val_path = "/content/drive/My Drive/RetinalFundusImages/Retinal Fundus Images/val"
save_dir = "/content/drive/My Drive/RetinalFundusModels"
os.makedirs(save_dir, exist_ok=True)

# Data preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(train_path, transform=transform)
val_dataset = datasets.ImageFolder(val_path, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

# Model setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = len(train_dataset.classes)
class_counts = [1276, 545, 2294, 4982, 1635, 1295, 1369, 1220, 1142, 1678, 2641]  # Example class counts
model = get_efficientnet_b0(num_classes).to(device)
criterion, optimizer = get_criterion_and_optimizer(model, class_counts, lr=0.0001)

# Training loop
epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    print(f"Epoch {epoch + 1} Training Started")

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        # Log every 10 batches
        if (batch_idx + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}, Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")

    # Save checkpoint
    epoch_loss = running_loss / len(train_loader)
    checkpoint_path = os.path.join(save_dir, f"efficientnet_b0_epoch_{epoch + 1}.pth")
    save_model_checkpoint(model, optimizer, epoch, epoch_loss, checkpoint_path)
    print(f"Model saved to {checkpoint_path}")

    # Validation step
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            val_loss += criterion(outputs, labels).item()
            correct += calculate_accuracy(outputs, labels)
            total += labels.size(0)

    val_loss /= len(val_loader)
    accuracy = 100 * correct / total
    print(f"Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%")
