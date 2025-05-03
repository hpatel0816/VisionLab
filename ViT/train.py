import torch
import torch.optim as optim
from tqdm import tqdm

from model import ViTImageClassifier
from dataset import create_mnist_dataset

MODEL_CONFIG = {
        "batch_size": 128,
        "image_size": 28,
        "lr": 1e-4,
        "d_model": 256,
        "num_epochs": 15,
        "model_folder": "weights",
        "model_filename": "ViTMNISTmodel_",
        "experiment_name": "runs/ViTMNISTmodel",
        "hidden_dims": 256,
        "upsample_mlp_dims": 512,
        "patch_size": 7,
        "num_channels": 1, 
        "dropout": 0.1,
        "num_attn_blocks": 6,
        "num_attn_heads": 4,
        "num_labels": 10
    }

# Define training parameters
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") # This is for apple silicon
num_epochs = MODEL_CONFIG["num_epochs"]
learning_rate = MODEL_CONFIG["lr"]
save_path = "models/best_vit_model.pth"

# Initialize model, loss function, and optimizer
model = ViTImageClassifier(MODEL_CONFIG).to(device)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

# Training and validation loops
def train(model, train_loader, val_loader, optimizer, num_epochs, save_path):
    best_val_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0
        
        loop = tqdm(train_loader, leave=True)
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            loss, outputs = model(images, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
            loop.set_postfix(loss=loss.item(), acc=100 * correct / total)
        
        train_acc = 100 * correct / total
        val_acc = validate(model, val_loader)
        
        print(f"Epoch {epoch+1}: Train Loss: {running_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"Best model saved at epoch: {epoch}")

def validate(model, val_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs[0], 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# Run training
train_data_loader, val_data_loader = create_mnist_dataset(batch_size=MODEL_CONFIG["batch_size"], shuffle=True)
train(model, train_data_loader, val_data_loader, optimizer, num_epochs, save_path)