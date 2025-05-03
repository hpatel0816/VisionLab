import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset

class MNISTDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = sample["image"]
        label = sample["label"]

        if self.transform:
            image = self.transform(image)

        return image, label


def create_mnist_dataset(batch_size, shuffle=True):
    mnist = load_dataset("mnist")

    # MNIST images are 1-channel (grayscale) and need to normalized accordingly
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.1307], std=[0.3081]),
    ])

    train_dataset = MNISTDataset(mnist["train"], transform=transform)
    test_dataset = MNISTDataset(mnist["test"], transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
