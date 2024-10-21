import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import seaborn as sns

# Define the dataset directory
data_dir = "./models/dicetoss_small"

# Define transformations
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),  
    transforms.RandomRotation(10),  
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  
])

# Load dataset and split it
def load_datasets():
    full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

    trainloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    valloader = DataLoader(val_dataset, batch_size=4, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    return trainloader, valloader, testloader

classes = ("A", "B", "C", "D", "E", "F")

# Neural Network definition
class DiceNet(nn.Module):
    def __init__(self):
        super(DiceNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 6)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Function to train the model
def train_model(net, trainloader, valloader, epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    train_loss_history = []
    val_loss_history = []

    for epoch in range(epochs):
        train_loss = 0.0
        val_loss = 0.0

        net.train()
        for inputs, labels in trainloader:
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss_history.append(train_loss / len(trainloader))

        net.eval()
        with torch.no_grad():
            for inputs, labels in valloader:
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        val_loss_history.append(val_loss / len(valloader))

        print(f"Epoch {epoch + 1}/{epochs} - Train loss: {train_loss / len(trainloader):.4f} - Val loss: {val_loss / len(valloader):.4f}")

    # Plot the loss history
    plt.figure(figsize=(8, 6))
    plt.plot(train_loss_history, label="Training Loss")
    plt.plot(val_loss_history, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss History")
    plt.legend()
    plt.show()
