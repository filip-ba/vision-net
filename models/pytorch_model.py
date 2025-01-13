import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
import numpy as np
from typing import Tuple
import os
from models.model_interface import ModelInterface

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
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

class PyTorchModel(ModelInterface):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Net().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    def _load_data(self):
        """Load and split dataset"""
        dataset = datasets.ImageFolder(root="models/dicetoss_small", transform=self.transform)
        train_size = int(0.7 * len(dataset))
        val_size = int(0.15 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size]
        )
        
        return (
            DataLoader(train_dataset, batch_size=4, shuffle=True),
            DataLoader(val_dataset, batch_size=4, shuffle=True),
            DataLoader(test_dataset, batch_size=4, shuffle=False)
        )
    
    def train(self, epochs=10) -> dict:
        """Train the model and return training history"""
        trainloader, valloader, _ = self._load_data()
        
        train_loss_history = []
        val_loss_history = []
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            for inputs, labels in trainloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, labels in valloader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    val_loss += loss.item()
            
            # Record losses
            train_loss_history.append(train_loss / len(trainloader))
            val_loss_history.append(val_loss / len(valloader))
            
        return {
            'train_loss': train_loss_history,
            'val_loss': val_loss_history
        }
    
    def test(self) -> Tuple[np.ndarray, float]:
        """Test the model and return confusion matrix and accuracy"""
        _, _, testloader = self._load_data()
        
        self.model.eval()
        y_pred = []
        y_true = []
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                y_pred.extend(predicted.cpu().numpy())
                y_true.extend(labels.cpu().numpy())
        
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred)
        accuracy = correct / total
        
        return cm, accuracy
    
    def predict(self, image: np.ndarray) -> Tuple[str, float]:
        """Predict class for single image"""
        self.model.eval()
        
        # Convert numpy image to tensor
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # Add batch dimension and move to device
        image_tensor = transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            predicted_class = torch.argmax(outputs, dim=1).item()
            confidence = probabilities[0][predicted_class].item() * 100
        
        # Convert class index to letter (A-F)
        predicted_letter = chr(65 + predicted_class)
        
        return predicted_letter, confidence
    
    def save(self, path: str) -> None:
        """Save model weights to file"""
        if not path.endswith('.pth'):
            path += '.pth'
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
    
    def load(self, path: str) -> None:
        """Load model weights from file"""
        if not path.endswith('.pth'):
            path += '.pth'
        
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.model.eval()