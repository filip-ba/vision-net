import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import numpy as np
import torchvision
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix


class DiceTossModel:
    def __init__(self):
        self.classes = ("A", "B", "C", "D", "E", "F")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = None
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)), 
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.test_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def load_data(self, data_dir):
        """Loads and prepares dataset"""
        full_dataset = datasets.ImageFolder(root=data_dir, transform=self.transform)
        train_size = int(0.7 * len(full_dataset))
        val_size = int(0.15 * len(full_dataset))
        test_size = len(full_dataset) - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size]
        )  
        self.trainloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        self.valloader = DataLoader(val_dataset, batch_size=4, shuffle=True)
        self.testloader = DataLoader(test_dataset, batch_size=4, shuffle=False)
        return len(train_dataset), len(val_dataset), len(test_dataset)

    def initialize_model(self):
        """Initializes the model"""
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

        self.net = Net().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None  # To be set during training

    def train(self, epochs, learning_rate, momentum, progress_callback=None):
        """Trains the model and returns the loss history"""
        if self.net is None:
            raise ValueError("Model is not initialized")
        self.optimizer = optim.SGD(self.net.parameters(), lr=learning_rate, momentum=momentum)
        train_loss_history = []
        val_loss_history = []
        for epoch in range(epochs):
            train_loss = 0.0
            val_loss = 0.0
            # Training
            self.net.train()
            for i, (inputs, labels) in enumerate(self.trainloader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                
                if progress_callback:
                    progress = (epoch * len(self.trainloader) + i) / (epochs * len(self.trainloader))
                    progress_callback(progress, train_loss / (i + 1))
            # Validation
            self.net.eval()
            with torch.no_grad():
                for inputs, labels in self.valloader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.net(inputs)
                    loss = self.criterion(outputs, labels)
                    val_loss += loss.item()
            train_loss_history.append(train_loss / len(self.trainloader))
            val_loss_history.append(val_loss / len(self.valloader))
        return train_loss_history, val_loss_history

    def test(self):
        """Tests the model and returns metrics"""
        if self.net is None:
            raise ValueError("Model is not initialized")
        self.net.eval()
        y_pred = []
        y_true = []
        with torch.no_grad():
            for images, labels in self.testloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.net(images)
                _, predicted = torch.max(outputs, 1)
                y_pred.extend(predicted.cpu().numpy())
                y_true.extend(labels.cpu().numpy())
        conf_mat = confusion_matrix(y_true, y_pred)
        precision, recall, _, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )  
        accuracy = (np.array(y_pred) == np.array(y_true)).mean()
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'confusion_matrix': conf_mat
        }

    def save_model(self, path):
        if self.net is None:
            raise ValueError("Model is not initialized")
        torch.save(self.net.state_dict(), path)

    def load_model(self, path):
        if self.net is None:
            self.initialize_model()
        self.net.load_state_dict(torch.load(path))
        self.net.eval()

    def predict_image(self, image_path):
        """Predicts class for one image"""
        if self.net is None:
            raise ValueError("Model is not initialized")
        # Load and transform an image
        image = torchvision.io.read_image(image_path).float()
        image = self.test_transform(image).unsqueeze(0).to(self.device)
        # Prediction
        self.net.eval()
        with torch.no_grad():
            outputs = self.net(image)
            _, predicted = torch.max(outputs, 1)
            probabilities = F.softmax(outputs, dim=1)
        return {
            'class': self.classes[predicted.item()],
            'probabilities': probabilities[0].cpu().numpy()
        }