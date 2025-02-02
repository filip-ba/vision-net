import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from PIL import Image


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
        # Dataset attributes
        self.trainloader = None
        self.valloader = None
        self.testloader = None
        self.dataset_loaded = False
        # Saving attributes
        self.training_params = {
            'epochs': None,
            'learning_rate': None,
            'momentum': None
        }
        self.metrics = {
            'accuracy': None,
            'precision': None,
            'recall': None,
            'confusion_matrix': None
        }
        self.history = {
            'train_loss': None,
            'val_loss': None
        }

    def reset_metrics(self):
        """Resets all metrics to their default state"""
        self.metrics = {
            'accuracy': None,
            'precision': None,
            'recall': None,
            'confusion_matrix': None
        }

    def is_data_loaded(self):
        """Check if dataset is loaded"""
        return self.dataset_loaded

    def load_data(self, data_dir):
        """Loads and prepares dataset if not already loaded"""
        if not self.dataset_loaded:
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
            self.dataset_loaded = True
            return len(train_dataset), len(val_dataset), len(test_dataset)
        return len(self.trainloader.dataset), len(self.valloader.dataset), len(self.testloader.dataset)

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
        if not self.dataset_loaded:
            raise ValueError("Dataset not loaded. Call load_data first.")   
        # Reset metrics before training
        self.reset_metrics()
        # Store training parameters
        self.training_params = {
            'epochs': epochs,
            'learning_rate': learning_rate,
            'momentum': momentum
        }    
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
        # Store history at the end of training
        self.history = {
            'train_loss': train_loss_history,
            'val_loss': val_loss_history
        }
        return train_loss_history, val_loss_history

    def test(self):
        """Tests the model and returns metrics"""
        if self.net is None:
            raise ValueError("Model is not initialized")
        if not self.dataset_loaded:
            raise ValueError("Dataset not loaded. Call load_data first.")
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
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'confusion_matrix': conf_mat
        }
        self.metrics = metrics
        return metrics

    def save_model(self, path):
        """Saves model state along with training parameters, metrics and history"""
        if self.net is None:
            raise ValueError("Model is not initialized")
        save_dict = {
            'model_state': self.net.state_dict(),
            'training_params': self.training_params,
            'metrics': self.metrics,
            'history': self.history,
            'classes': self.classes  # Save classes in case they change in future versions
        }
        torch.save(save_dict, path)

    def load_model(self, path):
        """Loads model state along with training parameters, metrics and history"""
        if self.net is None:
            self.initialize_model()    
        save_dict = torch.load(path)
        # Load model state
        self.net.load_state_dict(save_dict['model_state'])
        self.net.eval()
        # Load metadata
        self.training_params = save_dict['training_params']
        self.metrics = save_dict['metrics']
        self.history = save_dict['history']
        self.classes = save_dict['classes']
        return {
            'training_params': self.training_params,
            'metrics': self.metrics,
            'history': self.history
        }

    def predict_image(self, image_path):
        """Predicts class for one image"""
        if self.net is None:
            raise ValueError("Model is not initialized")
        try:
            with Image.open(image_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                image = self.test_transform(img).unsqueeze(0).to(self.device)
        except Exception as e:
            raise Exception(f"Error loading image: {str(e)}")
        self.net.eval()
        with torch.no_grad():
            outputs = self.net(image)
            _, predicted = torch.max(outputs, 1)
            probabilities = F.softmax(outputs, dim=1)
        return {
            'class': self.classes[predicted.item()],
            'probabilities': probabilities[0].cpu().numpy()
        }