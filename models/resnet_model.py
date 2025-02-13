import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, random_split
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from PIL import Image

class ResNetModel:
    def __init__(self):
        self.classes = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = None
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.trainloader = None
        self.valloader = None
        self.testloader = None
        self.dataset_loaded = False
        self.training_params = {'epochs': None, 'learning_rate': None, 'momentum': None}
        self.metrics = {'accuracy': None, 'precision': None, 'recall': None, 'confusion_matrix': None}
        self.history = {'train_loss': None, 'val_loss': None}

    def reset_metrics(self):
        self.metrics = {'accuracy': None, 'precision': None, 'recall': None, 'confusion_matrix': None}

    def is_data_loaded(self):
        return self.dataset_loaded

    def load_data(self, data_dir):
        if not self.dataset_loaded:
            full_dataset = datasets.ImageFolder(root=data_dir, transform=self.transform)
            self.classes = full_dataset.classes
            train_size = int(0.7 * len(full_dataset))
            val_size = int(0.15 * len(full_dataset))
            test_size = len(full_dataset) - train_size - val_size
            train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])
            self.trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            self.valloader = DataLoader(val_dataset, batch_size=32, shuffle=True)
            self.testloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
            self.dataset_loaded = True
            return len(train_dataset), len(val_dataset), len(test_dataset)
        return len(self.trainloader.dataset), len(self.valloader.dataset), len(self.testloader.dataset)

    def initialize_model(self):
        if self.classes is None:
            raise ValueError("Load dataset before initializing the model.")
        self.net = models.resnet18(pretrained=True)
        num_ftrs = self.net.fc.in_features
        self.net.fc = nn.Linear(num_ftrs, len(self.classes))
        self.net = self.net.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None

    def train(self, epochs, lr, momentum, progress_callback=None):
        if not self.net or not self.dataset_loaded:
            raise ValueError("Model or dataset is not initialized.")
        self.optimizer = optim.SGD(self.net.parameters(), lr=lr, momentum=momentum)
        self.training_params = {'epochs': epochs, 'learning_rate': lr, 'momentum': momentum}
        train_loss, val_loss = [], []
        for epoch in range(epochs):
            self.net.train()
            epoch_loss = 0.0
            for i, (inputs, labels) in enumerate(self.trainloader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                if progress_callback:
                    progress = (epoch * len(self.trainloader) + i) / (epochs * len(self.trainloader))
                    progress_callback(progress, epoch_loss/(i+1))
            train_loss.append(epoch_loss/len(self.trainloader))
            # Validace
            self.net.eval()
            val_loss_epoch = 0.0
            with torch.no_grad():
                for inputs, labels in self.valloader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.net(inputs)
                    val_loss_epoch += self.criterion(outputs, labels).item()
            val_loss.append(val_loss_epoch/len(self.valloader))
        self.history = {'train_loss': train_loss, 'val_loss': val_loss}
        return train_loss, val_loss

    def test(self):
        y_true, y_pred = [], []
        self.net.eval()
        with torch.no_grad():
            for inputs, labels in self.testloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.net(inputs)
                _, predicted = torch.max(outputs, 1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
        cm = confusion_matrix(y_true, y_pred)
        precision, recall, _, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
        self.metrics = {
            'accuracy': np.mean(np.array(y_true) == np.array(y_pred)),
            'precision': precision,
            'recall': recall,
            'confusion_matrix': cm
        }
        return self.metrics

    def save_model(self, path):
        torch.save({
            'model_state': self.net.state_dict(),
            'training_params': self.training_params,
            'metrics': self.metrics,
            'history': self.history,
            'classes': self.classes
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.classes = checkpoint['classes']
        if not self.net:
            self.initialize_model()
        self.net.load_state_dict(checkpoint['model_state'])
        self.training_params = checkpoint['training_params']
        self.metrics = checkpoint['metrics']
        self.history = checkpoint['history']

    def predict_image(self, image_path):
        with Image.open(image_path) as img:
            img = img.convert('RGB')
            img = self.test_transform(img).unsqueeze(0).to(self.device)
        self.net.eval()
        with torch.no_grad():
            outputs = self.net(img)
            probs = torch.nn.functional.softmax(outputs, dim=1)[0]
            _, pred = torch.max(outputs, 1)
        return {'class': self.classes[pred.item()], 'probabilities': probs.cpu().numpy()}