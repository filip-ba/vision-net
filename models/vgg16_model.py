import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix


class VGG16Model:
    def __init__(self):
        self.classes = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = None
        # VGG16 requires specific image size and normalization values
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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
        train_dataset = datasets.ImageFolder(root=f"{data_dir}/train", transform=self.transform)
        val_dataset = datasets.ImageFolder(root=f"{data_dir}/valid", transform=self.transform)
        test_dataset = datasets.ImageFolder(root=f"{data_dir}/test", transform=self.test_transform) 
        self.trainloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        self.valloader = DataLoader(val_dataset, batch_size=4, shuffle=False) 
        self.testloader = DataLoader(test_dataset, batch_size=4, shuffle=False)
        self.classes = train_dataset.classes
        self.dataset_loaded = True
        return len(train_dataset), len(val_dataset), len(test_dataset)

    def initialize_model(self):
        """Initializes the VGG16 model with transfer learning"""
        # Load pretrained VGG16
        self.net = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        # Freeze all layers except the last few
        for param in self.net.features.parameters():
            param.requires_grad = False
        # Replace the final classifier layer
        num_features = self.net.classifier[6].in_features
        self.net.classifier[6] = nn.Linear(num_features, 5)
        # Move to device
        self.net = self.net.to(self.device)
        # Loss function
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
        # Only optimize the classifier parameters
        self.optimizer = optim.SGD(self.net.classifier.parameters(), lr=learning_rate, momentum=momentum)
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
        # Store history
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

    def save_model(self, path):
        """Saves model state along with training parameters, metrics and history"""
        if self.net is None:
            raise ValueError("Model is not initialized")
        save_dict = {
            'model_state': self.net.state_dict(),
            'training_params': self.training_params,
            'metrics': self.metrics,
            'history': self.history,
        }
        torch.save(save_dict, path)

    def load_model(self, path):
        """Loads model state along with training parameters, metrics and history"""
        if self.net is None:
            self.initialize_model()     
        save_dict = torch.load(path, weights_only=False)
        # Load model state
        self.net.load_state_dict(save_dict['model_state'])
        self.net.eval()
        # Load metadata
        self.training_params = save_dict['training_params']
        self.metrics = save_dict['metrics']
        self.history = save_dict['history']
        return {
            'training_params': self.training_params,
            'metrics': self.metrics,
            'history': self.history
        }