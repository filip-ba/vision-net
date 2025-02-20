import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from models.base_model import BaseModel


class ResNetModel(BaseModel):
    def get_transforms(self):
        train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])  
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])    
        return train_transform, test_transform

    def initialize_model(self):
        """Initializes the ResNet model with transfer learning"""
        self.net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Freeze all layers
        for param in self.net.parameters():
            param.requires_grad = False
        # Replace the final fully connected layer
        num_features = self.net.fc.in_features
        self.net.fc = nn.Linear(num_features, 5)
        # Move to device
        self.net = self.net.to(self.device)
        # Only optimize the final layer parameters
        self.optimizer = optim.SGD(self.net.fc.parameters(), lr=0.001, momentum=0.9)