import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms

from .base_model import BaseModel 


class VGG16Model(BaseModel):

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
        """Initializes the VGG16 model with transfer learning"""
        self.net = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        for param in self.net.features.parameters():
            param.requires_grad = False
        num_features = self.net.classifier[6].in_features
        
        num_classes = len(self.classes) if self.classes is not None else 5
        self.net.classifier[6] = nn.Linear(num_features, num_classes)
        
        self.net = self.net.to(self.device)
        self.optimizer = optim.SGD(self.net.classifier.parameters(), lr=0.001, momentum=0.9)