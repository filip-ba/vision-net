import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms

from .base_model import BaseModel 


class EfficientNetModel(BaseModel):

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
        """Initializes the EfficientNet-B0 model with transfer learning"""
        self.net = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        for name, param in self.net.named_parameters():
            if "features.7" not in name and "classifier" not in name:
                param.requires_grad = False    
        num_ftrs = self.net.classifier[1].in_features
        
        num_classes = len(self.classes) if self.classes is not None else 6
        self.net.classifier[1] = nn.Linear(num_ftrs, num_classes)
        
        self.net = self.net.to(self.device)
        trainable_params = [p for p in self.net.parameters() if p.requires_grad]
        self.optimizer = optim.SGD(trainable_params, lr=0.001, momentum=0.9)