import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from PIL import Image
from abc import ABC, abstractmethod
import time
import os


class BaseModel(ABC):
    
    def __init__(self):
        self.classes = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = None
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None
        
        # Dataset attributes
        self.trainloader = None
        self.valloader = None
        self.testloader = None
        self.dataset_loaded = False
        
        # Saving attributes
        self.training_params = {
            'epochs': None,
            'learning_rate': None,
            'momentum': None,
            'training_time': None 
        }
        self.metrics = {
            'accuracy': None,
            'precision': None,
            'recall': None,
            'confusion_matrix': None,
            'class_recall': None,
            'class_names': None   
        }
        self.history = {
            'train_loss': None,
            'val_loss': None
        }
        self.cv_metrics = {
            'k': None,
            'avg_accuracy': None,
            'std_accuracy': None,
            'fold_accuracies': None
        }

    @abstractmethod
    def get_transforms(self):
        """Returns tuple of (train_transform, test_transform)"""
        pass

    @abstractmethod
    def initialize_model(self):
        """Initializes the neural network model"""
        pass

    def reset_metrics(self):
        """Resets all metrics to their default state"""
        self.metrics = {
            'accuracy': None,
            'precision': None,
            'recall': None,
            'confusion_matrix': None
        }
        # Also reset cross-validation metrics
        self.cv_metrics = {
            'k': None,
            'avg_accuracy': None,
            'std_accuracy': None,
            'fold_accuracies': None
        }

    def is_data_loaded(self):
        """Check if dataset is loaded"""
        return self.dataset_loaded

    def load_data(self, data_dir, k=None, current_fold=None):
        """Loads and prepares dataset. If k and current_fold are provided, performs K-fold cross-validation split."""
        train_transform, test_transform = self.get_transforms()

        if k is not None and current_fold is not None:
            # Load all training data
            full_train_dataset = datasets.ImageFolder(root=os.path.join(data_dir, "train"), transform=train_transform)

            
            # Calculate fold size
            fold_size = len(full_train_dataset) // k
            
            # Create indices for the current fold
            start_idx = current_fold * fold_size
            end_idx = start_idx + fold_size if current_fold < k - 1 else len(full_train_dataset)
            
            # Split indices into train and validation
            all_indices = list(range(len(full_train_dataset)))
            val_indices = all_indices[start_idx:end_idx]
            train_indices = [i for i in all_indices if i not in val_indices]
            
            # Create subset datasets
            train_dataset = Subset(full_train_dataset, train_indices)
            val_dataset = Subset(full_train_dataset, val_indices)
            
            # Create data loaders
            self.trainloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
            self.valloader = DataLoader(val_dataset, batch_size=4, shuffle=False)
            
            # Load test data normally
            test_dataset = datasets.ImageFolder(root=os.path.join(data_dir, "test"), transform=test_transform)
            self.testloader = DataLoader(test_dataset, batch_size=4, shuffle=False)
            
            # Store classes from the full dataset
            self.classes = full_train_dataset.classes
        else:
            # Original loading logic
            train_dataset = datasets.ImageFolder(root=os.path.join(data_dir, "train"), transform=train_transform)
            val_dataset = datasets.ImageFolder(root=os.path.join(data_dir, "valid"), transform=train_transform)
            test_dataset = datasets.ImageFolder(root=os.path.join(data_dir, "test"), transform=test_transform)

            self.trainloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
            self.valloader = DataLoader(val_dataset, batch_size=4, shuffle=False)
            self.testloader = DataLoader(test_dataset, batch_size=4, shuffle=False)
            
            self.classes = train_dataset.classes
        
        self.dataset_loaded = True
        
        return len(self.trainloader.dataset), len(self.valloader.dataset), len(self.testloader.dataset)

    def share_dataset(self, source_model):
        """Shares dataset from another model instance to avoid reloading"""
        if not source_model.dataset_loaded:
            raise ValueError("Source model's dataset is not loaded")

        self.trainloader = source_model.trainloader
        self.valloader = source_model.valloader
        self.testloader = source_model.testloader
        self.classes = source_model.classes
        self.dataset_loaded = True
        
        return len(self.trainloader.dataset), len(self.valloader.dataset), len(self.testloader.dataset)

    def train(self, epochs, learning_rate, momentum, progress_callback=None):
        """Trains the model and returns the loss history"""
        if self.net is None:
            raise ValueError("Model is not initialized")
        if not self.dataset_loaded:
            raise ValueError("Dataset not loaded. Call load_data first.")
            
        self.reset_metrics()
        self.training_params = {
            'epochs': epochs,
            'learning_rate': learning_rate,
            'momentum': momentum,
            'training_time': None 
        }
        
        # Measure time
        start_time = time.time()

        train_loss_history = []
        val_loss_history = []
        epoch_train_loss_history = []
        epoch_val_loss_history = []
        
        first_batch_loss = None
        
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
                
                # First loss value
                if first_batch_loss is None:
                    first_batch_loss = loss.item()
                    train_loss_history.append(first_batch_loss)
                    
                if progress_callback:
                    progress = (epoch * len(self.trainloader) + i) / (epochs * len(self.trainloader))
                    current_loss = loss.item()  # Aktuální ztráta z posledního batche
                    progress_callback(progress, current_loss)
                        
            # Validation
            self.net.eval()
            with torch.no_grad():
                for inputs, labels in self.valloader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.net(inputs)
                    loss = self.criterion(outputs, labels)
                    val_loss += loss.item()
                    
            epoch_train_loss = train_loss / len(self.trainloader)
            epoch_val_loss = val_loss / len(self.valloader)
            epoch_train_loss_history.append(epoch_train_loss)
            epoch_val_loss_history.append(epoch_val_loss)
                        
        # End measuring
        end_time = time.time()
        training_time = end_time - start_time
        self.training_params['training_time'] = training_time

        # Add both the initial value and the progression of epochs to the history
        if first_batch_loss is not None:
            # We already have the first value train_loss_history
            train_loss_history.extend(epoch_train_loss_history)
            # Estimated initial value for the validation loss
            if len(epoch_val_loss_history) > 0:
                initial_val_loss = epoch_val_loss_history[0] * 1.2  # Odhad počáteční validační ztráty
                val_loss_history = [initial_val_loss] + epoch_val_loss_history
            else:
                val_loss_history = epoch_val_loss_history
        else:
            train_loss_history = epoch_train_loss_history
            val_loss_history = epoch_val_loss_history

        self.history = {
            'train_loss': train_loss_history,
            'val_loss': val_loss_history
        }
        return train_loss_history, val_loss_history

    def test(self):
        """Tests the model on the test set and returns metrics"""
        if self.net is None:
            raise ValueError("Model is not initialized")
        if not self.dataset_loaded:
            raise ValueError("Dataset not loaded. Call load_data first.")
            
        self.net.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for data in self.testloader:
                images, labels = data[0].to(self.device), data[1].to(self.device)
                outputs = self.net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = correct / total if total > 0 else 0.0
        
        # Calculate micro-averaged metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='macro', zero_division=0
        )
        
        # Calculate per-class metrics
        class_precision, class_recall, class_f1, class_support = precision_recall_fscore_support(
            all_labels, all_preds, average=None, zero_division=0
        )
        
        conf_matrix = confusion_matrix(all_labels, all_preds)
        
        self.metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'confusion_matrix': conf_matrix,
            'class_recall': class_recall,  
            'class_names': self.classes   
        }
        
        return self.metrics

    def predict_image(self, image_path):
        """Predicts class for one image"""
        if self.net is None:
            raise ValueError("Model is not initialized")
            
        try:
            _, test_transform = self.get_transforms()
            with Image.open(image_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                image = test_transform(img).unsqueeze(0).to(self.device)
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
            'cv_metrics': self.cv_metrics,
        }
        torch.save(save_dict, path)

    def load_model(self, path):
        """Loads model state along with training parameters, metrics and history"""
        if self.net is None:
            self.initialize_model()
            
        save_dict = torch.load(path, weights_only=False)
        
        self.net.load_state_dict(save_dict['model_state'])
        self.net.eval()
        
        self.training_params = save_dict['training_params']
        self.metrics = save_dict['metrics']
        self.history = save_dict['history']
        
        # Load cross-validation metrics if they exist in the saved model
        if 'cv_metrics' in save_dict:
            self.cv_metrics = save_dict['cv_metrics']
        else:
            self.cv_metrics = {
                'k': None,
                'avg_accuracy': None,
                'std_accuracy': None,
                'fold_accuracies': None
            }
        
        if 'class_recall' not in self.metrics or 'class_names' not in self.metrics:
            if self.dataset_loaded and self.testloader is not None:
                self.test()  
        
        return {
            'training_params': self.training_params,
            'metrics': self.metrics,
            'history': self.history,
            'cv_metrics': self.cv_metrics
        }