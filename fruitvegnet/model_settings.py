from PyQt6.QtWidgets import ( 
    QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QGroupBox,
    QFileDialog, QStackedWidget, QMessageBox, QFrame, QDialog )
from PyQt6.QtCore import pyqtSignal
import os
import torch

from fruitvegnet.widgets.progress_dialog import ProgressDialog
from fruitvegnet.widgets.plot_widget import PlotWidget
from fruitvegnet.widgets.parameters_dialog import ParameterDialog
from fruitvegnet.widgets.parameters_widget import ParametersWidget
from fruitvegnet.widgets.metrics_widget import MetricsWidget
from fruitvegnet.widgets.model_info_widget import ModelInfoWidget
from models.simple_cnn_model import SimpleCnnModel
from models.resnet_model import ResNetModel
from models.efficientnet_model import EfficientNetModel
from models.vgg16_model import VGG16Model


class TabWidget(QWidget):
    # Signal to emit status messages to the main window
    status_message = pyqtSignal(str, int)

    def __init__(self, model_class=SimpleCnnModel):
        super().__init__()
        self.model_class = model_class
        self._create_ui()
        self.current_image_path = None

        # Model Initialization 
        self.model = model_class()
        self.model_loaded = False

        # Update model status
        self.model_info_widget.set_model_status("No model loaded") 

        # Try to load the dataset and default model on startup
        self._load_dataset()
        self._load_default_model()

        # Connect signals
        self._setup_connections()    

    def _setup_connections(self):
        self.load_model_btn.clicked.connect(self.load_model)
        self.save_model_btn.clicked.connect(self.save_model)
        self.train_model_btn.clicked.connect(self.train_model)
        self.clear_model_btn.clicked.connect(self.clear_model)

    def update_metrics_display(self, metrics):
        """Method to update the metrics"""
        self.accuracy_label.setText(f"Accuracy: {metrics['accuracy']:.2%}")
        self.precision_label.setText(f"Precision: {metrics['precision']:.2%}")
        self.recall_label.setText(f"Recall: {metrics['recall']:.2%}")

    def _update_ui_from_model_data(self):
        """Updates UI elements with model data after the application starts"""
        
        # Update metrics display if metrics exist
        if self.model.metrics['accuracy'] is not None:
            self.metrics_widget.update_metrics(self.model.metrics)
        else:
            self.metrics_widget.reset_metrics()

        # Update training parameters
        self.parameters_widget.update_parameters(self.model.training_params)

        # Update loss history plot if history exists
        if (self.model.history['train_loss'] is not None and 
            self.model.history['val_loss'] is not None):
            self.plot_widget1.plot_loss_history(
                self.plot_widget1,
                len(self.model.history['train_loss']),
                self.model.history['train_loss'],
                self.model.history['val_loss']
            )
        else:
            self.plot_widget1.plot_loss_history(self.plot_widget1)

        # Update confusion matrix plot if confusion matrix exists
        if self.model.metrics['confusion_matrix'] is not None:
            self.plot_widget2.plot_confusion_matrix(
                self.plot_widget2,
                self.model.metrics['confusion_matrix'],
                self.model.classes
            )
        else:
            self.plot_widget2.plot_confusion_matrix(self.plot_widget2)

    def clear_model(self):
        """Calls the reset_model function that resets the loaded model"""   
        reply = QMessageBox.question(
            self,
            'Clear Model',
            'Are you sure you want to clear the model? This will reset all charts, metrics and parameters.',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            self.status_message.emit("Model cleared successfully", 8000)
            self.reset_model()

    def reset_model(self):
        """Clears the current model and resets UI elements while preserving dataset"""   
        dataset_loaded = self.model.is_data_loaded()
        trainloader = self.model.trainloader
        valloader = self.model.valloader
        testloader = self.model.testloader
        old_classes = self.model.classes 
        self.model = self.model_class()
        self.model.initialize_model()

        # Restore dataset if it was loaded
        if dataset_loaded:
            self.model.dataset_loaded = dataset_loaded
            self.model.trainloader = trainloader
            self.model.valloader = valloader
            self.model.testloader = testloader
            self.model.classes = old_classes
        self.model_loaded = False

        # Reset UI elements
        self.model_info_widget.set_model_status("No model loaded", "red")
        self.model_info_widget.set_model_file("None")

        # Reset metrics and parameters
        self.metrics_widget.reset_metrics()
        self.parameters_widget.reset_parameters()

        # Reset plots
        self.plot_widget1.plot_loss_history(self.plot_widget1)
        self.plot_widget2.plot_confusion_matrix(self.plot_widget2)

        # Disable buttons
        self.save_model_btn.setEnabled(False)
        self.clear_model_btn.setEnabled(False)

    def _load_dataset(self):
        """Loads the dataset on startup"""
        try:
            train_size, val_size, test_size = self.model.load_data("./dataset/fruit_dataset")
        except Exception as e:
            print("--------------------------------------Error loading dataset: {str(e)}. ")
  
    def _load_default_model(self):
        """Attempts to load the default model on startup"""
        # Determine the correct default model path based on model class
        if self.model_class == SimpleCnnModel:
            default_model_path = "./models/trained_models/simple_cnn_default_model.pth"
            self.model_name = "Simple CNN"
        elif self.model_class == ResNetModel:
            default_model_path = "./models/trained_models/resnet_default_model.pth"
            self.model_name = "ResNet"
        elif self.model_class == EfficientNetModel:
            default_model_path = "./models/trained_models/efficientnet_default_model.pth"
            self.model_name = "EfficientNet"
        elif self.model_class == VGG16Model:
            default_model_path = "./models/trained_models/vgg16_default_model.pth"
            self.model_name = "VGG16"
        else:
            default_model_path = None
            self.model_info_widget.set_model_status("No model loaded", "red")
            return

        # Set the group box title and loaded model file name
        self.metrics_group.setTitle(f"{self.model_name} Stats")
        self.model_info_widget.set_model_file(default_model_path.split("/")[-1]) 

        # Attempt to load default model
        try:
            self.model.initialize_model()
            if os.path.exists(default_model_path):
                try:
                    metadata = self.model.load_model(default_model_path)

                    # Update UI with loaded data
                    self._update_ui_from_model_data()
                    self.model_loaded = True
                    self.model_info_widget.set_model_status("Model loaded successfully", "green")
                    self.save_model_btn.setEnabled(True)
                    self.clear_model_btn.setEnabled(True)
                except Exception as e:
                    self.model_info_widget.set_model_status("Error loading default model", "red")
            else:
                self.model_info_widget.set_model_status("No model loaded", "red")
        except Exception as e:
            self.model_info_widget.set_model_status("Error initializing model", "red")
            
    def load_model(self):
        """Loads the trained neural network model"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Model",
            "",
            "PyTorch Model (*.pth)"
        )

        if file_path:
            try:
                # Load model and Check if the it's architecture matches
                checkpoint = torch.load(file_path, weights_only=False)  

                if isinstance(self.model, SimpleCnnModel):
                    if not all(key in checkpoint['model_state'] for key in ['conv1.weight', 'conv2.weight']):
                        raise ValueError("This model file is not compatible with Simple CNN architecture")
                elif isinstance(self.model, ResNetModel):
                    if not all(key in checkpoint['model_state'] for key in ['layer1.0.conv1.weight', 'layer1.0.conv2.weight']):
                        raise ValueError("This model file is not compatible with ResNet architecture")
                elif isinstance(self.model, VGG16Model):
                    if not all(key in checkpoint['model_state'] for key in ['features.0.weight', 'features.2.weight', 'classifier.6.weight']):
                        raise ValueError("This model file is not compatible with VGG16 architecture")
                elif isinstance(self.model, EfficientNetModel):
                    if not all(key in checkpoint['model_state'] for key in ['features.0.0.weight', 'features.1.0.block.0.0.weight', 'classifier.1.weight']):
                        raise ValueError("This model file is not compatible with EfficientNet architecture")
                
                self.model.initialize_model()

                # Load model and get metadata
                metadata = self.model.load_model(file_path)

                # Update GroupBox title with model name
                self.metrics_group.setTitle(f"{self.model_name} Stats")

                # Update the model name label
                self.model_info_widget.set_model_file(file_path.split("/")[-1]) 

                # Reset metrics, parametrs and plots first
                self.metrics_widget.reset_metrics()
                self.parameters_widget.reset_parameters()
                self.plot_widget1.plot_loss_history(self.plot_widget1)
                self.plot_widget2.plot_confusion_matrix(self.plot_widget2)

                # Update UI with loaded data only if the data exists
                if metadata['training_params']['epochs'] is not None:
                    self._update_ui_from_model_data()

                self.model_loaded = True
                self.save_model_btn.setEnabled(True)
                self.clear_model_btn.setEnabled(True)
                self.model_info_widget.set_model_status("Model loaded successfully", "green")

                self.status_message.emit(f"Model loaded successfully (Accuracy: {metadata['metrics']['accuracy']:.2%})", 8000)
            except Exception as e:
                self.status_message.emit(f"Error loading model: {str(e)}", 8000)

    def save_model(self):
        """Saves the trained neural network model"""
        if not self.model_loaded:
            self.status_message.emit("No model to save", 8000)
            return
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Model",
            "",
            "PyTorch Model (*.pth)"
        )

        if file_path:
            try:
                self.model.save_model(file_path)
                self.status_message.emit("Model saved successfully", 8000)
            except Exception as e:
                self.status_message.emit(f"Error saving model: {str(e)}", 8000)

    def train_model(self):
        """Trains the neural network"""
        param_dialog = ParameterDialog(self)
        
        # If we have model parameters, set them in the dialog
        if self.model.training_params['epochs'] is not None:
            param_dialog.set_parameters(
                epochs=self.model.training_params['epochs'],
                learning_rate=self.model.training_params['learning_rate'],
                momentum=self.model.training_params['momentum']
            )
        
        # Show dialog and wait for user response
        result = param_dialog.exec()
        
        # If user clicked Cancel, do nothing
        if result != QDialog.DialogCode.Accepted:
            return
            
        # Get parameters from dialog
        params = param_dialog.get_parameters()
        epochs = params['epochs']
        learning_rate = params['learning_rate']
        momentum = params['momentum']

        try:
            # Model Initialization
            self.model.initialize_model()

            # Reset metrics display
            self.metrics_widget.reset_metrics()
            self.parameters_widget.reset_parameters()

            # Update model status
            self.model_info_widget.set_model_status("No model loaded")
            self.model_info_widget.set_model_file("")

            # Reset confusion matrix plot
            self.plot_widget2.plot_confusion_matrix(self.plot_widget2)  

            dialog = ProgressDialog(
                self, 
                "Training",
                epochs=epochs,
                learning_rate=learning_rate,
                momentum=momentum
            )

            result = dialog.start_training(self.model, epochs, learning_rate, momentum)    
            
            if result is not None:
                train_loss_history, val_loss_history = result

                # Plotting loss history
                self.plot_widget1.plot_loss_history(
                    self.plot_widget1, 
                    epochs, 
                    train_loss_history, 
                    val_loss_history
                )

                self.model_loaded = True
                self.save_model_btn.setEnabled(True)
                self.clear_model_btn.setEnabled(True)
                self.model_info_widget.set_model_status("Model trained successfully", "green")
                self.status_message.emit("Training completed", 8000)

                self.parameters_widget.update_parameters(self.model.training_params)

                # Test the model
                self.test_model()
            else:
                self.status_message.emit("Training canceled, model was reset", 8000)       
                self.reset_model()
        except Exception as e:
            self.status_message.emit("Training canceled, model was reset", 8000)  
            self.reset_model()

    def test_model(self):
        """Test the neural network model"""
        if not self.model_loaded:
            self.status_message.emit("No model loaded", 8000)
            return
        try:
            dialog = ProgressDialog(self, "Testing")
            metrics = dialog.start_testing(self.model)
            if metrics is not None:
                self.metrics_widget.update_metrics(metrics)

                # Confusion matrix 
                conf_mat = metrics['confusion_matrix']
                classes = self.model.classes
                self.plot_widget2.plot_confusion_matrix(self.plot_widget2, conf_mat, classes)
            else:
                self.status_message.emit(
                    f"Testing error: {getattr(dialog, 'error_message', 'Unknown error')}",
                    8000
                ) 
        except Exception as e:
            self.status_message.emit(f"Testing error: {str(e)}", 8000)    

    def _create_ui(self):
        # Main layout
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10   )

        # Left panel
        left_panel = self._create_left_panel()
        main_layout.addWidget(left_panel)
        
        # Right panel (plots)
        right_panel = self._create_right_panel()
        main_layout.addWidget(right_panel)

    def _create_left_panel(self):
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        #left_layout.setSpacing(10)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # Model Controls
        model_group = QGroupBox("Model Controls")
        model_group.setObjectName("model-controls")
        model_layout = QVBoxLayout()
        model_layout.setContentsMargins(10, 10, 10, 10)
        model_group.setLayout(model_layout)

        # Model buttons
        buttons_layout = QHBoxLayout()
        self.train_model_btn = QPushButton("Train")
        self.save_model_btn = QPushButton("Save")
        self.load_model_btn = QPushButton("Load")
        self.clear_model_btn = QPushButton("Clear")
        self.save_model_btn.setEnabled(False)
        self.clear_model_btn.setEnabled(False)

        # Add buttons to layout
        for btn in [self.train_model_btn, self.load_model_btn, self.save_model_btn, self.clear_model_btn]:
            buttons_layout.addWidget(btn)
        model_layout.addLayout(buttons_layout)

        # Metrics group box
        self.model_name = ""
        self.metrics_group = QGroupBox(f"{self.model_name} Stats")
        self.metrics_group.setObjectName("model-metrics")
        metrics_layout = QVBoxLayout()
        self.metrics_widget = MetricsWidget()
        metrics_layout.addWidget(self.metrics_widget)
        self.metrics_group.setLayout(metrics_layout)
        metrics_layout.setContentsMargins(10, 10, 10, 10)

        # Parameters group box
        self.parameters_group = QGroupBox("")
        self.parameters_group.setObjectName("model-parameters")
        parameters_layout = QVBoxLayout()
        parameters_layout.setContentsMargins(0, 18, 0, 18)
        self.parameters_widget = ParametersWidget()
        parameters_layout.addWidget(self.parameters_widget)
        self.parameters_group.setLayout(parameters_layout)
        self.parameters_group.setContentsMargins(0, 0, 0, 0)

        # Model info group box
        self.model_info_group = QGroupBox("")
        self.model_info_group.setObjectName("model-info")
        model_info_layout = QVBoxLayout()
        model_info_layout.setContentsMargins(0, 18, 0, 18)
        self.model_info_widget = ModelInfoWidget()  
        model_info_layout.addWidget(self.model_info_widget)
        self.model_info_group.setLayout(model_info_layout)
        self.model_info_group.setContentsMargins(0, 0, 0, 0)

        # Add all components to left panel
        for widget in [model_group, self.metrics_group, self.parameters_group, self.model_info_group]:
            left_layout.addWidget(widget)
        left_layout.addStretch()

        return left_panel

    def _create_right_panel(self):
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(20)
        right_layout.setContentsMargins(15, 30, 10, 10)
        
        # Create StyledFrame
        self.plot_frame = QFrame()
        self.plot_frame.setObjectName("styled-frame")

        frame_layout = QVBoxLayout(self.plot_frame)
        
        # QStackedWidget for switching between charts
        self.plot_stack = QStackedWidget()
        self.plot_widget1 = PlotWidget("Loss History")
        self.plot_widget2 = PlotWidget("Confusion Matrix")
        
        # Initialize empty plots
        self.plot_widget1.plot_loss_history(self.plot_widget1)
        self.plot_widget2.plot_confusion_matrix(self.plot_widget2)
        
        # Add plots to QStackedWidget
        self.plot_stack.addWidget(self.plot_widget2)  
        self.plot_stack.addWidget(self.plot_widget1) 
        frame_layout.addWidget(self.plot_stack)
        
        # Buttons
        buttons_layout = QHBoxLayout()
        buttons_layout.setContentsMargins(0, 0, 0, 0)
        buttons_layout.setSpacing(0)
        self.btn_confusion_matrix = QPushButton("Confusion Matrix")  
        self.btn_confusion_matrix.setObjectName("conf-matrix")
        self.btn_loss_history = QPushButton("Loss History")
        self.btn_loss_history.setObjectName("loss-history")
        self.btn_confusion_matrix.setCheckable(True)
        self.btn_loss_history.setCheckable(True)
        self.btn_confusion_matrix.setChecked(True)  

        self.btn_confusion_matrix.clicked.connect(lambda: self._switch_plot(0))  
        self.btn_loss_history.clicked.connect(lambda: self._switch_plot(1)) 
        
        # Add buttons to the layout 
        buttons_layout.addWidget(self.btn_confusion_matrix)  
        buttons_layout.addWidget(self.btn_loss_history)
        
        # Add buttons to the frame layout
        frame_layout.addLayout(buttons_layout)
        
        # Add frame to the right_layout
        right_layout.addWidget(self.plot_frame)
        right_layout.addStretch()
        
        return right_panel

    def _switch_plot(self, index):
        """Switches between plots and updates button status"""
        self.plot_stack.setCurrentIndex(index)
        buttons = [self.btn_confusion_matrix, self.btn_loss_history]
        for i, btn in enumerate(buttons):
            btn.setChecked(i == index)