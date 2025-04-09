from PyQt6.QtWidgets import ( 
    QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QGroupBox,
    QFileDialog, QStackedWidget, QFrame, QDialog, QSpinBox, QLabel, QMessageBox )
from PyQt6.QtCore import pyqtSignal, Qt
import os

from src.ui.dialogs.progress_dialog import ProgressDialog
from src.ui.widgets.training_plot_widget import TrainingPlotWidget
from src.ui.dialogs.parameters_dialog import ParametersDialog
from src.ui.widgets.parameters_widget import ParametersWidget
from src.ui.widgets.metrics_widget import MetricsWidget
from src.ui.widgets.model_info_widget import ModelInfoWidget
from src.models.simple_cnn_model import SimpleCnnModel
from src.models.resnet_model import ResNetModel
from src.models.efficientnet_model import EfficientNetModel
from src.models.vgg16_model import VGG16Model


class ModelTab(QWidget):
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
        self.kfold_train_btn.clicked.connect(self.train_kfold)

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
            
        # Update cross-validation results if they exist
        if self.model.cv_metrics['k'] is not None:
            # Update the spinbox value to match the k value from cross-validation
            self.k_spinbox.setValue(self.model.cv_metrics['k'])
            
            # Use HTML formatting to increase line spacing
            result_text = "<html><body style='line-height:140%;'>"
            result_text += f"K-fold Cross-validation Results (k={self.model.cv_metrics['k']}):<br>"
            result_text += f"Average Accuracy: {self.model.cv_metrics['avg_accuracy']:.2%}<br>"
            result_text += f"Standard Deviation: {self.model.cv_metrics['std_accuracy']:.2%}<br><br>"
            result_text += "Individual Fold Accuracies:<br>"
            for i, acc in enumerate(self.model.cv_metrics['fold_accuracies']):
                result_text += f"Fold {i + 1}: {acc:.2%}<br>"
            result_text += "</body></html>"
            self.kfold_result_label.setText(result_text)
        else:
            self.kfold_result_label.setText("No cross-validation performed yet")

    def clear_model(self):
        """Clears the current model"""
        if not self.model_loaded:
            self.status_message.emit("No model loaded", 8000)
            return

        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Clear Model")
        msg_box.setText("Are you sure you want to clear the model? This will reset all charts, metrics and parameters.")
        msg_box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        msg_box.setDefaultButton(QMessageBox.StandardButton.No)
        msg_box.setObjectName("clear-model-message")

        reply = msg_box.exec()

        if reply == QMessageBox.StandardButton.Yes:
            self.status_message.emit("Model cleared successfully", 8000)
            self.reset_model()

    def reset_model(self):
        """Resets model and UI elements"""
        # Reset model and UI
        self.model.reset_metrics()
        self.model.training_params = {
            'epochs': None,
            'learning_rate': None,
            'momentum': None,
            'training_time': None
        }
        self.model.history = {
            'train_loss': None,
            'val_loss': None
        }
        
        # Reset UI elements
        self.metrics_widget.reset_metrics()
        self.parameters_widget.reset_parameters()
        self.plot_widget1.plot_loss_history(self.plot_widget1)
        self.plot_widget2.plot_confusion_matrix(self.plot_widget2)
        self.kfold_result_label.setText("No cross-validation performed yet")

        # Update model status
        self.model_info_widget.set_model_status("No model loaded") 
        self.model_info_widget.set_model_file("")
        
        # Disable buttons
        self.save_model_btn.setEnabled(False)
        self.clear_model_btn.setEnabled(False)
        self.model_loaded = False

    def _load_dataset(self):
        """Loads the dataset on startup"""
        try:
            train_size, val_size, test_size = self.model.load_data("./dataset/fruitveg-dataset")
            self.status_message.emit(f"Dataset loaded successfully: {len(self.model.classes)} classes detected", 8000)
        except Exception as e:
            error_msg = f"Error loading dataset: {str(e)}"
            print(f"--------------------------------------{error_msg}")
            self.status_message.emit(error_msg, 8000)
  
    def _load_default_model(self):
        """Attempts to load the default model on startup"""
        # Determine the correct default model path based on model class
        if self.model_class == SimpleCnnModel:
            default_model_path = "./saved_models/simple_cnn_default_model.pth"
            self.model_name = "Simple CNN"
        elif self.model_class == ResNetModel:
            default_model_path = "./saved_models/resnet_default_model.pth"
            self.model_name = "ResNet"
        elif self.model_class == EfficientNetModel:
            default_model_path = "./saved_models/efficientnet_default_model.pth"
            self.model_name = "EfficientNet"
        elif self.model_class == VGG16Model:
            default_model_path = "./saved_models/vgg16_default_model.pth"
            self.model_name = "VGG16"
        else:
            default_model_path = None
            self.model_info_widget.set_model_status("No model loaded", "red")
            return

        self.metrics_group.setTitle(f"{self.model_name} Stats")

        # Attempt to load default model
        try:
            self.model.initialize_model()
            if os.path.exists(default_model_path):
                try:
                    metadata = self.model.load_model(default_model_path)

                    # Update UI with loaded data
                    self._update_ui_from_model_data()
                    # Display only the filename, not the full path
                    filename = os.path.basename(default_model_path)
                    self.model_info_widget.set_model_file(filename) 
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
                # Reset metrics, parameters and plots first
                self.metrics_widget.reset_metrics()
                self.parameters_widget.reset_parameters()
                self.plot_widget1.plot_loss_history(self.plot_widget1)
                self.plot_widget2.plot_confusion_matrix(self.plot_widget2)
                self.kfold_result_label.setText("No cross-validation performed yet")

                metadata = self.model.load_model(file_path)

                # Display filename
                filename = os.path.basename(file_path)
                self.model_info_widget.set_model_file(filename)

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
        param_dialog = ParametersDialog(self)
        
        # Get parameters from dialog
        result = param_dialog.exec()
        
        # If dialog was canceled, return
        if result != QDialog.DialogCode.Accepted:
            return
        
        # Get parameters from dialog
        params = param_dialog.get_parameters()
        epochs = params['epochs']
        learning_rate = params['learning_rate']
        momentum = params['momentum']
        
        try:
            # Check if dataset is loaded
            if not self.model.is_data_loaded():
                self.status_message.emit("Dataset not loaded. Please load the dataset first.", 8000)
                return
                
            # Model Initialization
            self.model.initialize_model()

            # Reset metrics display
            self.metrics_widget.reset_metrics()
            self.parameters_widget.reset_parameters()
            self.kfold_result_label.setText("No cross-validation performed yet")

            # Update model status
            self.model_info_widget.set_model_status("Training in progress...", "blue")
            self.model_info_widget.set_model_file("")

            # Reset plots
            self.plot_widget1.plot_loss_history(self.plot_widget1)
            self.plot_widget2.plot_confusion_matrix(self.plot_widget2)  

            dialog = ProgressDialog(
                self, 
                epochs=epochs,
                learning_rate=learning_rate,
                momentum=momentum,
                model_name=self.model_name
            )
            
            # Connect signals when training and testing are finished
            dialog.complete.connect(self.handle_training_and_testing_complete)

            # Handle errors
            dialog.error_occurred.connect(self.handle_training_and_testing_error)

            self.status_message.emit("Training started...", 8000)

            dialog.start_training(self.model, epochs, learning_rate, momentum)
            
        except Exception as e:
            error_message = f"Error: {str(e)}"
            print(f"Training error: {error_message}")
            self.status_message.emit(error_message, 8000)  
            self.reset_model()
            
    def handle_training_and_testing_complete(self, training_result, testing_result):
        """Handle training completion from the progress dialog"""
        if training_result is not None:
            train_loss_history, val_loss_history = training_result

            # Plotting loss history
            self.plot_widget1.plot_loss_history(
                self.plot_widget1, 
                self.model.training_params['epochs'], 
                train_loss_history, 
                val_loss_history
            )

            self.model_loaded = True
            self.save_model_btn.setEnabled(True)
            self.clear_model_btn.setEnabled(True)
            self.model_info_widget.set_model_status("Model trained successfully", "green")
            self.parameters_widget.update_parameters(self.model.training_params)
            
            # Update UI with testing results
            if testing_result is not None:
                self.metrics_widget.update_metrics(testing_result)
                
                # Confusion matrix
                conf_mat = testing_result['confusion_matrix']
                classes = self.model.classes
                self.plot_widget2.plot_confusion_matrix(self.plot_widget2, conf_mat, classes)
                
                self.status_message.emit("Training and testing completed", 8000)
            else:
                self.status_message.emit("Training completed but testing failed", 8000)
        else:
            self.status_message.emit("Training canceled or failed, model was reset", 8000)       
            self.reset_model()
            
    def handle_training_and_testing_error(self, error_message):
        print(f"Training and testing error: {error_message}")
        self.status_message.emit(f"Error: {error_message}", 8000)  
        self.reset_model()

    def train_kfold(self):
        """Performs K-fold cross-validation on the model"""
        try:
            # Check if dataset is loaded
            if not self.model.is_data_loaded():
                self.status_message.emit("Dataset not loaded. Please load the dataset first.", 8000)
                return

            k = self.k_spinbox.value()
            self.kfold_result_label.setText("Cross-validation in progress...")
            self.kfold_train_btn.setEnabled(False)

            # Get the training parameters from the current model
            epochs = self.model.training_params['epochs']
            learning_rate = self.model.training_params['learning_rate']
            momentum = self.model.training_params['momentum']

            # If parameters are not set, show the parameters dialog
            if epochs is None or learning_rate is None or momentum is None:
                param_dialog = ParametersDialog(self)
                result = param_dialog.exec()
                
                if result != QDialog.DialogCode.Accepted:
                    self.kfold_result_label.setText("No cross-validation performed yet")
                    self.kfold_train_btn.setEnabled(True)
                    return
                    
                # Get parameters from dialog
                params = param_dialog.get_parameters()
                epochs = params['epochs']
                learning_rate = params['learning_rate']
                momentum = params['momentum']

            # Reset cross-validation metrics in the model
            self.model.cv_metrics = {
                'k': None,
                'avg_accuracy': None,
                'std_accuracy': None,
                'fold_accuracies': None
            }

            # Perform K-fold cross-validation
            self.accuracies = []  # Store accuracies as instance variable
            self.current_fold = 0  # Track current fold
            
            # Start the first fold
            self._start_next_fold(k, epochs, learning_rate, momentum)

        except Exception as e:
            error_message = f"Error during cross-validation: {str(e)}"
            print(f"Cross-validation error: {error_message}")
            self.status_message.emit(error_message, 8000)
            self.kfold_result_label.setText("Error during cross-validation")
            self.kfold_train_btn.setEnabled(True)
            
    def _start_next_fold(self, k, epochs, learning_rate, momentum):
        """Starts the next fold of K-fold cross-validation"""
        try:
            if self.current_fold < k:
                # Create a new model instance for this fold
                fold_model = self.model_class()
                fold_model.initialize_model()
                
                # Split the data into k folds
                fold_model.load_data("./dataset/fruitveg-dataset", k=k, current_fold=self.current_fold)
                
                # Create and show progress dialog for this fold
                dialog = ProgressDialog(
                    self, 
                    epochs=epochs,
                    learning_rate=learning_rate,
                    momentum=momentum,
                    model_name=self.model_name
                )
                
                # Connect signals
                dialog.complete.connect(lambda train_result, test_result: self._handle_fold_complete(train_result, test_result, k, epochs, learning_rate, momentum))
                dialog.error_occurred.connect(self._handle_fold_error)
                
                # Start training
                dialog.start_training(fold_model, epochs, learning_rate, momentum)
                
                # Update progress
                self.kfold_result_label.setText(f"Fold {self.current_fold + 1}/{k} in progress...")
            else:
                # All folds completed, calculate final results
                self._calculate_final_results(k)
        except Exception as e:
            error_message = f"Error starting fold {self.current_fold + 1}: {str(e)}"
            print(f"Fold error: {error_message}")
            self.status_message.emit(error_message, 8000)
            self.kfold_result_label.setText("Error during cross-validation")
            self.kfold_train_btn.setEnabled(True)
            
    def _handle_fold_complete(self, training_result, testing_result, k, epochs, learning_rate, momentum):
        """Handle completion of a single fold"""
        try:
            if testing_result is not None and 'accuracy' in testing_result:
                self.accuracies.append(testing_result['accuracy'])
                
                # Move to the next fold
                self.current_fold += 1
                
                # Start the next fold
                self._start_next_fold(k, epochs, learning_rate, momentum)
            else:
                error_message = "No valid testing results received"
                print(f"Fold error: {error_message}")
                self.status_message.emit(error_message, 8000)
                self.kfold_result_label.setText("Error during cross-validation")
                self.kfold_train_btn.setEnabled(True)
        except Exception as e:
            error_message = f"Error processing fold results: {str(e)}"
            print(f"Fold error: {error_message}")
            self.status_message.emit(error_message, 8000)
            self.kfold_result_label.setText("Error during cross-validation")
            self.kfold_train_btn.setEnabled(True)
            
    def _calculate_final_results(self, k):
        """Calculate and display final results of K-fold cross-validation"""
        try:
            if len(self.accuracies) > 0:
                avg_accuracy = sum(self.accuracies) / len(self.accuracies)
                if len(self.accuracies) > 1:
                    std_accuracy = (sum((x - avg_accuracy) ** 2 for x in self.accuracies) / (len(self.accuracies) - 1)) ** 0.5
                else:
                    std_accuracy = 0.0
                
                # Store cross-validation metrics in model
                self.model.cv_metrics = {
                    'k': k,
                    'avg_accuracy': avg_accuracy,
                    'std_accuracy': std_accuracy,
                    'fold_accuracies': self.accuracies.copy()
                }
                
                # Make sure the spinbox reflects the current k value
                self.k_spinbox.setValue(k)
                
                # Use HTML formatting to increase line spacing
                result_text = "<html><body style='line-height:140%;'>"
                result_text += f"K-fold Cross-validation Results (k={k}):<br>"
                result_text += f"Average Accuracy: {avg_accuracy:.2%}<br>"
                result_text += f"Standard Deviation: {std_accuracy:.2%}<br><br>"
                result_text += "Individual Fold Accuracies:<br>"
                for i, acc in enumerate(self.accuracies):
                    result_text += f"Fold {i + 1}: {acc:.2%}<br>"
                result_text += "</body></html>"
                
                self.kfold_result_label.setText(result_text)
                self.status_message.emit(f"K-fold cross-validation completed. Average accuracy: {avg_accuracy:.2%}", 8000)
            else:
                self.kfold_result_label.setText("Error: No valid results from cross-validation")
                self.status_message.emit("Error: No valid results from cross-validation", 8000)
        except Exception as e:
            error_message = f"Error calculating final results: {str(e)}"
            print(f"Results error: {error_message}")
            self.status_message.emit(error_message, 8000)
            self.kfold_result_label.setText("Error calculating final results")
        finally:
            self.kfold_train_btn.setEnabled(True)

    def _handle_fold_error(self, error_message):
        """Handle error during a fold"""
        print(f"Fold error: {error_message}")
        self.status_message.emit(f"Error: {error_message}", 8000)
        self.kfold_result_label.setText("Error during cross-validation")
        self.kfold_train_btn.setEnabled(True)
        
        # Reset cross-validation metrics in the model
        self.model.cv_metrics = {
            'k': None,
            'avg_accuracy': None,
            'std_accuracy': None,
            'fold_accuracies': None
        }

    def _create_ui(self):
        # Main layout
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(0)

        # Left panel
        left_panel = self._create_left_panel()
        main_layout.addWidget(left_panel)
        
        # Right panel (plots)
        right_panel = self._create_right_panel()
        main_layout.addWidget(right_panel)

    def _create_left_panel(self):
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(20)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        # Dataset, model and training
        model_group = QGroupBox("Model Controls")
        model_group.setObjectName("model-controls")
        
        model_layout = QVBoxLayout()
        model_layout.setContentsMargins(0, 0, 0, 0)
        model_group.setLayout(model_layout)

        # Model buttons
        buttons_layout = QHBoxLayout()
        self.train_model_btn = QPushButton("Train")
        self.save_model_btn = QPushButton("Save")
        self.load_model_btn = QPushButton("Load")
        self.clear_model_btn = QPushButton("Clear")
        self.save_model_btn.setEnabled(False)
        self.clear_model_btn.setEnabled(False)
        for btn in [self.train_model_btn, self.load_model_btn, self.save_model_btn, self.clear_model_btn]:
            buttons_layout.addWidget(btn)
        model_layout.addLayout(buttons_layout)

        # K-fold Cross-validation
        kfold_group = QGroupBox("k-Fold Cross-Validation")
        kfold_group.setObjectName("model-controls")
        kfold_layout = QVBoxLayout()
        kfold_layout.setContentsMargins(0, 0, 0, 0)
        kfold_group.setLayout(kfold_layout)

        # K-fold controls
        kfold_controls_layout = QHBoxLayout()
        self.kfold_train_btn = QPushButton("Train")
        k_label = QLabel("k:")
        k_label.setFixedWidth(20)
        self.k_spinbox = QSpinBox()
        self.k_spinbox.setRange(2, 10)
        self.k_spinbox.setValue(5)
        self.k_spinbox.setFixedWidth(60)
        
        kfold_controls_layout.addWidget(self.kfold_train_btn)
        kfold_controls_layout.addSpacing(5)
        kfold_controls_layout.addWidget(k_label)
        kfold_controls_layout.addWidget(self.k_spinbox)
        kfold_controls_layout.addStretch()
        
        # K-fold results
        self.kfold_result_label = QLabel("No cross-validation performed yet")
        self.kfold_result_label.setWordWrap(True)
        self.kfold_result_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse | Qt.TextInteractionFlag.TextSelectableByKeyboard)
        
        kfold_layout.addLayout(kfold_controls_layout)
        kfold_layout.addSpacing(10)
        kfold_layout.addWidget(self.kfold_result_label)

        # Metrics group box
        self.model_name = ""
        self.metrics_group = QGroupBox(f"{self.model_name} Stats")
        self.metrics_group.setObjectName("model-metrics")
        metrics_layout = QVBoxLayout()
        metrics_layout.setContentsMargins(18, 0, 18, 0)
        self.metrics_widget = MetricsWidget()
        metrics_layout.addWidget(self.metrics_widget)
        self.metrics_group.setLayout(metrics_layout)

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

        # Create widget containers for the left panel
        for widget in [model_group, self.metrics_group, self.parameters_group, kfold_group]:
            left_layout.addWidget(widget)
            
        left_layout.addStretch()
        
        return left_panel

    def _create_right_panel(self):
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(0)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        # Plots
        self.plot_frame = QFrame()
        self.plot_frame.setObjectName("plot-1-2-frame")

        frame_layout = QVBoxLayout(self.plot_frame)
        frame_layout.setSpacing(9)

        self.plot_stack = QStackedWidget()
        self.plot_stack.setMinimumWidth(400)
        self.plot_widget1 = TrainingPlotWidget("Loss History")
        self.plot_widget2 = TrainingPlotWidget("Confusion Matrix")
        
        # Initialize empty plots
        self.plot_widget1.plot_loss_history(self.plot_widget1)
        self.plot_widget2.plot_confusion_matrix(self.plot_widget2)
        
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
        
        buttons_layout.addWidget(self.btn_confusion_matrix)  
        buttons_layout.addWidget(self.btn_loss_history)
        
        frame_layout.addLayout(buttons_layout)

        right_layout.addWidget(self.plot_frame)
        
        # Add model info group below the plot frame
        right_layout.addWidget(self.model_info_group)
        
        right_layout.addStretch()
        
        return right_panel

    def _switch_plot(self, index):
        """Switches between plots and updates button status"""
        self.plot_stack.setCurrentIndex(index)
        buttons = [self.btn_confusion_matrix, self.btn_loss_history]
        for i, btn in enumerate(buttons):
            btn.setChecked(i == index)