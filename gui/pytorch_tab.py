from PyQt6.QtWidgets import ( 
    QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QSlider, QDoubleSpinBox, 
    QSpinBox, QSizePolicy, QFileDialog, QScrollArea, QFrame, QStatusBar, QGridLayout, QMessageBox )
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QImageReader, QFont
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import os
from gui.progress_dialog import ProgressDialog
from models.pytorch_model import DiceTossModel
import numpy as np


class PlotWidget(QWidget):
    def __init__(self, title, parent=None):
        super().__init__(parent)
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)
        main_layout.setContentsMargins(0, 0, 0, 0)
        # Frame styled like a group box
        frame = QFrame()
        frame.setObjectName("StyledFrame")  
        frame.setStyleSheet("""
            QFrame#StyledFrame {
                border: 1px solid #c4c8cc;
                border-radius: 6px;
                padding: 5px;
                background-color: white;
            }
        """)
        frame_layout = QVBoxLayout(frame)
        main_layout.addWidget(frame)
        # Title label
        self.title_label = QLabel(title)
        self.title_label.setFont(QFont('Arial', 10, QFont.Weight.Bold))
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        frame_layout.addWidget(self.title_label)
        # Figure
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        frame_layout.addWidget(self.canvas)

    def plot_confusion_matrix(self, plot_widget, conf_mat = None, classes = None):
        plot_widget.figure.clear()
        plot_widget.figure.subplots_adjust(left=0.25, right=0.85, bottom=0.25, top=0.85)
        plot_widget.figure.tight_layout()
        ax = plot_widget.figure.add_subplot(111)
        ax.set_xlabel('Predicted', labelpad=10)
        ax.set_ylabel('True', labelpad=10)
        # Empty confusion matrix
        if conf_mat is None or classes is None:
            ax.set_title('(Will be populated after testing)', pad=10)
        else:   # Populated confusion matrix
            im = ax.imshow(conf_mat, cmap='Blues', aspect='auto')
            cbar = plot_widget.figure.colorbar(im)
            cbar.ax.tick_params(labelsize=8)
            ax.set_xticks(np.arange(len(classes)))
            ax.set_yticks(np.arange(len(classes)))
            ax.set_xticklabels(classes)
            ax.set_yticklabels(classes)
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            for i in range(len(classes)):
                for j in range(len(classes)):
                    ax.text(j, i, conf_mat[i, j], 
                        ha="center", va="center",
                        color="white" if conf_mat[i, j] > conf_mat.max() / 2 else "black")         
        plot_widget.canvas.draw()

    def plot_loss_history(self, plot_widget, epochs = None, train_loss_history = None, val_loss_history = None):
        plot_widget.figure.clear()
        plot_widget.figure.subplots_adjust(left=0.25, right=0.85, bottom=0.25, top=0.85)
        plot_widget.figure.tight_layout()
        ax = plot_widget.figure.add_subplot(111)
        ax.set_xlabel('Epochs', labelpad=10)
        ax.set_ylabel('Loss', labelpad=10)
        ax.grid(True)
        if epochs is not None or train_loss_history is not None or val_loss_history is not None:
            ax.plot(range(1, epochs + 1), train_loss_history, label='Training Loss')
            ax.plot(range(1, epochs + 1), val_loss_history, label='Validation Loss')
            ax.legend()       
        plot_widget.canvas.draw()


class ParameterWidget(QWidget):
    def __init__(self, label, min_val, max_val, default_val, decimals=0, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout()
        self.setLayout(layout)
        param_label = QLabel(label)
        param_label.setMinimumWidth(100)
        layout.addWidget(param_label)
        # SpinBox setup
        if decimals == 0:
            self.spinbox = QSpinBox()
        else:
            self.spinbox = QDoubleSpinBox()
            self.spinbox.setDecimals(decimals)
        self.spinbox.setRange(min_val, max_val)
        self.spinbox.setValue(default_val)
        self.spinbox.setFixedWidth(80)
        layout.addWidget(self.spinbox)
        # Slider setup
        self.slider = QSlider(Qt.Orientation.Horizontal)
        if decimals > 0:
            self.slider.setRange(0, int(max_val * (10 ** decimals)))
            self.slider.setValue(int(default_val * (10 ** decimals)))
        else:
            self.slider.setRange(int(min_val), int(max_val))
            self.slider.setValue(int(default_val))
        layout.addWidget(self.slider)
        # Connect signals
        self._setup_connections()

    def _setup_connections(self):
        self.slider.valueChanged.connect(self._slider_changed)
        self.spinbox.valueChanged.connect(self._spinbox_changed)
    
    def _slider_changed(self, value):
        if isinstance(self.spinbox, QDoubleSpinBox):
            self.spinbox.setValue(value / (10 ** self.spinbox.decimals()))
        else:
            self.spinbox.setValue(value)
    
    def _spinbox_changed(self, value):
        if isinstance(self.spinbox, QDoubleSpinBox):
            self.slider.setValue(int(value * (10 ** self.spinbox.decimals())))
        else:
            self.slider.setValue(int(value))


class ImageClassificationWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout()
        self.setLayout(layout)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        # Controls area with buttons
        controls_layout = QVBoxLayout()
        self.load_image_btn = QPushButton("Load Image")
        self.load_image_btn.setStyleSheet("padding: 8px;")
        self.classify_btn = QPushButton("Classify Image")
        self.classify_btn.setStyleSheet("padding: 8px;")
        self.result_label = QLabel("Classification: None")
        self.result_label.setStyleSheet("""
            QLabel {
                font-weight: bold;
                color: #2c3e50;
                padding: 10px;
                background-color: #ecf0f1;
                border-radius: 5px;
            }
        """)
        # Small image preview
        self.image_display = QLabel()
        self.image_display.setFixedSize(120, 120)
        self.image_title = QLabel("Image Preview")
        self.image_display.setStyleSheet("""
            QLabel {
                border: 2px solid #ccc;
                border-radius: 5px;
                background-color: #f8f9fa;
            }
        """)
        self.image_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        controls_layout.addWidget(self.load_image_btn)
        controls_layout.addWidget(self.classify_btn)
        controls_layout.addWidget(self.image_display, alignment=Qt.AlignmentFlag.AlignHCenter)
        controls_layout.addWidget(self.result_label)
        # Classification plot
        self.figure = Figure(figsize=(4, 3))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setFixedSize(300, 300) 
        self.init_plot()
        layout.addWidget(self.canvas)
        layout.addLayout(controls_layout)
        # Load placeholder image
        self._load_placeholder()

    def init_plot(self):
        """Initialize empty probability plot"""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.set_title('Class Probabilities', fontsize=10)
        ax.set_xlabel('Class', fontsize=8)
        ax.set_ylabel('Probability', fontsize=8)
        ax.tick_params(axis='both', labelsize=8)
        ax.set_ylim(0, 1)
        self.figure.tight_layout()
        self.canvas.draw()

    def update_plot(self, classes, probabilities):
        """Update probability plot with new data"""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        bars = ax.bar(classes, probabilities)
        # Add value labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.0%}',
                   ha='center', va='bottom')
        ax.set_title('Class Probabilities', fontsize=10)
        ax.set_xlabel('Class', fontsize=8)
        ax.set_ylabel('Probability', fontsize=8)
        ax.tick_params(axis='both', labelsize=8)
        ax.set_ylim(0, 1.2)  # Make room for percentage labels
        plt.setp(ax.get_xticklabels(), rotation=45)
        self.figure.tight_layout()
        self.canvas.draw()

    def _load_placeholder(self):
        placeholder_path = os.path.join(os.path.dirname(__file__), "..", "placeholder_img.jpg")
        if os.path.exists(placeholder_path):
            pixmap = QPixmap(placeholder_path)
            self.image_display.setPixmap(pixmap.scaled(
                120, 120,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            ))
        else:
            self.image_display.setText("No image")

    def update_image(self, pixmap):
        """Method for updating the displayed image"""
        if pixmap:
            scaled_pixmap = pixmap.scaled(
                120, 120,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.image_display.setPixmap(scaled_pixmap)
        else:
            self.image_display.setText("No image")


class MetricsWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout()
        self.setLayout(layout)
        layout.setContentsMargins(0, 0, 0, 0)
        metrics_grid = QGridLayout()
        metrics_grid.setSpacing(10)
        self.accuracy_label = QLabel("Accuracy: -")
        self.precision_label = QLabel("Precision: -")
        self.recall_label = QLabel("Recall: -")
        metric_style = """
            QLabel {
                font-size: 12px;
                padding: 8px;
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 4px;
            }
        """
        for label in [self.accuracy_label, self.precision_label, self.recall_label]:
            label.setStyleSheet(metric_style)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        metrics_grid.addWidget(self.accuracy_label, 0, 0)
        metrics_grid.addWidget(self.precision_label, 0, 1)
        metrics_grid.addWidget(self.recall_label, 0, 2)
        layout.addLayout(metrics_grid)

    def update_metrics(self, metrics):
        """Updates the displayed metrics"""
        self.accuracy_label.setText(f"Accuracy: {metrics['accuracy']:.2%}")
        self.precision_label.setText(f"Precision: {metrics['precision']:.2%}")
        self.recall_label.setText(f"Recall: {metrics['recall']:.2%}")

    def reset_metrics(self):
        """Resets all metrics to their initial state"""
        self.accuracy_label.setText("Accuracy: -")
        self.precision_label.setText("Precision: -")
        self.recall_label.setText("Recall: -")

class PytorchTab(QWidget):
    def __init__(self):
        super().__init__()
        self._create_ui()
        self.current_image_path = None
        # Model Initialization 
        self.model = DiceTossModel()
        self.model_loaded = False
        # Update model status
        self.update_model_status("No model loaded") 
        # Try to load the dataset and default model on startup
        self._try_load_dataset_and_default_model()
        # Connect signals
        self._setup_connections()    

    def _setup_connections(self):
        self.image_widget.load_image_btn.clicked.connect(self.load_image)
        self.image_widget.classify_btn.clicked.connect(self.classify_image)
        self.load_model_btn.clicked.connect(self.load_model)
        self.save_model_btn.clicked.connect(self.save_model)
        self.train_model_btn.clicked.connect(self.train_model)
        self.test_model_btn.clicked.connect(self.test_model)
        self.clear_model_btn.clicked.connect(self.clear_model)

    def update_metrics_display(self, metrics):
        """Method to update the metrics"""
        self.accuracy_label.setText(f"Accuracy: {metrics['accuracy']:.2%}")
        self.precision_label.setText(f"Precision: {metrics['precision']:.2%}")
        self.recall_label.setText(f"Recall: {metrics['recall']:.2%}")

    def _update_ui_from_model_data(self):
        """Updates UI elements with model data"""
        # Update parameter widgets if training params exist
        if self.model.training_params['epochs'] is not None:
            self.epochs_widget.spinbox.setValue(self.model.training_params['epochs'])
            self.learning_rate_widget.spinbox.setValue(self.model.training_params['learning_rate'])
            self.momentum_widget.spinbox.setValue(self.model.training_params['momentum'])    
        # Update metrics display if metrics exist
        if self.model.metrics['accuracy'] is not None:
            self.metrics_widget.update_metrics(self.model.metrics)
        else:
            self.metrics_widget.reset_metrics()
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
        """Clears the current model and resets UI elements while preserving dataset"""   
        # Show confirmation dialog
        reply = QMessageBox.question(
            self,
            'Clear Model',
            'Are you sure you want to clear the model? This will reset all charts, metrics and parameters.',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            dataset_loaded = self.model.is_data_loaded()
            trainloader = self.model.trainloader
            valloader = self.model.valloader
            testloader = self.model.testloader
            # Reset model
            self.model = DiceTossModel()
            self.model.initialize_model()
            # Restore dataset if it was loaded
            if dataset_loaded:
                self.model.dataset_loaded = dataset_loaded
                self.model.trainloader = trainloader
                self.model.valloader = valloader
                self.model.testloader = testloader
            self.model_loaded = False
            # Reset UI elements
            self.update_model_status("No model loaded", "red")
            # Reset parameters to defaults
            self.epochs_widget.spinbox.setValue(10)
            self.learning_rate_widget.spinbox.setValue(0.001)
            self.momentum_widget.spinbox.setValue(0.9)
            # Reset metrics
            self.metrics_widget.reset_metrics()
            # Reset plots
            self.plot_widget1.plot_loss_history(self.plot_widget1)
            self.plot_widget2.plot_confusion_matrix(self.plot_widget2)
            # Reset image classification
            self.image_widget.result_label.setText("Classification: None")
            self.image_widget.init_plot()
            # Disable buttons
            self.save_model_btn.setEnabled(False)
            self.test_model_btn.setEnabled(False)
            self.status_bar.showMessage("Model cleared successfully", 8000)

    def load_image(self):
        supported_formats = [f"*.{fmt.data().decode()}" for fmt in QImageReader.supportedImageFormats()]
        filter_string = "Image Files ({})".format(" ".join(supported_formats))
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image",
            "",
            filter_string
        )
        if file_path:
            pixmap = QPixmap(file_path)
            if not pixmap.isNull():
                self.current_image_path = file_path
                self.image_widget.update_image(pixmap)
                self.image_widget.result_label.setText("Classification: None")
                self.status_bar.showMessage(f"Loaded image: {os.path.basename(file_path)}", 8000)
            else:
                self.image_widget.image_display.setText("Failed to load image")
                self.current_image_path = None
                self.status_bar.showMessage("Failed to load image", 8000)

    def update_model_status(self, status, color="red"):
        self.model_status.setText(status)
        self.model_status.setStyleSheet(f"""
            QLabel {{
                font-weight: bold;
                color: {color};
                padding: 5px;
            }}
        """)

    def classify_image(self):
        if not self.current_image_path:
            self.status_bar.showMessage("No image loaded", 8000)
            return 
        if not self.model_loaded:
            self.status_bar.showMessage("No model loaded", 8000)
            return 
        try:
            result = self.model.predict_image(self.current_image_path)
            predicted_class = result['class']
            probabilities = result['probabilities']
            
            # Update UI
            self.image_widget.result_label.setText(f"Classification: {predicted_class}")
            self.image_widget.update_plot(self.model.classes, probabilities)
            
            self.status_bar.showMessage("Classification complete", 8000)
        except Exception as e:
            self.status_bar.showMessage(f"Classification error: {str(e)}", 8000)

    def _try_load_dataset_and_default_model(self):
        """Attempts to load the dataset and the default model on startup"""
        dataset_message = ""
        # Attempt to load dataset
        try:
            train_size, val_size, test_size = self.model.load_data("./models/dicetoss_small")
            dataset_message = f"Dataset loaded: {train_size} train, {val_size} val, {test_size} test. "
        except Exception as e:
            dataset_message = f"Error loading dataset: {str(e)}. "
        # Attempt to load default model
        try:
            self.model.initialize_model()
            default_model_path = "./models/default_model.pth" 
            if os.path.exists(default_model_path):
                try:
                    metadata = self.model.load_model(default_model_path)
                    # Update UI with loaded data
                    self._update_ui_from_model_data()
                    self.model.load_model(default_model_path)
                    self.model_loaded = True
                    self.update_model_status("Model loaded successfully", "green")
                    dataset_message += "Default model loaded successfully."
                    self.save_model_btn.setEnabled(True)
                    self.test_model_btn.setEnabled(True)
                except Exception as e:
                    dataset_message += f"Error loading default model: {str(e)}"
                    self.update_model_status("Error loading default model", "red")
            else:
                dataset_message += "No default model found."
                self.update_model_status("No model loaded", "red")
        except Exception as e:
            dataset_message += f"Error initializing model: {str(e)}"
            self.update_model_status("Error initializing model", "red")     
        # Show status message
        self.status_bar.showMessage(dataset_message, 10000)

    def load_model(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Model",
            "",
            "PyTorch Model (*.pth)"
        )
        if file_path:
            try:
                self.model.initialize_model()
                # Load model and get metadata
                metadata = self.model.load_model(file_path)
                # Reset all metrics and plots first
                self.metrics_widget.reset_metrics()
                self.plot_widget1.plot_loss_history(self.plot_widget1)
                self.plot_widget2.plot_confusion_matrix(self.plot_widget2)
                # Update UI with loaded data only if the data exists
                if metadata['training_params']['epochs'] is not None:
                    self._update_ui_from_model_data()
                self.model_loaded = True
                self.save_model_btn.setEnabled(True)
                self.test_model_btn.setEnabled(True)
                self.update_model_status("Model loaded successfully", "green")
                # Customize status message based on whether metrics exist
                if (metadata['metrics']['accuracy'] is not None):
                    self.status_bar.showMessage(f"Model loaded successfully (Accuracy: {metadata['metrics']['accuracy']:.2%})", 8000)
                else:
                    self.status_bar.showMessage("Model loaded successfully (untested model)", 8000)
            except Exception as e:
                self.status_bar.showMessage(f"Error loading model: {str(e)}", 8000)

    def save_model(self):
        if not self.model_loaded:
            self.status_bar.showMessage("No model to save", 8000)
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
                self.status_bar.showMessage("Model saved successfully", 8000)
            except Exception as e:
                self.status_bar.showMessage(f"Error saving model: {str(e)}", 8000)

    def train_model(self):
        epochs = self.epochs_widget.spinbox.value()
        learning_rate = self.learning_rate_widget.spinbox.value()
        momentum = self.momentum_widget.spinbox.value()
        try:
            # Model Initialization
            self.model.initialize_model()
            # Reset metrics display
            self.metrics_widget.reset_metrics()
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
                self.test_model_btn.setEnabled(True)
                self.update_model_status("Model trained successfully", "green")
                self.status_bar.showMessage("Training completed", 8000)
            else:
                self.status_bar.showMessage(
                    f"Training error: {getattr(dialog, 'error_message', 'Unknown error')}",
                    8000
                )       
        except Exception as e:
            self.status_bar.showMessage(f"Training error: {str(e)}", 8000)

    def test_model(self):
        if not self.model_loaded:
            self.status_bar.showMessage("No model loaded", 8000)
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
                # Status update
                self.status_bar.showMessage("Testing completed", 8000)
            else:
                self.status_bar.showMessage(
                    f"Testing error: {getattr(dialog, 'error_message', 'Unknown error')}",
                    8000
                ) 
        except Exception as e:
            self.status_bar.showMessage(f"Testing error: {str(e)}", 8000)    

    def _create_ui(self):
        # Main layout with scroll area
        main_layout = QVBoxLayout(self)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_layout = QHBoxLayout(scroll_widget)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll_layout.setSpacing(0)  
        # Left panel
        left_panel = self._create_left_panel()
        scroll_layout.addWidget(left_panel)
        # Right panel (plots)
        right_panel = self._create_right_panel()
        scroll_layout.addWidget(right_panel)
        # Set up scroll area
        scroll_area.setWidget(scroll_widget)
        main_layout.addWidget(scroll_area)
        # Add status bar
        self.status_bar = QStatusBar()
        self.status_bar.setSizeGripEnabled(False)
        main_layout.addWidget(self.status_bar)

    def _create_left_panel(self):
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(10)
        left_layout.setContentsMargins(10, 10, 15, 10)
        left_panel.setFixedWidth(500)
        # Model Controls
        model_group = QGroupBox("Model Controls")
        model_layout = QVBoxLayout()
        # Model buttons
        buttons_layout = QHBoxLayout()
        self.load_model_btn = QPushButton("Load Model")
        self.train_model_btn = QPushButton("Train Model")
        self.save_model_btn = QPushButton("Save Model")
        self.save_model_btn.setEnabled(False)
        self.test_model_btn = QPushButton("Test Model")
        self.test_model_btn.setEnabled(False)
        # Add buttons to layout
        for btn in [self.load_model_btn, self.train_model_btn, self.save_model_btn, self.test_model_btn]:
            btn.setStyleSheet("""
                QPushButton {
                    padding: 8px;
                    background-color: #f8f9fa;
                    border: 1px solid #dee2e6;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #e9ecef;
                }
                QPushButton:disabled {
                    background-color: #e9ecef;
                    color: #6c757d;
                }
            """)
            buttons_layout.addWidget(btn)
        model_layout.addLayout(buttons_layout)
        # Status layout with Clear button
        status_layout = QHBoxLayout()
        self.model_status = QLabel("No model loaded")
        self.model_status.setStyleSheet("""
            QLabel {
                font-weight: bold;
                color: #dc3545;
                padding: 5px;
            }
        """)
        status_layout.addWidget(self.model_status)
        # Clear Model button with distinct style
        self.clear_model_btn = QPushButton("Clear Model")
        self.clear_model_btn.setFixedWidth(107)  # Make button smaller
        self.clear_model_btn.setStyleSheet("""
            QPushButton {
                padding: 5px;
                background-color: #f8f9fa;
                border: 1px solid #6c757d;
                border-radius: 4px;
                color: #3d4145;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #e9ecef;
            }
            QPushButton:pressed {
                background-color: #dee2e6;
            }
        """)
        status_layout.addWidget(self.clear_model_btn)
        model_layout.addLayout(status_layout)
        
        model_group.setLayout(model_layout)
        # Parameters
        params_group = QGroupBox("Parameters")
        params_layout = QVBoxLayout()
        # Create parameter widgets
        self.epochs_widget = ParameterWidget("Epochs:", 1, 1000, 10)
        self.learning_rate_widget = ParameterWidget("Learning Rate:", 0.000001, 1.0, 0.001, 6)
        self.momentum_widget = ParameterWidget("Momentum:", 0.0, 1.0, 0.9, 6)
        for widget in [self.epochs_widget, self.learning_rate_widget, self.momentum_widget]:
            params_layout.addWidget(widget)
        params_group.setLayout(params_layout)
        # Image Classification
        self.image_widget = ImageClassificationWidget()
        image_group = QGroupBox("Image Classification")
        image_layout = QVBoxLayout()
        image_layout.addWidget(self.image_widget)
        image_group.setLayout(image_layout)
        # Metrics Group Box
        self.metrics_widget = MetricsWidget()
        metrics_group = QGroupBox("Model Metrics")
        metrics_layout = QVBoxLayout()
        metrics_layout.addWidget(self.metrics_widget)
        metrics_group.setLayout(metrics_layout)
        # Add all components to left panel
        for widget in [model_group, params_group, image_group, metrics_group]:
            widget.setStyleSheet("""
                QGroupBox {
                    font-weight: bold;
                    border: 1px solid #c4c8cc;
                    border-radius: 6px;
                    margin-top: 20px;
                    padding: 5px;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 0px;
                    padding: 0 3px 0 3px;
                }
            """)
            left_layout.addWidget(widget)
        left_layout.addStretch()
        return left_panel

    def _create_right_panel(self):
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(20)  # Increased spacing between plots
        right_layout.setContentsMargins(15, 30, 10, 10)
        # Plot widgets
        self.plot_widget1 = PlotWidget("Loss History") 
        self.plot_widget2 = PlotWidget("Confusion Matrix")  
        # Initialize empty Confusion Matrix
        self.plot_widget1.plot_loss_history(self.plot_widget1)
        self.plot_widget2.plot_confusion_matrix(self.plot_widget2)
        # Add to layout
        right_layout.addWidget(self.plot_widget1)
        right_layout.addWidget(self.plot_widget2)
        right_layout.addStretch()
        return right_panel