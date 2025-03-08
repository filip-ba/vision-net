from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QGridLayout
from PyQt6.QtCore import Qt
from utils.custom_separator import create_separator


class MetricsWidget(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout()
        self.setLayout(layout)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Add model name label at the top
        self.model_name_label = QLabel("Model: -")
        title_style = """
            QLabel {
                font-size: 16px;
                font-weight: bold;
                padding: 5px;
                color: #333;
            }
        """
        self.model_name_label.setStyleSheet(title_style)
        self.model_name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.model_name_label)

        # Metrics grid for accuracy, precision, recall
        metrics_grid = QGridLayout()
        metrics_grid.setSpacing(10)

        self.accuracy_label = QLabel("Accuracy: -")
        self.precision_label = QLabel("Precision: -")
        self.recall_label = QLabel("Recall: -")

        metric_style = """
            QLabel {
                font-size: 14px;
                padding: 8px;
                background-color: #f0f0f0;
                border: 1px solid #bbb;
                border-radius: 5px;
                font-weight: 600;
            }
        """
        for label in [self.accuracy_label, self.precision_label, self.recall_label]:
            label.setStyleSheet(metric_style)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            
        metrics_grid.addWidget(self.accuracy_label, 0, 0)
        metrics_grid.addWidget(self.precision_label, 0, 1)
        metrics_grid.addWidget(self.recall_label, 0, 2)
        metrics_grid.setSpacing(10)
        layout.addLayout(metrics_grid)
        
        # Add training parameters section
        param_style = """
            QLabel {
                font-size: 13px;
                padding: 6px;
                background-color: #ffffff;
                border: 0px solid #ddd;
                border-radius: 4px;
            }
        """
        
        # Training parameters
        parameters_layout = QVBoxLayout()
        parameters_layout.setSpacing(8)
        
        # Add parameters labels
        self.time_label = QLabel("Training Time: -")
        self.epochs_label = QLabel("Epochs: -")
        self.lr_label = QLabel("Learning Rate: -")
        self.momentum_label = QLabel("Momentum: -")
        
        for label in [self.time_label, self.epochs_label, self.lr_label, self.momentum_label]:
            label.setStyleSheet(param_style)
            parameters_layout.addWidget(label)
            parameters_layout.addWidget(create_separator())

        layout.addLayout(parameters_layout)

    def update_metrics(self, metrics):
        self.accuracy_label.setText(f"Accuracy: {metrics['accuracy']:.2%}")
        self.precision_label.setText(f"Precision: {metrics['precision']:.2%}")
        self.recall_label.setText(f"Recall: {metrics['recall']:.2%}")
    
    def update_parameters(self, params):
        """Update training parameters display"""
        if params.get('epochs') is not None:
            self.epochs_label.setText(f"Epochs: {params['epochs']}")
        
        if params.get('learning_rate') is not None:
            self.lr_label.setText(f"Learning Rate: {params['learning_rate']}")
        
        if params.get('momentum') is not None:
            self.momentum_label.setText(f"Momentum: {params['momentum']}")
        
        if params.get('training_time') is not None:
            # Format time as minutes and seconds
            time_seconds = params['training_time']
            minutes = int(time_seconds // 60)
            seconds = int(time_seconds % 60)
            self.time_label.setText(f"Training Time: {minutes}m {seconds}s")
    
    def set_model_name(self, name):
        """Set the model name in the header"""
        self.model_name_label.setText(f"Model: {name}")

    def reset_metrics(self):
        self.accuracy_label.setText("Accuracy: -")
        self.precision_label.setText("Precision: -")
        self.recall_label.setText("Recall: -")
    
    def reset_parameters(self):
        """Reset training parameters display"""
        self.time_label.setText("Training Time: -")
        self.epochs_label.setText("Epochs: -")
        self.lr_label.setText("Learning Rate: -")
        self.momentum_label.setText("Momentum: -")
        self.model_name_label.setText("Model: -")