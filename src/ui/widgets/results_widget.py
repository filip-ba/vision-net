from PyQt6.QtWidgets import QWidget, QGroupBox, QVBoxLayout, QHBoxLayout, QLabel, QSizePolicy
from PyQt6.QtCore import Qt

from ...utils.custom_separator import create_separator


class ResultsWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.model_names = {
            'simple_cnn': 'Simple CNN',
            'resnet': 'ResNet',
            'efficientnet': 'EfficientNet',
            'vgg16': 'VGG 16'
        }
        self.result_labels = {}
        self._create_ui()

    def _create_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        group_box = QGroupBox("Results")
        group_box.setObjectName("results-group")
        group_box.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        group_box.setMinimumWidth(100)

        results_main_layout = QVBoxLayout(group_box)
        results_main_layout.setContentsMargins(0, 18, 0, 18)
        
        results_columns_layout = QHBoxLayout()
        results_columns_layout.setContentsMargins(0, 0, 0, 0)
        
        left_column = QVBoxLayout()
        left_column.setSpacing(18)
        left_column.setContentsMargins(0, 0, 0, 0)

        right_column = QVBoxLayout()
        right_column.setSpacing(18)
        right_column.setContentsMargins(0, 0, 0, 0)

        model_labels = {
            'simple_cnn': QLabel("Simple CNN"),
            'resnet': QLabel("ResNet"),
            'efficientnet': QLabel("EfficientNet"),
            'vgg16': QLabel("VGG 16")
        }
        
        # Add model labels to the left column
        for model_id, label in model_labels.items():
            label.setObjectName(f"Model{model_id.title().replace('_', '')}")
            label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
            label.setMinimumWidth(70)
            left_column.addWidget(label, 1)
            
            # Add separator after each label except the last one
            if model_id != list(self.model_names.keys())[-1]:
                left_column.addWidget(create_separator("horizontal"))

        # Create result labels
        self.result_labels = {}
        for i, model_id in enumerate(self.model_names.keys()):
            result_label = QLabel("None") 
            result_label.setStyleSheet("font-weight: 700;")
            result_label.setObjectName("ModelResultLabel")
            result_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            result_label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
            result_label.setMinimumWidth(70)
            self.result_labels[model_id] = result_label
            right_column.addWidget(result_label, 1)
            
            # Add separator after each except the last one
            if model_id != list(self.model_names.keys())[-1]:
                right_column.addWidget(create_separator("horizontal"))
        
        results_columns_layout.addLayout(left_column)
        results_columns_layout.addWidget(create_separator("vertical"))
        results_columns_layout.addLayout(right_column)

        results_main_layout.addLayout(results_columns_layout)
        layout.addWidget(group_box)
        
        self.setLayout(layout)
        
    def update_result(self, model_type, result):
        """Update the classification result for a specific model"""
        if model_type in self.result_labels:
            self.result_labels[model_type].setText(result)
            
    def reset_results(self):
        """Reset all result labels to None"""
        for model_id in self.result_labels:
            self.result_labels[model_id].setText("None") 