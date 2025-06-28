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
        self.model_labels = {}
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
        self.model_labels = {}
        model_items = list(self.model_names.items())
        for i, (model_id, model_name) in enumerate(model_items):
            label = QLabel(model_name)
            label.setObjectName(f"Model")
            label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
            label.setMinimumWidth(70)
            self.model_labels[model_id] = label
            left_column.addWidget(label, 1)

            if i < len(model_items) - 1:
                left_column.addWidget(create_separator("horizontal"))

        # Create result labels
        self.result_labels = {}
        for i, model_id in enumerate(self.model_names.keys()):
            result_label = QLabel("") 
            result_label.setStyleSheet("font-weight: 700;")
            result_label.setObjectName("ModelResultLabel")
            result_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            result_label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
            result_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
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
        if model_type in self.result_labels:
            model_label = self.model_labels[model_type]
            result_label = self.result_labels[model_type]
            
            if result == "No model":
                result_label.setText("")
                model_label.setToolTip("No model of this architecture is loaded.")
            else:
                result_label.setText(result)
                model_label.setToolTip("")
    
    def reset_results(self):
        for model_id in self.result_labels:
            self.result_labels[model_id].setText("") 
            self.result_labels[model_id].setStyleSheet("font-weight: 700;")
            if model_id in self.model_labels:
                self.model_labels[model_id].setToolTip("") 