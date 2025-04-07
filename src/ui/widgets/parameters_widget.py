from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt

from src.utils.custom_separator import create_separator


class ParametersWidget(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout()
        layout.setSpacing(18)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.time_label = QLabel("Training Time: -")
        self.time_label.setObjectName("ModelTimeLabel")
        self.time_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse | Qt.TextInteractionFlag.TextSelectableByKeyboard)
        
        self.epochs_label = QLabel("Epochs: -")
        self.epochs_label.setObjectName("ModelEpochsLabel")
        self.epochs_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse | Qt.TextInteractionFlag.TextSelectableByKeyboard)
        
        self.lr_label = QLabel("Learning Rate: -")
        self.lr_label.setObjectName("ModelLrLabel")
        self.lr_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse | Qt.TextInteractionFlag.TextSelectableByKeyboard)
        
        self.momentum_label = QLabel("Momentum: -")
        self.momentum_label.setObjectName("ModelMomentumLabel")
        self.momentum_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse | Qt.TextInteractionFlag.TextSelectableByKeyboard)
        
        layout.addWidget(self.time_label)
        layout.addWidget(create_separator("horizontal"))
        layout.addWidget(self.epochs_label)
        layout.addWidget(create_separator("horizontal"))
        layout.addWidget(self.lr_label)
        layout.addWidget(create_separator("horizontal"))
        layout.addWidget(self.momentum_label)

        self.setLayout(layout)
  
    def update_parameters(self, params):
        if params.get('epochs') is not None:
            self.epochs_label.setText(f"Epochs: {params['epochs']}")
        
        if params.get('learning_rate') is not None:
            self.lr_label.setText(f"Learning Rate: {params['learning_rate']}")
        
        if params.get('momentum') is not None:
            self.momentum_label.setText(f"Momentum: {params['momentum']}")
        
        if params.get('training_time') is not None:
            time_seconds = params['training_time']
            minutes = int(time_seconds // 60)
            seconds = int(time_seconds % 60)
            self.time_label.setText(f"Training Time: {minutes}m {seconds}s")

    def reset_parameters(self):
        self.time_label.setText("Training Time: -")
        self.epochs_label.setText("Epochs: -")
        self.lr_label.setText("Learning Rate: -")
        self.momentum_label.setText("Momentum: -")