from PyQt6.QtWidgets import ( QDialog, QVBoxLayout, QGroupBox, QHBoxLayout, QDialogButtonBox,
                            QWidget, QHBoxLayout, QLabel, QSlider, QDoubleSpinBox, QSpinBox )
from PyQt6.QtCore import Qt


class ParametersWidget(QWidget):

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
        self.spinbox.setFixedWidth(100)
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

        # Signals
        self._setup_connections()

    def _setup_connections(self):
        self.slider.valueChanged.connect(self._slider_changed)
        self.spinbox.valueChanged.connect(self._spinbox_changed)
    
    def _slider_changed(self, value):
        """Changes value on spinbox if the slider changes value"""
        if isinstance(self.spinbox, QDoubleSpinBox):
            self.spinbox.setValue(value / (10 ** self.spinbox.decimals()))
        else:
            self.spinbox.setValue(value)
    
    def _spinbox_changed(self, value):
        """Changes value on slider if the spinbox changes value"""
        if isinstance(self.spinbox, QDoubleSpinBox):
            self.slider.setValue(int(value * (10 ** self.spinbox.decimals())))
        else:
            self.slider.setValue(int(value))


class ParametersDialog(QDialog):
    """Creates dialog with parameters for training"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Training Parameters")
        self.resize(500, 300)
        
        main_layout = QVBoxLayout(self)
        
        # Parameters group
        params_group = QGroupBox("")
        params_group.setObjectName("params-group")
        params_layout = QVBoxLayout()
        
        # Create parameter widgets
        self.epochs_widget = ParametersWidget("Epochs:", 1, 100, 10)
        self.learning_rate_widget = ParametersWidget("Learning Rate:", 0.000001, 1.0, 0.001, 6)
        self.momentum_widget = ParametersWidget("Momentum:", 0.0, 1.0, 0.9, 6)
        
        # Add parameter widgets to layout
        for widget in [self.epochs_widget, self.learning_rate_widget, self.momentum_widget]:
            params_layout.addWidget(widget)
        
        params_group.setLayout(params_layout)
        params_layout.setContentsMargins(10, 10, 10, 10)

        # Add the group box to the main layout
        main_layout.addWidget(params_group)
        
        # Dialog buttons
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        
        main_layout.addWidget(button_box)
    
    def get_parameters(self):
        """Returns the current parameter values as a dictionary"""
        return {
            'epochs': self.epochs_widget.spinbox.value(),
            'learning_rate': self.learning_rate_widget.spinbox.value(),
            'momentum': self.momentum_widget.spinbox.value()
        }
    
    def set_parameters(self, epochs=10, learning_rate=0.001, momentum=0.9):
        """Sets the parameter values from the provided arguments"""
        self.epochs_widget.spinbox.setValue(epochs)
        self.learning_rate_widget.spinbox.setValue(learning_rate)
        self.momentum_widget.spinbox.setValue(momentum)