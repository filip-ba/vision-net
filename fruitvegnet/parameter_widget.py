from PyQt6.QtWidgets import QWidget, QHBoxLayout, QLabel, QSlider, QDoubleSpinBox, QSpinBox
from PyQt6.QtCore import Qt


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