from PyQt6.QtWidgets import QDialog, QProgressBar, QVBoxLayout, QLabel, QPushButton
from PyQt6.QtCore import Qt, QThread, pyqtSignal
import time


class TrainingThread(QThread):
    """Class for model training in a separate thread"""
    progress_updated = pyqtSignal(float, float)  # (progress, loss)
    finished = pyqtSignal(tuple)  # (train_loss_history, val_loss_history)
    error = pyqtSignal(str)

    def __init__(self, model, epochs, learning_rate, momentum):
        super().__init__()
        self.model = model
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.momentum = momentum
        self._is_canceled = False

    def run(self):
        try:
            def progress_callback(progress, loss):
                if self._is_canceled:
                    raise InterruptedError("Training canceled by user")
                self.progress_updated.emit(progress, loss)

            result = self.model.train(
                self.epochs,
                self.learning_rate,
                self.momentum,
                progress_callback
            )
            if not self._is_canceled:
                self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))

    def cancel(self):
        self._is_canceled = True


class TestingThread(QThread):
    """Class for model testing in a separate thread"""
    finished = pyqtSignal(dict)  # metrics dictionary
    error = pyqtSignal(str)

    def __init__(self, model):
        super().__init__()
        self.model = model

    def run(self):
        try:
            metrics = self.model.test()
            self.finished.emit(metrics)
        except Exception as e:
            self.error.emit(str(e))


class ProgressDialog(QDialog):
    def __init__(self, parent=None, operation_type="Training", epochs=None, 
                 learning_rate=None, momentum=None):
        super().__init__(parent)
        self.operation_type = operation_type
        self.start_time = None
        self.thread = None
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.setup_ui()
        
    def setup_ui(self):
        """UI component initialization"""
        self.setWindowTitle(f"Model {self.operation_type}")
        self.setFixedSize(400, 300)  
        self.setModal(True)
        layout = QVBoxLayout()
        # Add training parameters if available
        if self.operation_type == "Training" and all(param is not None for param in 
            [self.epochs, self.learning_rate, self.momentum]):
            params_label = QLabel(
                f"Training Parameters:\n"
                f"Epochs: {self.epochs}\n"
                f"Learning Rate: {self.learning_rate}\n"
                f"Momentum: {self.momentum}"
            )
            params_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
            params_label.setStyleSheet("""
                QLabel {
                    background-color: #f8f9fa;
                    padding: 10px;
                    border-radius: 5px;
                    border: 1px solid #dee2e6;
                }
            """)
            layout.addWidget(params_label)
        # Status label
        self.status_label = QLabel(f"{self.operation_type} in progress...")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)
        # Time estimate label
        self.time_label = QLabel("Estimated time remaining: calculating...")
        self.time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.time_label)
        # Cancel button
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.cancel)
        layout.addWidget(self.cancel_button)
        self.setLayout(layout)

    def start_training(self, model, epochs, learning_rate, momentum):
        """Starts training in new thread"""
        self.thread = TrainingThread(model, epochs, learning_rate, momentum)
        self.thread.progress_updated.connect(self.update_progress)
        self.thread.finished.connect(self.on_training_finished)
        self.thread.error.connect(self.on_error)
        self.start_time = time.time()
        self.thread.start()
        self.exec()  
        return getattr(self, 'result', None)

    def start_testing(self, model):
        """Starts training in new thread"""
        self.thread = TestingThread(model)
        self.thread.finished.connect(self.on_testing_finished)
        self.thread.error.connect(self.on_error)
        self.start_time = time.time()
        self.progress_bar.setMaximum(0) 
        self.thread.start()
        self.exec()  
        return getattr(self, 'result', None)

    def update_progress(self, progress, current_loss):
        """Updates progress bar and time estimate"""
        progress_percent = int(progress * 100)
        self.progress_bar.setValue(progress_percent)
        # Calculating the remaining time
        if self.start_time and progress > 0:
            elapsed_time = time.time() - self.start_time
            estimated_total_time = elapsed_time / progress
            remaining_time = estimated_total_time - elapsed_time
            minutes = int(remaining_time // 60)
            seconds = int(remaining_time % 60)
            time_str = f"{minutes}m {seconds}s"
            status_text = f"{self.operation_type} in progress... "
            if current_loss is not None:
                status_text += f"(Loss: {current_loss:.4f})"
            self.status_label.setText(status_text)
            self.time_label.setText(f"Estimated time remaining: {time_str}")

    def cancel(self):
        """Cancels the operation"""
        if isinstance(self.thread, TrainingThread):
            self.thread.cancel()
        self.reject()

    def on_training_finished(self, result):
        """Training finished callback"""
        self.result = result
        self.accept()

    def on_testing_finished(self, metrics):
        """Testing finished callback"""
        self.result = metrics
        self.accept()

    def on_error(self, error_message):
        """Error callback"""
        self.result = None
        self.error_message = error_message
        self.reject()