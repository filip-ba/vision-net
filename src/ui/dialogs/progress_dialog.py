from PyQt6.QtWidgets import QDialog, QProgressBar, QVBoxLayout, QLabel, QPushButton
from PyQt6.QtCore import Qt, QThread, pyqtSignal
import time


class TrainingThread(QThread):
    """Class for model training (and testing) in a separate thread"""
    
    progress_updated = pyqtSignal(float, float)  # (progress, loss)
    training_finished = pyqtSignal(tuple)  # (train_loss_history, val_loss_history)
    testing_finished = pyqtSignal(dict)  # metrics dictionary
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

            # Start training
            result = self.model.train(
                self.epochs,
                self.learning_rate,
                self.momentum,
                progress_callback
            )
            
            if self._is_canceled:
                return
                
            # Emit signal when training is finished
            self.training_finished.emit(result)
            
            # Start testing
            if not self._is_canceled:
                try:
                    metrics = self.model.test()
                    self.testing_finished.emit(metrics)
                except Exception as e:
                    self.error.emit(f"Testing error: {str(e)}")
                    
        except Exception as e:
            self.error.emit(str(e))

    def cancel(self):
        self._is_canceled = True


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
        self.training_result = None
        self.testing_result = None
        self._create_ui()
        
    def _create_ui(self):
        self.setWindowTitle("Training...")
        self.resize(400, 300)
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowMinimizeButtonHint)
        # Make dialog non-modal
        self.setModal(False)

        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        if self.operation_type == "Training" and all(param is not None for param in 
            [self.epochs, self.learning_rate, self.momentum]):
            params_label = QLabel(
                f"Training Parameters:\n"
                f"Epochs: {self.epochs}\n"
                f"Learning Rate: {self.learning_rate}\n"
                f"Momentum: {self.momentum}"
            )
            params_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
            params_label.setObjectName("training-dialog-parameters")
            layout.addWidget(params_label)

        self.status_label = QLabel(f"{self.operation_type} in progress...")
        self.status_label.setObjectName("dialog-status-label")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)

        self.time_label = QLabel("Estimated time remaining: calculating...")
        self.time_label.setObjectName("dialog-time-label")
        self.time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.time_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.cancel)
        layout.addWidget(self.cancel_button)
        self.setLayout(layout)

    def start_training(self, model, epochs, learning_rate, momentum):
        self.thread = TrainingThread(model, epochs, learning_rate, momentum)
        self.thread.progress_updated.connect(self.update_progress)
        self.thread.training_finished.connect(self.on_training_finished)
        self.thread.testing_finished.connect(self.on_testing_finished)
        self.thread.error.connect(self.on_error)
        self.start_time = time.time()
        self.thread.start()
        self.show()
        return self.training_result, self.testing_result

    def update_progress(self, progress, current_loss):
        """Updates progress bar and time estimate"""
        progress_percent = int(progress * 100)
        self.progress_bar.setValue(progress_percent)

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
        if self.thread:
            self.thread.cancel()
        self.close()

    def on_training_finished(self, result):
        self.training_result = result
        self.status_label.setText("Training completed. Starting testing...")
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximum(0)  # Switch to indeterminate mode for testing
        self.time_label.setText("Testing in progress...")

    def on_testing_finished(self, metrics):
        self.testing_result = metrics
        self.status_label.setText("Testing completed.")
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(100)
        self.time_label.setText("Done!")
        self.close()

    def on_error(self, error_message):
        self.training_result = None
        self.testing_result = None
        self.error_message = error_message
        self.close()