from PyQt6.QtWidgets import QDialog, QProgressBar, QVBoxLayout, QLabel, QPushButton, QMessageBox
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
            
            # Start testing automatically after training
            if not self._is_canceled:
                try:
                    metrics = self.model.test()
                    if metrics is None:
                        raise ValueError("Testing returned no results")
                    if 'accuracy' not in metrics:
                        raise ValueError("Testing results do not contain accuracy metric")
                    self.testing_finished.emit(metrics)
                except Exception as e:
                    self.error.emit(f"Testing error: {str(e)}")
                    
        except InterruptedError as e:
            self.error.emit(str(e))
        except Exception as e:
            self.error.emit(f"Training error: {str(e)}")

    def cancel(self):
        self._is_canceled = True


class ProgressDialog(QDialog):
    complete = pyqtSignal(tuple, dict)  # signals to emit results to the main window (training_result, testing_result)
    error_occurred = pyqtSignal(str)

    def __init__(self, parent=None, epochs=None, 
                 learning_rate=None, momentum=None, model_name=None):
        super().__init__(parent)
        self.thread = None
        self.start_time = None
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.model_name = model_name
        self.training_result = None
        self.testing_result = None
        self._create_ui()
        
    def _create_ui(self):
        title = "Training..."
        if self.model_name:
            title = f"Training {self.model_name}"
        self.setWindowTitle(title)
        self.resize(400, 300)
        # Set window flags to allow minimization and make it stay on top
        self.setWindowFlags(Qt.WindowType.Window | 
                           Qt.WindowType.WindowMinimizeButtonHint | 
                           Qt.WindowType.WindowTitleHint |
                           Qt.WindowType.CustomizeWindowHint |
                           Qt.WindowType.WindowStaysOnTopHint )

        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        if all(param is not None for param in 
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

        self.status_label = QLabel("Training in progress...")
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

    def update_progress(self, progress, current_loss):
        """Updates progress bar and time estimate"""
        progress_percent = int(progress * 100)
        self.progress_bar.setValue(progress_percent)

        status_text = "Training in progress... "
        if current_loss is not None:
            status_text += f"(Loss: {current_loss:.4f})"
        self.status_label.setText(status_text)
        
        if self.start_time:
            elapsed_time = time.time() - self.start_time
            if progress < 0.001: 
                estimated_total_time = elapsed_time * 100
            else:
                estimated_total_time = elapsed_time / progress
                
            remaining_time = estimated_total_time - elapsed_time
            minutes = int(remaining_time // 60)
            seconds = int(remaining_time % 60)
            time_str = f"{minutes}m {seconds}s"
            self.time_label.setText(f"Estimated time remaining: {time_str}")

    def cancel(self):
        # Zavřeme okno, což vyvolá closeEvent, kde se zobrazí dialog
        self.close()

    def closeEvent(self, event):
        # Pokud je trénování stále v běhu, zobraz potvrzovací dialog
        if self.thread and self.thread.isRunning():
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("Cancel")
            msg_box.setText("Are you sure you want to cancel the training?")
            msg_box.setInformativeText("The training process will be terminated and all progress will be lost.")
            msg_box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            msg_box.setDefaultButton(QMessageBox.StandardButton.No)
            msg_box.setObjectName("training-cancel-dialog")

            reply = msg_box.exec()
            
            if reply == QMessageBox.StandardButton.Yes:
                if self.thread:
                    self.thread.cancel()
                self.error_occurred.emit("Training canceled by user")
                event.accept()  # Povolí zavření okna
            else:
                event.ignore()  # Zabrání zavření okna
        else:
            # Pokud trénování není v běhu, zavři dialog bez potvrzení
            event.accept()

    def on_training_finished(self, result):
        """Changes design of the dialog when training is finished and saves the training result"""
        self.training_result = result
        self.status_label.setText("Training completed. Starting testing...")
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximum(0)
        self.time_label.setText("Testing in progress...")

    def on_testing_finished(self, metrics): 
        """Saves the testing result and emits it with training result to the main window"""
        self.testing_result = metrics
        self.status_label.setText("Testing completed.")
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(100)
        self.time_label.setText("Done!")
        self.complete.emit(self.training_result, self.testing_result)
        self.close()

    def on_error(self, error_message):
        self.training_result = None
        self.testing_result = None
        self.error_message = error_message
        self.error_occurred.emit(error_message)
        self.close()