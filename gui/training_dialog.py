from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QProgressBar, 
                           QLabel, QPushButton)

class TrainingDialog(QDialog):
    def __init__(self, model, parent=None):
        super().__init__(parent)
        self.model = model
        self.setWindowTitle("Training Progress")
        self.setModal(True)
        
        # Create layout
        layout = QVBoxLayout(self)
        
        # Progress display
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.time_label = QLabel("Estimated time remaining: calculating...")
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.time_label)
        
        # Cancel button
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        layout.addWidget(self.cancel_button)
        
        # Create and start trainer thread
        self.trainer = ModelTrainer(self.model)
        self.trainer.progress_updated.connect(self._update_progress)
        self.trainer.time_updated.connect(self._update_time)
        self.trainer.finished.connect(self.accept)
        self.trainer.start()
    
    def _update_progress(self, progress: int):
        """Update progress bar"""
        self.progress_bar.setValue(progress)
    
    def _update_time(self, minutes: float):
        """Update estimated time"""
        self.time_label.setText(f"Estimated time remaining: {minutes:.1f} minutes")
    
    def reject(self):
        """Handle dialog cancellation"""
        self.trainer.stop()
        super().reject()