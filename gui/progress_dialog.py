from PyQt6.QtWidgets import QDialog, QProgressBar, QVBoxLayout, QLabel, QPushButton
from PyQt6.QtCore import Qt, QTimer
import time

class ProgressDialog(QDialog):
    def __init__(self, parent=None, operation_type="Training"):
        super().__init__(parent)
        self.operation_type = operation_type
        self.start_time = None
        self.setup_ui()
        
    def setup_ui(self):
        """Inicializace UI komponent"""
        self.setWindowTitle(f"Model {self.operation_type}")
        self.setFixedSize(400, 150)
        self.setModal(True)
        
        layout = QVBoxLayout()
        
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
        self.cancel_button.clicked.connect(self.reject)
        layout.addWidget(self.cancel_button)
        
        self.setLayout(layout)
        
    def start(self):
        """Spustí dialog a inicializuje časovač"""
        self.start_time = time.time()
        self.show()
        
    def update_progress(self, progress, current_loss=None):
        """Aktualizuje progress bar a časový odhad
        
        Args:
            progress (float): Hodnota mezi 0 a 1 reprezentující celkový průběh
            current_loss (float, optional): Aktuální hodnota loss funkce
        """
        # Aktualizace progress baru
        progress_percent = int(progress * 100)
        self.progress_bar.setValue(progress_percent)
        
        # Výpočet zbývajícího času
        if self.start_time and progress > 0:
            elapsed_time = time.time() - self.start_time
            estimated_total_time = elapsed_time / progress
            remaining_time = estimated_total_time - elapsed_time
            
            # Formátování času
            minutes = int(remaining_time // 60)
            seconds = int(remaining_time % 60)
            time_str = f"{minutes}m {seconds}s"
            
            # Aktualizace labelů
            status_text = f"{self.operation_type} in progress... "
            if current_loss is not None:
                status_text += f"(Loss: {current_loss:.4f})"
            
            self.status_label.setText(status_text)
            self.time_label.setText(f"Estimated time remaining: {time_str}")
        
    def finish(self):
        """Dokončí operaci a zavře dialog"""
        self.accept()