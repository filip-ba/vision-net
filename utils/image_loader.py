from PyQt6.QtGui import QPixmap

def load_image(file_path):
    """Load an image from the specified file path and return a QPixmap."""
    pixmap = QPixmap(file_path)
    return pixmap