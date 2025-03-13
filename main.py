import sys
from PyQt6.QtWidgets import QApplication

from utils.style_manager import StyleManager
from fruitvegnet.widgets.main_window import MainWindow

def main():
    app = QApplication(sys.argv)
    style_manager = StyleManager(app)
    window = MainWindow(style_manager)
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()