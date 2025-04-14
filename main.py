import ctypes
myappid = 'mycompany.fruitvegnet.version'
ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

import sys
from PyQt6.QtWidgets import QApplication

from src.ui.styles.style_manager import StyleManager
from src.core.main_window import MainWindow

def main():
    app = QApplication(sys.argv)
    style_manager = StyleManager(app)
    window = MainWindow(style_manager)
    window.show() 
    sys.exit(app.exec())

if __name__ == "__main__":
    main()