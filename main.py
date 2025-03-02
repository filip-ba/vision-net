import sys
from PyQt6.QtWidgets import QApplication
from widgets.main_window import MainWindow

from utils.set_palette import palette

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")    
    app.setPalette(palette)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()