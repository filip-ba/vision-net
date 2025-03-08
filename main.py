import sys
from PyQt6.QtWidgets import QApplication

from fruitvegnet.widgets.main_window import MainWindow

def load_stylesheet(path):
    with open(path, "r") as f:
        return f.read()

def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(load_stylesheet("utils/styles.css"))
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()