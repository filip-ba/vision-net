from gui.main_window import MainWindow
from PyQt6.QtWidgets import QApplication
from models.dicetoss_model import DiceNet, train_model, load_datasets
import sys

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()

    # Load the datasets and initialize the model
    trainloader, valloader, testloader = load_datasets()
    net = DiceNet()

    # Train the model
    train_model(net, trainloader, valloader)

    sys.exit(app.exec())
