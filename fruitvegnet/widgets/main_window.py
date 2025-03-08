from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QTabWidget,
                             QSizePolicy, QSpacerItem, QStatusBar, QFrame,
                             QHBoxLayout, QStackedWidget, QPushButton, QLabel)
from PyQt6.QtGui import QIcon, QAction
from PyQt6.QtCore import Qt
import os

from fruitvegnet.model_settings import TabWidget
from fruitvegnet.image_classification import ImageClassificationWidget
from models.simple_cnn_model import SimpleCnnModel
from models.resnet_model import ResNetModel
from models.efficientnet_model import EfficientNetModel
from models.vgg16_model import VGG16Model


class MainWindow(QMainWindow):
    
    def __init__(self):
        super().__init__()
        self._create_ui()
        
    def _create_ui(self):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        icon_path = os.path.join(project_root, "assets", "icon.png")  

        self.setWindowIcon(QIcon(icon_path))
        self.setWindowTitle("FruitVegNet")
        self.setGeometry(50, 50, 1300, 700)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Create Sidebar
        self.sidebar = self._create_sidebar()
        
        # Vytvoření kontejneru pro obsah a status bar
        content_container = QWidget()
        content_layout = QVBoxLayout(content_container)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)
        
        # Vytvoření stacked widgetu pro obsah
        self.content_stack = QStackedWidget()
        
        # Vytvoření tab widgetu (první stránka)
        self.tab_widget = QTabWidget()
        self.tab_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)  
        self.simple_cnn_tab = TabWidget(model_class=SimpleCnnModel)
        self.resnet_tab = TabWidget(model_class=ResNetModel)
        self.efficientnet_tab = TabWidget(model_class=EfficientNetModel)
        self.vgg16_tab = TabWidget(model_class=VGG16Model)
        
        self.tab_widget.addTab(self.simple_cnn_tab, "Simple CNN")
        self.tab_widget.addTab(self.resnet_tab, "ResNet")
        self.tab_widget.addTab(self.efficientnet_tab, "EfficientNet-B0")
        self.tab_widget.addTab(self.vgg16_tab, "VGG16")
        
        # Vytvoření widgetu pro klasifikaci obrázků (druhá stránka)
        self.image_classification_widget = ImageClassificationWidget()
        self.image_classification_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        # Přidání widgetů do stacku
        self.content_stack.addWidget(self.tab_widget)
        self.content_stack.addWidget(self.image_classification_widget)
        
        # Vytvoření status baru pro pravou část
        self.status_bar = QStatusBar()
        self.status_bar.setObjectName("status-bar")
        
        # Přidání obsahu a status baru do kontejneru obsahu
        content_layout.addWidget(self.content_stack, 1)
        content_layout.addWidget(self.status_bar, 0)
        
        # Přidání sidebaru a kontejneru obsahu do hlavního layoutu
        main_layout.addWidget(self.sidebar, 1)  # Sidebar zabere 1 díl
        main_layout.addWidget(content_container, 5)  # Obsah zabere 5 dílů
        
        # Propojení signálů
        self.image_classification_widget.classify_clicked.connect(self._classify_all) 
        self.image_classification_widget.image_loaded.connect(self.update_status_bar) 
        self._connect_tab_status_signals()
        
        # Nastavení menu
        #self._setup_menu()

    def _create_sidebar(self):
        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        #sidebar.setMinimumWidth(200)
        #sidebar.setMaximumWidth(250)
        
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(0)
        
        # Přidání loga/hlavičky do sidebaru
        logo_widget = QWidget()
        logo_widget.setObjectName("logo-widget")
        logo_widget.setMinimumHeight(80)
        logo_layout = QVBoxLayout(logo_widget)
        
        # Logo a název aplikace
        title_label = QLabel("FruitVegNet")
        title_label.setObjectName("app-title")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        logo_layout.addWidget(title_label)
        
        sidebar_layout.addWidget(logo_widget)

        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Plain)
        separator.setStyleSheet("color: #e3e3e3; background-color: #f2f2f2; height: 1px; margin-left: 10px; margin-right: 10px;")

        sidebar_layout.addWidget(separator)

        # Přidání mezery pod čárou
        spacer = QSpacerItem(20, 30, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        sidebar_layout.addItem(spacer)
        
        # Přidání navigačních tlačítek
        models_btn = self._create_sidebar_button("Models", 0)
        classify_btn = self._create_sidebar_button("Classification", 1)
        
        sidebar_layout.addWidget(models_btn)
        sidebar_layout.addWidget(classify_btn)
        
        # Přidání roztažení, aby bylo tlačítka nahoře
        sidebar_layout.addStretch()
        
        return sidebar
    
    def _create_sidebar_button(self, text, page_index):
        button = QPushButton(text)
        button.setObjectName("sidebar-button")
        button.setMinimumHeight(50)
        button.setCheckable(True)
        
        # Nastavení prvního tlačítka jako označeného
        if page_index == 0:
            button.setChecked(True)
        
        # Připojení tlačítka k přepínání stránek
        button.clicked.connect(lambda: self._switch_page(page_index, button))
        
        return button
    
    def _switch_page(self, index, clicked_button):
        # Přepnutí na vybranou stránku
        self.content_stack.setCurrentIndex(index)
        
        # Aktualizace stavu tlačítek
        for i in range(self.sidebar.layout().count() - 1):  # Přeskočení stretch itemu
            item = self.sidebar.layout().itemAt(i)
            if item and isinstance(item.widget(), QPushButton):
                item.widget().setChecked(item.widget() == clicked_button)
        
    def _connect_tab_status_signals(self):
        """Propojení signálů z každé záložky do hlavního status baru"""
        tabs = [self.simple_cnn_tab, self.resnet_tab, self.efficientnet_tab, self.vgg16_tab]
        for tab in tabs:
            tab.status_message.connect(self.update_status_bar)

    def update_status_bar(self, message, timeout=8000):
        """Aktualizace hlavního status baru zprávou"""
        self.status_bar.showMessage(message, timeout)
        
    def _setup_menu(self):
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("File")
        
        # Export a import akcí
        self.save_action = QAction("Save Model", self)
        self.load_action = QAction("Load Model", self)
        file_menu.addAction(self.save_action)
        file_menu.addAction(self.load_action)
        self.save_action.triggered.connect(self._save_current_model)
        self.load_action.triggered.connect(self._load_current_model)
        file_menu.addSeparator()
        
        # Akce ukončení
        self.quit_action = QAction("Quit", self)
        file_menu.addAction(self.quit_action)  
        self.quit_action.triggered.connect(self.close)
            
    def _classify_all(self):
        """Klasifikace aktuálního obrázku pomocí všech načtených modelů"""
        if not self.image_classification_widget.current_image_path:
            self.update_status_bar("No image loaded for classification")
            return
            
        model_map = {
            self.simple_cnn_tab: {'type': 'simple_cnn', 'name': 'Simple CNN'},
            self.resnet_tab: {'type': 'resnet', 'name': 'ResNet'},
            self.efficientnet_tab: {'type': 'efficientnet', 'name': 'EfficientNet'},
            self.vgg16_tab: {'type': 'vgg16', 'name': 'VGG16'}
        }
        
        results = []
        for tab, model_info in model_map.items():
            if tab.model_loaded:
                try:
                    # Získání predikce
                    result = tab.model.predict_image(self.image_classification_widget.current_image_path)
                    predicted_class = result['class']
                    probabilities = result['probabilities']
                    
                    # Aktualizace výsledku
                    self.image_classification_widget.update_result(model_info['type'], predicted_class)
                    
                    # Aktualizace grafu pro tento model
                    self.image_classification_widget.update_plot(model_info['type'], tab.model.classes, probabilities)
                    
                    results.append(f"{model_info['name']}: {predicted_class}")

                    self.update_status_bar("Classification complete")
                except Exception as e:
                    self.image_classification_widget.update_result(model_info['type'], "Error")
                    print(f"Error in {model_info['name']} classification: {str(e)}")
                    self.update_status_bar("No models available for classification")
            else:
                self.image_classification_widget.update_result(model_info['type'], "No model")

    def _save_current_model(self):
        """Uložení modelu z aktuálně aktivní záložky"""
        current_tab = self.tab_widget.currentWidget()
        if current_tab:
            current_tab.save_model()
            
    def _load_current_model(self):
        """Načtení modelu do aktuálně aktivní záložky"""
        current_tab = self.tab_widget.currentWidget()
        if current_tab:
            current_tab.load_model()