# ui/main_window.py

import sys
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QLabel, QVBoxLayout, QPushButton, QStackedWidget,
    QPlainTextEdit, QMessageBox
)
from PyQt6.QtCore import Qt, pyqtSignal, QObject, QTimer

from ui.home_page import HomePage
from ui.asset_initializer import AssetInitializationWindow
from ui.modules.price_viewer import PriceViewer
from ui.components.selection_widget import SelectionWidget  # ✅ import
from ui.modules.gex_viewer_module import GEXViewer




class EmittingStream(QObject):
    text_written = pyqtSignal(str)

    def write(self, text):
        if text.strip():
            self.text_written.emit(str(text))

    def flush(self):
        pass


class MainWindow(QMainWindow):
    def __init__(self, offline_mode=False):
        super().__init__()

        # Redirection stdout vers log
        stream = EmittingStream()
        stream.text_written.connect(self.append_log)
        sys.stdout = stream

        self.setWindowTitle("Volatility Trader Pro")
        self.setMinimumSize(1200, 800)
        self.offline_mode = offline_mode
        self.page_dict = {}

        central_widget = QWidget()
        self.layout = QVBoxLayout(central_widget)

        # Stack des pages principales
        self.stack = QStackedWidget()
        self.layout.addWidget(self.stack)

        # 🏠 Accueil
        self.home_page = HomePage(self.change_page)
        self.stack.addWidget(self.home_page)
        self.page_dict["🏠 Accueil"] = self.home_page

        # 📊 Page analyse (ticker / période / graph)
        self.selection_widget = SelectionWidget()
        self.selection_widget.selection_changed.connect(self.handle_selection)  # ❗ Connecte le signal
        self.stack.addWidget(self.selection_widget)
        self.page_dict["Analyse de la volatilité"] = self.selection_widget

        # 📈 Viewer graphique (à part)
        self.price_viewer = PriceViewer(main_window=self)
        self.stack.addWidget(self.price_viewer)
        self.page_dict["📈 Visualiser les prix"] = self.price_viewer

        # Modules encore vides (à remplir plus tard)
        #self.gex_viewer = GEXViewer()
        self.orderflow_viewer = None
        self.macro_viewer = None

        # ⚙️ Page d’initialisation
        self.init_page = self.create_init_page()
        self.stack.addWidget(self.init_page)
        self.page_dict["⚙ Initialisation"] = self.init_page

        # 📜 Log
        self.log_label = QLabel("📝 Journal d’activité :")
        self.log_textbox = QPlainTextEdit()
        self.log_textbox.setReadOnly(True)
        self.log_textbox.setStyleSheet("background-color: #1e1e1e; color: #00FF00; font-family: Consolas;")
        self.layout.addWidget(self.log_label)
        self.layout.addWidget(self.log_textbox)

        self.setCentralWidget(central_widget)

        QTimer.singleShot(0, lambda: self.stack.currentChanged.connect(self.handle_page_change))
        print("✅ Application lancée.")

    def handle_page_change(self, index):
        try:
            current_widget = self.stack.widget(index)
            if hasattr(self, "log_label") and hasattr(self, "log_textbox"):
                if current_widget in [self.home_page, self.init_page]:
                    self.log_label.show()
                    self.log_textbox.show()
                else:
                    self.log_label.hide()
                    self.log_textbox.hide()
        except Exception as e:
            print(f"⚠️ Erreur dans handle_page_change : {e}")

    def change_page(self, module_key):
        mapping = {
            "home": self.home_page,
            "analyse": self.price_viewer,
            "orderflow": self.orderflow_viewer,
            "macro": self.macro_viewer,
            "init": self.init_page,
        }

        if module_key == "gex":
            self.gex_viewer = GEXViewer()
            self.stack.addWidget(self.gex_viewer)
            self.stack.setCurrentWidget(self.gex_viewer)
            return

        if module_key in mapping and mapping[module_key] is not None:
            self.stack.setCurrentWidget(mapping[module_key])
        else:
            QMessageBox.warning(self, "Module indisponible", f"Le module '{module_key}' n'est pas encore initialisé.")

    def display_statistical_analysis(self):
        if hasattr(self, "dfvol"):
            viewer = RegimeVolatiliteViewer(self.dfvol, parent=self)
            self.stack.setCurrentWidget(viewer)
            self.stack.addWidget(viewer)
            self.append_log("📊 Affichage du module Régime Volatilité")
        else:
            self.append_log("❌ dfvol introuvable dans MainWindow.")

    def handle_selection(self, ticker, period, graph_type):
        print(f"➡️ Sélection : {ticker} / {period} / {graph_type}")
        # Ajoute ici le dispatch vers les graphes selon `graph_type`
        # Exemple temporaire :
        if graph_type == "Historique des prix":
            self.price_viewer.plot_price_graph(ticker, period)
            self.stack.setCurrentWidget(self.price_viewer)


    def append_log(self, message: str):
        self.log_textbox.appendPlainText(message)

    def launch_initialisation(self):
        self.asset_window = AssetInitializationWindow()
        self.asset_window.show()
        print("🧩 Initialisation de la base de données lancée...")

    def create_init_page(self):
        page = QWidget()
        page.setStyleSheet("background-color: #1e1e1e; color: white;")
        layout = QVBoxLayout(page)

        title = QLabel("📈 Volatility Trader Pro")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 28px; font-weight: bold; margin-top: 40px; color: white;")
        layout.addWidget(title)

        if self.offline_mode:
            warning = QLabel("⚠ Mode hors ligne activé : certaines fonctions sont désactivées.")
            warning.setAlignment(Qt.AlignmentFlag.AlignCenter)
            warning.setStyleSheet("color: orange; font-size: 16px; margin: 10px;")
            layout.addWidget(warning)

        button_style = """
            QPushButton {
                background-color: #2e2e2e;
                color: white;
                border: 1px solid #00ffcc;
                border-radius: 8px;
                padding: 10px 20px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #3e3e3e;
            }
        """

        init_button = QPushButton("🛠 Initialiser ou mettre à jour les bases de données")
        init_button.setStyleSheet(button_style)
        init_button.clicked.connect(self.launch_initialisation)
        layout.addWidget(init_button, alignment=Qt.AlignmentFlag.AlignCenter)

        modules_button = QPushButton("🏠 Retour à l’accueil")
        modules_button.setStyleSheet(button_style)
        modules_button.clicked.connect(lambda: self.stack.setCurrentWidget(self.home_page))
        layout.addWidget(modules_button, alignment=Qt.AlignmentFlag.AlignCenter)


        return page
