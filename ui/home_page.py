import os
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import QObject, pyqtSlot, QUrl

# ✅ Mode sans WebEngine possible
SAFE_MODE = os.getenv("SAFE_MODE", "0") == "1"
if not SAFE_MODE:
    try:
        from PyQt6.QtWebEngineWidgets import QWebEngineView
        from PyQt6.QtWebChannel import QWebChannel
    except ImportError as e:
        print(f"⚠️ WebEngine indisponible : {e}")
        QWebEngineView = None
        QWebChannel = None
else:
    QWebEngineView = None
    QWebChannel = None


class HomeBridge(QObject):
    def __init__(self, navigate_to_module):
        super().__init__()
        self.navigate_to_module = navigate_to_module

    @pyqtSlot(str)
    def pycmd(self, module_key):
        self.navigate_to_module(module_key)


class HomePage(QWidget):
    def __init__(self, navigate_to_module):
        super().__init__()
        layout = QVBoxLayout()
        self.setLayout(layout)

        if QWebEngineView is None or QWebChannel is None:
            # 🛡️ Fallback texte si SAFE_MODE ou pas de WebEngine
            label = QLabel("SAFE MODE: Home view désactivée\n(ou WebEngine non disponible)")
            label.setStyleSheet("color: red;")
            layout.addWidget(label)
            return

        # ✅ WebEngine view
        self.view = QWebEngineView()
        layout.addWidget(self.view)

        # ✅ Bridge & Channel
        self.channel = QWebChannel()
        self.bridge = HomeBridge(navigate_to_module)
        self.channel.registerObject("bridge", self.bridge)
        self.view.page().setWebChannel(self.channel)

        # ✅ Load the local HTML file
        html_path = os.path.abspath("ui/assets/home.html")
        if os.path.exists(html_path):
            self.view.setUrl(QUrl.fromLocalFile(html_path))
        else:
            print("❌ Erreur : home.html introuvable")
