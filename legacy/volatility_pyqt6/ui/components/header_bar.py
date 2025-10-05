from PyQt6.QtWidgets import QWidget, QHBoxLayout, QPushButton
from PyQt6.QtCore import pyqtSignal


class HeaderBar(QWidget):
    return_home = pyqtSignal()

    def __init__(self):
        super().__init__()

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 10)  # espacement bas
        layout.setSpacing(10)

        home_button = QPushButton("🏠 Retour à l’accueil")
        home_button.setStyleSheet("""
            QPushButton {
                background-color: #2e2e2e;
                color: white;
                border: 1px solid #00ffcc;
                border-radius: 8px;
                padding: 6px 12px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #3e3e3e;
            }
        """)
        home_button.clicked.connect(self.return_home.emit)

        layout.addWidget(home_button)
        layout.addStretch()
        self.setLayout(layout)
