# ui/components/selection_widget.py

from PyQt6.QtWidgets import QWidget, QHBoxLayout, QLabel, QComboBox, QPushButton
from PyQt6.QtCore import pyqtSignal
import sqlite3


class SelectionWidget(QWidget):
    selection_changed = pyqtSignal(str, str, str)  # ticker, période, type de graph

    def __init__(self, db_path="db/market_data.db"):
        super().__init__()
        self.db_path = db_path

        layout = QHBoxLayout()

        layout.addWidget(QLabel("Indice :"))
        self.ticker_select = QComboBox()
        layout.addWidget(self.ticker_select)

        layout.addWidget(QLabel("Période :"))
        self.period_select = QComboBox()
        self.period_select.addItems(["1 an", "5 ans", "Complet"])
        layout.addWidget(self.period_select)

        layout.addWidget(QLabel("Graphique :"))
        self.graph_select = QComboBox()
        self.graph_select.addItems([
            "Historique des prix",
            "Moyennes de volatilité",
            "Prévision HAR-RV",
            "Distribution",
            "Heatmap journalière",
            "Volatilité historique",
            "Analyse des régimes de volatilité"
        ])
        layout.addWidget(self.graph_select)

        self.load_button = QPushButton("Afficher le graphique")
        self.load_button.clicked.connect(self.emit_selection)
        layout.addWidget(self.load_button)

        self.setLayout(layout)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(10)

        self.refresh_ticker_list()

    def refresh_ticker_list(self):
        tickers = self.get_available_tickers_from_db()
        current = self.ticker_select.currentText()

        self.ticker_select.blockSignals(True)
        self.ticker_select.clear()
        self.ticker_select.addItems(tickers)
        if current in tickers:
            self.ticker_select.setCurrentText(current)
        self.ticker_select.blockSignals(False)

    def emit_selection(self):
        ticker = self.ticker_select.currentText()
        period = self.period_select.currentText()
        graph = self.graph_select.currentText()
        self.selection_changed.emit(ticker, period, graph)

    def get_available_tickers_from_db(self):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            conn.close()
            return sorted([t.replace("_data", "").upper() for t in tables if t.endswith("_data")])
        except Exception as e:
            print(f"❌ Erreur DB : {e}")
            return ["SPX", "NDX", "VIX"]
