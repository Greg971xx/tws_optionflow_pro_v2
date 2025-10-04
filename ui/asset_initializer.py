from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QPushButton, QCheckBox,
    QScrollArea, QWidget, QMessageBox, QLineEdit, QComboBox
)
from core.tickers_catalog import TICKERS_CATALOG
from core.database_initializer import initialize_database


def clean_symbol_input(symbol_raw: str) -> str:
    """
    Nettoie le symbole d'un ticker : supprime les suffixes (.PA, .DE, etc.)
    et retourne la partie utile (ex: AIR.PA -> AIR).
    """
    symbol = symbol_raw.strip().upper()
    known_suffixes = [".PA", ".DE", ".MI", ".BR", ".AS", ".ST", ".L", ".HE", ".SW", ".VX", ".OL"]
    for suffix in known_suffixes:
        if symbol.endswith(suffix):
            symbol = symbol.replace(suffix, "")
            break
    return symbol


class AssetInitializationWindow(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sélection des actifs à importer")
        self.setMinimumSize(600, 700)

        self.checkboxes = {}
        self.layout = QVBoxLayout(self)

        warning = QLabel("\u26a0\ufe0f Veuillez ne sélectionner que les actifs pour lesquels vous disposez d’un abonnement actif chez IBKR ou LYNX.")
        warning.setStyleSheet("color: orange; font-weight: bold;")
        self.layout.addWidget(warning)

        self.scroll_area = QScrollArea()
        self.scroll_widget = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_widget)

        for symbol in TICKERS_CATALOG.keys():
            checkbox = QCheckBox(symbol)
            checkbox.setChecked(False)
            self.checkboxes[symbol] = checkbox
            self.scroll_layout.addWidget(checkbox)

        self.scroll_area.setWidget(self.scroll_widget)
        self.scroll_area.setWidgetResizable(True)
        self.layout.addWidget(self.scroll_area)

        self.layout.addWidget(QLabel("\u2795 Ajouter un ticker manuellement :"))

        self.symbol_input = QLineEdit()
        self.symbol_input.setPlaceholderText("Symbol (ex: AIR, MC, SAP)")
        symbol_hint = QLabel("\u26a0\ufe0f Ne pas inclure de suffixe .PA, .DE, etc. — utilisez uniquement le code de base (ex: AIR, MC, SAP).")
        symbol_hint.setStyleSheet("color: gray; font-size: 10px;")

        self.exchange_input = QLineEdit()
        self.exchange_input.setPlaceholderText("Exchange (ex: SMART, GLOBEX)")

        self.currency_input = QLineEdit()
        self.currency_input.setPlaceholderText("Currency (ex: USD, EUR)")

        self.secType_combo = QComboBox()
        self.secType_combo.addItem("🏦 Action (STK)", "STK")
        self.secType_combo.addItem("📈 Indice (IND)", "IND")
        self.secType_combo.addItem("⚙️ Future (FUT)", "FUT")

        self.layout.addWidget(self.symbol_input)
        self.layout.addWidget(symbol_hint)
        self.layout.addWidget(self.exchange_input)
        self.layout.addWidget(self.currency_input)
        self.layout.addWidget(QLabel("Type de produit :"))
        self.layout.addWidget(self.secType_combo)

        self.add_button = QPushButton("Ajouter ce ticker")
        self.add_button.clicked.connect(self.add_manual_ticker)
        self.layout.addWidget(self.add_button)

        self.import_button = QPushButton("Créer la base ou mise à jour avec les actifs sélectionnés")
        self.import_button.clicked.connect(self.create_database)
        self.layout.addWidget(self.import_button)

    def add_manual_ticker(self):
        raw_symbol = self.symbol_input.text()
        symbol = clean_symbol_input(raw_symbol)
        exchange = self.exchange_input.text().strip().upper()
        currency = self.currency_input.text().strip().upper()
        secType = self.secType_combo.currentData()

        if not (symbol and exchange and currency and secType):
            QMessageBox.warning(self, "Champs manquants", "Veuillez remplir tous les champs.")
            return

        if symbol in self.checkboxes:
            QMessageBox.warning(self, "Déjà présent", f"{symbol} est déjà dans la liste.")
            return

        TICKERS_CATALOG[symbol] = {
            "symbol": symbol,
            "exchange": exchange,
            "currency": currency,
            "secType": secType
        }

        checkbox = QCheckBox(symbol)
        checkbox.setChecked(True)
        self.checkboxes[symbol] = checkbox
        self.scroll_layout.addWidget(checkbox)

        self.symbol_input.clear()
        self.exchange_input.clear()
        self.currency_input.clear()

        QMessageBox.information(self, "Ajouté", f"{symbol} a été ajouté avec succès.")

    def create_database(self):
        selected = [sym for sym, cb in self.checkboxes.items() if cb.isChecked()]

        if not selected:
            QMessageBox.warning(self, "Aucun actif sélectionné", "Veuillez sélectionner au moins un actif.")
            return

        self.close()
        initialize_database(selected)
