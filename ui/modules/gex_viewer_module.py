# ui/modules/gex_viewer_module.py

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton, QCheckBox
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import sqlite3
import pandas as pd
from core.ib_gex_utils import fetch_expirations, fetch_gex_data
from datetime import datetime

DB_PATH = "db/market_data.db"

class GEXViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GEX Viewer")

        self.ticker_select = QComboBox()
        self.expiry_select = QComboBox()
        self.fetch_expirations_button = QPushButton("📆 Charger échéances")
        self.fetch_gex_button = QPushButton("📥 Récupérer GEX")
        self.refresh_button = QPushButton("🔄 Rafraîchir le graphe")
        self.mode_checkbox = QCheckBox("Afficher GEX absolu")
        self.back_button = QPushButton("🏠 Retour à l'accueil")
        self.canvas = FigureCanvas(Figure(figsize=(10, 6)))
        self.ax = self.canvas.figure.subplots()
        self.gex_total_label = QLabel("GEX Net : N/A")

        self.back_button.setStyleSheet("""
            QPushButton {
                background-color: #1e1e1e;
                color: white;
                border: 1px solid #00ffcc;
                border-radius: 8px;
                padding: 6px 14px;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #2e2e2e;
            }
        """)

        top_layout = QHBoxLayout()
        top_layout.addWidget(QLabel("Ticker:"))
        top_layout.addWidget(self.ticker_select)
        top_layout.addWidget(QLabel("Échéance:"))
        top_layout.addWidget(self.expiry_select)
        top_layout.addWidget(self.fetch_expirations_button)
        top_layout.addWidget(self.fetch_gex_button)
        top_layout.addWidget(self.refresh_button)
        top_layout.addWidget(self.mode_checkbox)
        top_layout.addWidget(self.back_button)

        layout = QVBoxLayout()
        layout.addLayout(top_layout)
        layout.addWidget(self.gex_total_label)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        self.check_or_create_table()
        self.populate_tickers()

        self.ticker_select.currentIndexChanged.connect(self.clear_expirations)
        self.expiry_select.currentIndexChanged.connect(self.update_graph)
        self.fetch_expirations_button.clicked.connect(self.fetch_expirations_from_ib)
        self.fetch_gex_button.clicked.connect(self.fetch_and_store_gex)
        self.refresh_button.clicked.connect(self.update_graph)
        self.mode_checkbox.stateChanged.connect(self.update_graph)
        self.back_button.clicked.connect(self.return_home)

    def check_or_create_table(self):
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS gex_observations (
                date TEXT,
                ticker TEXT,
                expiry TEXT,
                strike REAL,
                type TEXT,
                openInterest INTEGER,
                gamma REAL,
                contractMultiplier INTEGER,
                undPrice REAL
            )
        """)
        conn.commit()
        conn.close()

    def populate_tickers(self):
        conn = sqlite3.connect(DB_PATH)
        try:
            tickers = pd.read_sql("SELECT DISTINCT ticker FROM gex_observations", conn)
            self.ticker_select.clear()
            self.ticker_select.addItems(tickers['ticker'].tolist())
        except Exception as e:
            print(f"⚠️ Erreur chargement tickers : {e}")
        finally:
            conn.close()

    def clear_expirations(self):
        self.expiry_select.clear()

    def fetch_expirations_from_ib(self):
        symbol = self.ticker_select.currentText()
        if not symbol:
            return

        try:
            expiries = fetch_expirations(symbol)
            self.expiry_select.clear()
            self.expiry_select.addItems(expiries)
            print(f"✅ Échéances chargées pour {symbol} : {expiries}")
        except Exception as e:
            print(f"❌ Erreur récupération échéances IB : {e}")

    def fetch_and_store_gex(self):
        symbol = self.ticker_select.currentText()
        expiry = self.expiry_select.currentText()
        if not symbol or not expiry:
            print("❗ Sélectionne un ticker et une échéance")
            return

        try:
            df = fetch_gex_data(symbol, expiry)

            # 🧹 Nettoyage AVANT insertion
            # 🧹 Nettoyage robuste des valeurs NaN et non numériques
            cols = ['gamma', 'openInterest', 'contractMultiplier', 'strike']
            df = df[cols].apply(pd.to_numeric, errors='coerce')  # force les colonnes en numériques
            df = df.dropna(subset=cols)  # enlève les NaN
            df = df.astype({'openInterest': int, 'contractMultiplier': int, 'strike': int})

            df['date'] = datetime.now().strftime("%Y-%m-%d")
            df['ticker'] = symbol
            df['expiry'] = expiry

            conn = sqlite3.connect(DB_PATH)
            df.to_sql("gex_observations", conn, if_exists="append", index=False)
            conn.close()

            print(f"✅ Données GEX insérées pour {symbol} - {expiry}")
            self.update_graph()

        except Exception as e:
            print(f"❌ Erreur récupération GEX : {e}")

    def update_graph(self):
        ticker = self.ticker_select.currentText()
        expiry = self.expiry_select.currentText()

        if not ticker or not expiry:
            return

        conn = sqlite3.connect(DB_PATH)
        try:
            df = pd.read_sql("""
                SELECT strike, type, gamma, openInterest, contractMultiplier, undPrice
                FROM gex_observations WHERE ticker = ? AND expiry = ?
            """, conn, params=(ticker, expiry))
        except Exception as e:
            print(f"⚠️ Erreur lecture GEX : {e}")
            df = pd.DataFrame()
        finally:
            conn.close()

        self.ax.clear()

        if df.empty:
            self.ax.set_title("Aucune donnée de GEX disponible.")
            self.gex_total_label.setText("GEX Net : N/A")
        else:
            df['gex'] = df['gamma'] * df['openInterest'] * df['contractMultiplier'] * 0.01
            gex_calls = df[df['type'] == 'C'].groupby('strike')['gex'].sum()
            gex_puts = df[df['type'] == 'P'].groupby('strike')['gex'].sum() * -1

            strikes = sorted(set(gex_calls.index).union(set(gex_puts.index)))
            gex_calls = gex_calls.reindex(strikes, fill_value=0)
            gex_puts = gex_puts.reindex(strikes, fill_value=0)
            gex_net_by_strike = gex_calls + gex_puts

            if self.mode_checkbox.isChecked():
                self.ax.bar(gex_net_by_strike.index, gex_net_by_strike.abs(), color='purple', label='|GEX Net|')
                total = round(gex_net_by_strike.abs().sum(), 2)
                self.gex_total_label.setText(f"GEX Absolu : {total:,.2f}")
            else:
                self.ax.bar(strikes, gex_calls, color='green', label='Calls')
                self.ax.bar(strikes, gex_puts, color='orange', label='Puts')
                total = round(gex_net_by_strike.sum(), 2)
                self.gex_total_label.setText(f"GEX Net : {total:,.2f}")

            if 'undPrice' in df.columns and not df['undPrice'].isna().all():
                spot = df['undPrice'].dropna().iloc[0]
                self.ax.axvline(x=spot, color='red', linestyle='--', label=f"Spot: {spot}")

            self.ax.set_title(f"Gamma Exposure pour {ticker} - {expiry}")
            self.ax.set_xlabel("Strike")
            self.ax.set_ylabel("GEX")
            self.ax.legend()

        self.canvas.draw()

    def return_home(self):
        from ui.main_window import MainWindow
        mw = self.parent()
        while mw and not isinstance(mw, MainWindow):
            mw = mw.parent()
        if mw:
            mw.stack.setCurrentWidget(mw.home_page)
