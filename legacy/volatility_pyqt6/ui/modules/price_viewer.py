import os
SAFE_MODE = os.getenv("SAFE_MODE", "0") == "1"
try:
    if not SAFE_MODE:
        from PyQt6.QtWebEngineWidgets import QWebEngineView  # ok seulement si non-safe
    else:
        QWebEngineView = None
except Exception as e:
    print(f"⚠️ WebEngine indisponible: {e}")
    QWebEngineView = None


from PyQt6.QtWidgets import QWidget, QVBoxLayout, QStackedWidget, QSizePolicy
from PyQt6.QtCore import QUrl
import sqlite3
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import tempfile

from ui.components.selection_widget import SelectionWidget
from ui.components.header_bar import HeaderBar
from ui.modules.volatility_viewer import VolatilityViewer
from ui.modules.har_rv_viewer import HARVolatilityViewer
from ui.modules.volatility_distribution import VolatilityDistributionViewer
from ui.modules.volatilite_jour_semaine_viewer import VolatiliteJourSemaineViewer
from ui.modules.volatilite_historique_viewer import VolatiliteHistoriqueViewer
from ui.modules.volatility_regime_viewer import VolatilityRegimeViewer



class PriceViewer(QWidget):
    def __init__(self, main_window=None, parent=None):
        super().__init__(parent)
        self.main_window = main_window

        main_layout = QVBoxLayout()

        # Header avec bouton retour
        self.header = HeaderBar()
        self.header.return_home.connect(self.return_to_home)
        main_layout.addWidget(self.header)

        # Sélecteur ticker + période + graphique
        self.selector = SelectionWidget()
        self.selector.setMaximumHeight(35)
        self.selector.selection_changed.connect(self.handle_selection)
        main_layout.addWidget(self.selector)

        # Stack de modules
        self.module_stack = QStackedWidget()
        self.module_stack.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        main_layout.addWidget(self.module_stack)

        # Module 1 : graphique des prix
        self.web_view = QWebEngineView()
        self.web_view.setMinimumHeight(500)
        self.module_stack.addWidget(self.web_view)

        # Module 2 : Volatilité Réalisée
        self.volatility_viewer = VolatilityViewer()
        self.module_stack.addWidget(self.volatility_viewer)

        # Module 3 : Prévision HAR-RV
        self.har_rv_viewer = HARVolatilityViewer()
        self.module_stack.addWidget(self.har_rv_viewer)

        # Module 4 : distribution volatilité
        self.volatility_distribution = VolatilityDistributionViewer()
        self.module_stack.addWidget(self.volatility_distribution)

        # Module 5 : distribution par jour semaine
        self.volatilite_jour_semaine_viewer = VolatiliteJourSemaineViewer()
        self.module_stack.addWidget(self.volatilite_jour_semaine_viewer)

        # Module 6 : volatilité historique
        self.volatilite_historique_viewer = VolatiliteHistoriqueViewer()
        self.module_stack.addWidget(self.volatilite_historique_viewer)

        # Module 7 : régime de volatilité
        self.volatility_regime_viewer = VolatilityRegimeViewer()
        self.module_stack.addWidget(self.volatility_regime_viewer)


        self.setLayout(main_layout)

        # Affichage par défaut
        self.plot_price_graph("SPX", "1 an")

    def plot_prices(self, ticker, period):
        try:
            conn = sqlite3.connect("db/market_data.db")
            df = pd.read_sql(f"SELECT * FROM {ticker.lower()}_data", conn, parse_dates=["date"])
            conn.close()

            df = df.sort_values("date")

            if period == "1 an":
                df = df[df["date"] >= datetime.now() - timedelta(days=365)]
            elif period == "5 ans":
                df = df[df["date"] >= datetime.now() - timedelta(days=5 * 365)]

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df["date"], y=df["close"], mode='lines', name='Close'))

            fig.update_layout(
                title=f"{ticker} - Historique des prix",
                xaxis_title="Date",
                yaxis_title="Prix de clôture",
                margin=dict(l=20, r=20, t=40, b=20),
                template="plotly_dark",
                autosize=True,
            )

            with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as f:
                fig.write_html(f.name, full_html=True, include_plotlyjs='True', config={"responsive": True})
                self.web_view.load(QUrl.fromLocalFile(f.name))

        except Exception as e:
            print(f"Erreur lors de l'affichage du graphique : {e}")

    def handle_selection(self, ticker, period, graph_type):
        print(f"➡️ Sélection : {ticker} / {period} / {graph_type}")
        try:
            if graph_type == "Historique des prix":
                self.plot_price_graph(ticker, period)
            elif graph_type == "Moyennes de volatilité":
                self.volatility_viewer.set_parameters(ticker, period)
                self.module_stack.setCurrentWidget(self.volatility_viewer)
            elif graph_type == "Prévision HAR-RV":
                self.har_rv_viewer.set_parameters(ticker, period)
                self.module_stack.setCurrentWidget(self.har_rv_viewer)
            elif graph_type == "Distribution":
                self.volatility_distribution.set_parameters(ticker, period)
                self.module_stack.setCurrentWidget(self.volatility_distribution)
            elif graph_type == "Heatmap journalière":
                self.volatilite_jour_semaine_viewer.set_parameters(ticker, period)
                self.module_stack.setCurrentWidget(self.volatilite_jour_semaine_viewer)
            elif graph_type == "Volatilité historique":
                self.volatilite_historique_viewer.set_parameters(ticker, period)
                self.module_stack.setCurrentWidget(self.volatilite_historique_viewer)
            elif graph_type == "Analyse des régimes de volatilité":
                self.module_stack.setCurrentWidget(self.volatility_regime_viewer)
                self.volatility_regime_viewer.update_view(ticker)
            else:
                print(f"⚠️ Graphique inconnu : {graph_type}")
        except Exception as e:
            print(f"❌ Erreur lors du chargement du graphe '{graph_type}' : {e}")

    def plot_price_graph(self, ticker, period):
        self.module_stack.setCurrentWidget(self.web_view)
        self.plot_prices(ticker, period)

    def return_to_home(self):
        parent = self.parent()
        while parent and not hasattr(parent, "change_page"):
            parent = parent.parent()
        if parent and callable(getattr(parent, "change_page", None)):
            parent.change_page("home")

    def refresh_selector(self):
        self.selector.refresh_ticker_list()
