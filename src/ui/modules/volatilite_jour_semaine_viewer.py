import os
import sqlite3
import pandas as pd
import plotly.graph_objects as go
from plotly.io import to_html
import tempfile

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QCheckBox, QLabel
from PyQt6.QtCore import QUrl

# 🛡️ SAFE MODE: pas de QWebEngineView si SAFE_MODE=1
SAFE_MODE = os.getenv("SAFE_MODE", "0") == "1"
if not SAFE_MODE:
    try:
        from PyQt6.QtWebEngineWidgets import QWebEngineView
    except Exception as e:
        print(f"⚠️ WebEngine indisponible: {e}")
        QWebEngineView = None
else:
    QWebEngineView = None


class VolatiliteJourSemaineViewer(QWidget):
    def __init__(self):
        super().__init__()

        self.ticker = "SPX"
        self.period = "1 an"
        self.mode = "C2C"
        self.use_abs = True
        self.db_path = "db/market_data.db"

        layout = QVBoxLayout()
        self.setLayout(layout)

        # Contrôles
        controls_layout = QHBoxLayout()

        self.mode_select = QComboBox()
        self.mode_select.addItems(["C2C", "O2C"])
        self.mode_select.currentTextChanged.connect(self.update_graph)
        controls_layout.addWidget(self.mode_select)

        self.abs_checkbox = QCheckBox("Volatilité absolue")
        self.abs_checkbox.setChecked(True)
        self.abs_checkbox.stateChanged.connect(self.update_graph)
        controls_layout.addWidget(self.abs_checkbox)

        layout.addLayout(controls_layout)

        # Zone d'affichage (WebEngine si dispo, sinon fallback QLabel)
        if QWebEngineView is not None:
            self.web_view = QWebEngineView()
            layout.addWidget(self.web_view)
            self.web_fallback = None
        else:
            self.web_view = None
            self.web_fallback = QLabel("SAFE MODE: Web view désactivée (ou WebEngine indisponible)")
            self.web_fallback.setStyleSheet("color: red;")
            self.web_fallback.setMinimumHeight(400)
            layout.addWidget(self.web_fallback)

        self.update_graph()

    def set_parameters(self, ticker, period):
        self.ticker = ticker
        self.period = period
        self.update_graph()

    def update_graph(self):
        self.mode = self.mode_select.currentText()
        self.use_abs = self.abs_checkbox.isChecked()

        html = compute_volatilite_jour_semaine(
            ticker=self.ticker,
            db_path=self.db_path,
            mode=self.mode,
            use_abs=self.use_abs
        )

        # 🔁 Affichage robuste: fichier temporaire + load()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as f:
            f.write(html.encode("utf-8"))
            html_path = f.name

        if self.web_view is not None:
            self.web_view.load(QUrl.fromLocalFile(html_path))
        else:
            self.web_fallback.setText(
                "SAFE MODE: heatmap générée.\n"
                f"Ouvre ce fichier dans ton navigateur :\n{html_path}"
            )


def compute_volatilite_jour_semaine(ticker, db_path, mode="C2C", use_abs=True):
    table = f"{ticker.lower()}_data"
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(f"SELECT date, open, close FROM {table} ORDER BY date ASC", conn)
    conn.close()

    if df.empty:
        # Retourne une page très simple pour éviter un écran blanc
        return "<html><body><h3>Aucune donnée disponible pour ce ticker.</h3></body></html>"

    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(subset="date")
    df = df[df["date"].dt.weekday < 5]  # uniquement lundi à vendredi

    if mode == "C2C":
        df["ret"] = df["close"].pct_change()
    else:
        df["ret"] = (df["close"] - df["open"]) / df["open"]

    df.dropna(subset=["ret"], inplace=True)
    if df.empty:
        return "<html><body><h3>Aucune donnée exploitable après nettoyage.</h3></body></html>"

    df["year"] = df["date"].dt.year
    df["weekday"] = df["date"].dt.day_name()

    if use_abs:
        df["vol"] = df["ret"].abs() * 100
        title = f"\U0001F525 Volatilité absolue moyenne par jour de la semaine ({mode}) — {ticker}"
        colorbar_title = "Volatilité absolue (%)"
        colorscale = "Blues"
    else:
        df["vol"] = df["ret"] * 100
        title = f"\U0001F525 Volatilité signée moyenne par jour de la semaine ({mode}) — {ticker}"
        colorbar_title = "Volatilité (%)"
        colorscale = "RdBu_r"

    pivot = df.pivot_table(index="year", columns="weekday", values="vol", aggfunc="mean")

    # S'assurer de l'ordre Lundi→Vendredi même si années incomplètes
    desired_cols = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    # garde seulement celles présentes pour éviter une heatmap toute blanche
    present_cols = [c for c in desired_cols if c in pivot.columns]
    pivot = pivot.reindex(columns=present_cols)

    if pivot.empty or len(present_cols) == 0:
        return "<html><body><h3>Aucune donnée pour construire la heatmap (colonnes manquantes).</h3></body></html>"

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=list(pivot.columns),
        y=list(pivot.index),
        colorscale=colorscale,
        colorbar=dict(title=colorbar_title),
        text=pivot.round(2).astype(str),
        texttemplate="%{text}%",
        hovertemplate="weekday: %{x}<br>year: %{y}<br>Volatilité : %{z:.2f}%<extra></extra>"
    ))

    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=50, r=30, t=60, b=50),
        height=700,
        hovermode='x unified',
        font=dict(size=12),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0.5)"
        ),
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.2)'
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.2)'
        )
    )

    # HTML autonome (inline = pas de CDN)
    return to_html(fig, include_plotlyjs='inline')
