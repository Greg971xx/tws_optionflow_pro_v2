import os
import sqlite3
from datetime import datetime, timedelta
import tempfile

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.io import to_html

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QComboBox, QCheckBox, QPushButton,
    QHBoxLayout, QGroupBox, QFormLayout, QSizePolicy
)
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

from core.greeks_fetcher import update_iv_from_greeks


def compute_rank_percentile_slope(series: pd.Series, window: int = 252) -> pd.DataFrame:
    """Calcule Rank, Percentile et Tendance sur les séries de volatilité."""
    series = series.dropna()

    rank = series.rolling(window).apply(
        lambda x: (x[-1] - x.min()) / (x.max() - x.min()) if (x.max() - x.min()) != 0 else np.nan,
        raw=True
    )

    percentile = series.rolling(window).apply(
        lambda x: (x < x[-1]).mean(),
        raw=True
    )

    slope = pd.Series(index=series.index, dtype=float)
    for i in range(5, len(series)):
        old = series.iloc[i - 5]
        new = series.iloc[i]
        slope.iloc[i] = (new - old) / old if old != 0 else 0

    return pd.DataFrame({
        "value": series,
        "rank": rank,
        "percentile": percentile,
        "slope": slope
    })


class VolatiliteHistoriqueViewer(QWidget):
    def __init__(self):
        super().__init__()

        self.ticker = "SPX"
        self.period = "1 an"
        self.db_path = "db/market_data.db"

        layout = QVBoxLayout()
        self.setLayout(layout)

        self.update_button = QPushButton("🔁 Mettre à jour IV + Greeks")
        self.update_button.clicked.connect(self.update_iv_data)
        layout.addWidget(self.update_button)

        # Zone d'affichage: WebEngine si dispo, sinon fallback QLabel
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
        html = compute_volatilite_historique(
            ticker=self.ticker,
            db_path=self.db_path,
            period=self.period
        )
        from PyQt6.QtCore import QUrl
        import tempfile

        if self.web_view is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as f:
                f.write(html.encode("utf-8"))
                html_path = f.name
            self.web_view.load(QUrl.fromLocalFile(html_path))
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as f:
                f.write(html.encode("utf-8"))
                html_path = f.name
            self.web_fallback.setText(
                "SAFE MODE: graphique généré.\n"
                f"Ouvre ce fichier dans ton navigateur :\n{html_path}"
            )

    def update_iv_data(self):
        try:
            print(f"⏳ Mise à jour IV + Greeks pour {self.ticker}...")
            iv_0dte, _ = update_iv_from_greeks(self.ticker)

            if iv_0dte:
                print(f"✅ IV 0DTE calculée pour {self.ticker} : {iv_0dte * 100:.2f}%")
            else:
                print(f"⚠️ IV 0DTE non disponible pour {self.ticker}")

            self.update_graph()

        except Exception as e:
            print(f"❌ Erreur lors de la mise à jour IV/Greeks pour {self.ticker} : {e}")


def compute_volatilite_historique(ticker, db_path, period="1 an"):
    table = f"{ticker.lower()}_data"
    conn = sqlite3.connect(db_path)

    df = pd.read_sql_query(f"SELECT date, close FROM {table} ORDER BY date ASC", conn)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.drop_duplicates(subset="date")
    df = df[df["date"].dt.weekday < 5]
    df = df.sort_values("date")
    df["returns"] = df["close"].pct_change()

    # Lecture IV depuis greeks_observations (delta ~ 0.5)
    iv_df = pd.read_sql_query(
        """
        SELECT date, iv, delta
        FROM greeks_observations
        WHERE ticker = ? AND type = 'CALL'
        """,
        conn, params=(ticker,)
    )
    conn.close()

    iv_df["date"] = pd.to_datetime(iv_df["date"]).dt.normalize()
    iv_df["delta_distance"] = (iv_df["delta"] - 0.5).abs()
    iv_filtered = iv_df.sort_values("delta_distance").drop_duplicates(subset="date")
    iv_filtered = iv_filtered.rename(columns={"iv": "iv_0dte"})[["date", "iv_0dte"]]
    iv_filtered["iv_0dte"] *= 100

    # Ajoute la ligne du jour si absente (pour afficher une prédiction/continuité)
    # Ajoute la ligne du jour si absente (pour afficher une prédiction/continuité)
    today = pd.to_datetime(datetime.now().date()).normalize()
    if today not in df["date"].values:
        last_close = df["close"].iloc[-1]

        # Crée un dictionnaire avec toutes les colonnes présentes dans df
        row_dict = {col: np.nan for col in df.columns}
        row_dict["date"] = today
        row_dict["close"] = last_close

        # Convertir en DataFrame et concaténer
        new_row = pd.DataFrame([row_dict])
        df = pd.concat([df, new_row], ignore_index=True)
        df = df.sort_values("date")

    # Merge IV
    df = pd.merge(df, iv_filtered, on="date", how="left")

    # Filtre période
    if period == "1 an":
        df = df[df["date"] >= datetime.now() - timedelta(days=365)]
    elif period == "5 ans":
        df = df[df["date"] >= datetime.now() - timedelta(days=5 * 365)]

    # HV annualisée (%)
    for window in [5, 20, 60, 120, 252]:
        df[f"HV{window}"] = df["returns"].rolling(window).std() * (252 ** 0.5) * 100

    fig = go.Figure()

    # Courbes HV + stats enrichies
    for window in [5, 20, 60, 120, 252]:
        col = f"HV{window}"
        stats = compute_rank_percentile_slope(df[col])
        aligned_dates = df["date"].iloc[-len(stats):]

        fig.add_trace(go.Scatter(
            x=aligned_dates,
            y=stats["value"],
            mode="lines",
            name=f"HV{window}",
            customdata=np.stack([stats["rank"], stats["percentile"], stats["slope"]], axis=-1),
            hovertemplate=(
                f"Date : %{{x|%Y-%m-%d}}<br>"
                f"{col} : %{{y:.2f}} %<br>"
                "Rank : %{customdata[0]:.0%}<br>"
                "Percentile : %{customdata[1]:.0%}<br>"
                "Tendance : %{customdata[2]:+.1%} sur 5j"
                "<extra></extra>"
            )
        ))

    # IV 0DTE si dispo
    if df["iv_0dte"].notna().any():
        fig.add_trace(go.Scatter(
            x=df["date"],
            y=df["iv_0dte"],
            mode="lines+markers",
            name="IV 0DTE",
            line=dict(dash="dash"),
            marker=dict(size=6)
        ))

    fig.update_layout(
        title=f"Volatilité Historique — {ticker}",
        xaxis_title="Date",
        yaxis_title="Volatilité annualisée (%)",
        template="plotly_dark",
        margin=dict(l=40, r=20, t=60, b=40),
        height=650,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    # HTML autonome pour QWebEngineView.setHtml
    return to_html(fig, include_plotlyjs='inline')
