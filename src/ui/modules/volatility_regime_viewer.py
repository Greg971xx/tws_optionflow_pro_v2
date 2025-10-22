import os
import sqlite3
from datetime import datetime, timedelta
import tempfile

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QTextEdit
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


class VolatilityRegimeViewer(QWidget):
    def __init__(self, db_path="db/market_data.db"):
        super().__init__()
        self.db_path = db_path
        self.ticker = None
        self.period = None
        self.df = None

        # Layout principal
        self.layout = QVBoxLayout(self)

        # Zone d'affichage: WebEngine si dispo, sinon fallback QLabel
        if QWebEngineView is not None:
            self.web_view = QWebEngineView()
            self.web_view.setMinimumHeight(700)
            self.layout.addWidget(self.web_view)
            self.web_fallback = None
        else:
            self.web_view = None
            self.web_fallback = QLabel("SAFE MODE: Web view désactivée (ou WebEngine indisponible)")
            self.web_fallback.setStyleSheet("color: red;")
            self.web_fallback.setMinimumHeight(400)
            self.layout.addWidget(self.web_fallback)

        # Zone de texte pour l’analyse
        self.analysis_box = QTextEdit()
        self.analysis_box.setReadOnly(True)
        self.layout.addWidget(self.analysis_box)

    # --- Helpers d'affichage ---
    def _render_html(self, html: str):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as f:
            f.write(html.encode("utf-8"))
            path = f.name
        if self.web_view is not None:
            self.web_view.load(QUrl.fromLocalFile(path))
        else:
            self.web_fallback.setText(
                "SAFE MODE: graphique généré.\n"
                f"Ouvre ce fichier dans ton navigateur :\n{path}"
            )

    # --- API publique ---
    def update_view(self, ticker, period=None):
        self.ticker = ticker
        self.period = period
        self.load_data()
        if self.df is not None and not self.df.empty:
            self.plot_chart()
            self.analyze_regime()
        else:
            self._render_html("<h3>Aucune donnée à afficher</h3>")

    # --- Data ---
    def load_data(self):
        try:
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query(
                "SELECT * FROM volatility_stats WHERE ticker = ? ORDER BY date",
                conn,
                params=[self.ticker]
            )
            conn.close()

            if df.empty:
                self.df = None
                self.analysis_box.setText("Aucune donnée trouvée dans volatility_stats pour ce ticker.")
                return

            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date")

            if self.period and self.period != "Complet":
                period_map = {
                    "1 an": 252,
                    "2 ans": 252 * 2,
                    "5 ans": 252 * 5
                }
                nb_days = period_map.get(self.period, 252)
                df = df.tail(nb_days)

            self.df = df.reset_index(drop=True)

        except Exception as e:
            self.df = None
            self.analysis_box.setText(f"Erreur lors du chargement des données : {e}")

    # --- Plot ---
    def plot_chart(self):
        fig = go.Figure()

        # Volatilité réalisée C2C
        if "ma20_c2c" in self.df.columns:
            fig.add_trace(go.Scatter(
                x=self.df["date"], y=self.df["ma20_c2c"] * np.sqrt(252), name="VR20 C2C"
            ))
        if "ma252_c2c" in self.df.columns:
            fig.add_trace(go.Scatter(
                x=self.df["date"], y=self.df["ma252_c2c"] * np.sqrt(252), name="VR252 C2C"
            ))

        # Volatilité réalisée O2C
        if "ma20_o2c" in self.df.columns:
            fig.add_trace(go.Scatter(
                x=self.df["date"], y=self.df["ma20_o2c"] * np.sqrt(252), name="VR20 O2C"
            ))
        if "ma252_o2c" in self.df.columns:
            fig.add_trace(go.Scatter(
                x=self.df["date"], y=self.df["ma252_o2c"] * np.sqrt(252), name="VR252 O2C"
            ))

        # Volatilité historique (déjà annualisée dans la table, a priori)
        if "hv5" in self.df.columns:
            fig.add_trace(go.Scatter(x=self.df["date"], y=self.df["hv5"], name="VH5 C2C"))
        if "hv20" in self.df.columns:
            fig.add_trace(go.Scatter(x=self.df["date"], y=self.df["hv20"], name="VH20 C2C"))
        if "hv252" in self.df.columns:
            fig.add_trace(go.Scatter(x=self.df["date"], y=self.df["hv252"], name="VH252 C2C"))

        # Rank 252 (de 0 à 1) en axe secondaire
        if "rank252_ma20_c2c" in self.df.columns:
            fig.add_trace(go.Scatter(
                x=self.df["date"], y=self.df["rank252_ma20_c2c"],
                name="Rank 252 glissant", yaxis="y2"
            ))

        fig.update_layout(
            title=f"Volatilité – {self.ticker}",
            xaxis=dict(title="Date"),
            yaxis=dict(title="Volatilité"),
            yaxis2=dict(title="Ranks", overlaying="y", side="right"),
            template="plotly_dark",
            margin=dict(l=40, r=20, t=60, b=40),
            height=650,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        html = f"""
        <html>
        <head>
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <style>body {{ margin: 0; padding: 0; }} .responsive-plot {{ width: 100%; height: 100%; }}</style>
        </head>
        <body>
            <div class="responsive-plot">
                {fig.to_html(include_plotlyjs='inline', full_html=True, config={"responsive": True})}
            </div>
        </body>
        </html>
        """

        self._render_html(html)

    # --- Analyse textuelle ---
    def analyze_regime(self):
        try:
            if self.df is None or self.df.empty:
                self.analysis_box.setPlainText("Aucune donnée pour analyse.")
                return

            last_row = self.df.iloc[-1]
            lines = []

            # Analyse VR C2C
            if "ma20_c2c" in last_row and "ma252_c2c" in last_row:
                vr20 = (last_row["ma20_c2c"] or 0) * 100
                vr252 = (last_row["ma252_c2c"] or 0) * 100
                diff = ((vr20 - vr252) / vr252) * 100 if vr252 else 0
                tendance = "🔺 en hausse" if diff > 0 else "🔻 en baisse"
                lines.append(f"- VR20 C2C : {vr20:.2f}% | VR252 C2C : {vr252:.2f}% → {tendance} ({diff:.0f}% vs moyenne)")
                lines.append(f"  ↳ Volatilité journalière moyenne actuelle ≈ {vr20:.2f}%")

            # Analyse VR O2C
            if "ma20_o2c" in last_row and "ma252_o2c" in last_row:
                vr20 = (last_row["ma20_o2c"] or 0) * 100
                vr252 = (last_row["ma252_o2c"] or 0) * 100
                diff = ((vr20 - vr252) / vr252) * 100 if vr252 else 0
                tendance = "🔺 en hausse" if diff > 0 else "🔻 en baisse"
                lines.append(f"- VR20 O2C : {vr20:.2f}% | VR252 O2C : {vr252:.2f}% → {tendance} ({diff:.0f}% vs moyenne)")
                lines.append(f"  ↳ Volatilité intra-journalière moyenne actuelle ≈ {vr20:.2f}%")

            # Analyse VH
            if "hv5" in last_row and "hv252" in last_row:
                vh5 = (last_row["hv5"] or 0)
                vh252 = (last_row["hv252"] or 0)
                diff = ((vh5 - vh252) / vh252) * 100 if vh252 else 0
                tendance = "🔺 en hausse" if diff > 0 else "🔻 en baisse"
                lines.append(f"- VH5 : {vh5:.2f}% | VH252 : {vh252:.2f}% → {tendance} ({diff:.0f}% vs long terme)")
                ratio = (vh5 / vh252) if vh252 else None
                if ratio:
                    if ratio > 2:
                        lines.append("🔍 Pattern : Tension extrême (VH5 > 2 × VH252)")
                    elif ratio < 0.5:
                        lines.append("🔍 Pattern : Compression extrême (VH5 < 0.5 × VH252)")

            # Rank
            if "rank252_ma20_c2c" in last_row:
                rank = (last_row["rank252_ma20_c2c"] or 0) * 100
                if rank < 20:
                    lines.append(f"- 🧊 Rank 252 bas — vol historiquement basse ({rank:.0f}%)")
                elif rank > 80:
                    lines.append(f"- 🔥 Rank 252 élevé — vol historiquement élevée ({rank:.0f}%)")
                else:
                    lines.append(f"- 📊 Rank 252 médian — vol modérée ({rank:.0f}%)")

            # Orientation simple
            if len(self.df) > 5 and "ma20_c2c" in self.df.columns:
                series = self.df["ma20_c2c"].fillna(0)
                trend = series.diff().rolling(5).mean().iloc[-1]
                orientation = "📈 Tendance haussière (MA20 C2C)" if trend > 0 else "📉 Tendance baissière (MA20 C2C)"
                lines.append(f"- {orientation}")

                if all(k in last_row for k in ["ma20_c2c", "ma120_c2c", "ma252_c2c"]):
                    vr20 = last_row["ma20_c2c"] or 0
                    vr120 = last_row["ma120_c2c"] or 0
                    vr252 = last_row["ma252_c2c"] or 0
                    if vr20 < vr120 < vr252:
                        lines.append("🔍 Pattern : Compression extrême (VR20 < VR120 < VR252)")
                    elif vr20 > vr120 > vr252:
                        lines.append("🔍 Pattern : Tension extrême (VR20 > VR120 > VR252)")
                    if vr20 < vr252 and trend > 0:
                        lines.append("🔍 Pattern : Reversion haussière (VR20 < VR252 et pente > 0)")
                    elif vr20 > vr252 and trend < 0:
                        lines.append("🔍 Pattern : Reversion baissière (VR20 > VR252 et pente < 0)")

                if len(self.df) > 3 and "ma20_c2c" in self.df.columns:
                    vr20_now = series.iloc[-1]
                    vr20_3d_ago = series.iloc[-4]
                    if vr20_3d_ago:
                        jump = ((vr20_now - vr20_3d_ago) / vr20_3d_ago) * 100
                        if jump > 30:
                            lines.append("🔍 Pattern : Saut de volatilité +30% sur 3j")
                        elif jump < -30:
                            lines.append("🔍 Pattern : Baisse de volatilité -30% sur 3j")

            text = f"🎯 Diagnostic volatilité – {self.ticker}\n" + "\n".join(lines) if lines else "Aucun diagnostic disponible."
            self.analysis_box.setPlainText(text)

        except Exception as e:
            self.analysis_box.setPlainText(f"Erreur lors de l'analyse du régime : {e}")
