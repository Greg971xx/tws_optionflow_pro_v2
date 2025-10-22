import os
import sqlite3
import tempfile
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go

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


def compute_rank_percentile(series: pd.Series, window: int = 252) -> pd.DataFrame:
    """
    Calcule value, rank (glissant), percentile (glissant) et pente sur 5 jours.
    """
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


class VolatilityViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Volatilité Réalisée")

        self.ticker = "SPX"
        self.period = "1 an"
        self.vol_type = "Les deux"
        self.mode = "Pourcentage"
        self.show_daily = True

        main_layout = QVBoxLayout()
        options_layout = QHBoxLayout()

        # Zone d'affichage (WebEngine si dispo, sinon fallback QLabel)
        if QWebEngineView is not None:
            self.web_view = QWebEngineView()
            self.web_view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            self.display_widget = self.web_view
        else:
            self.web_view = None
            self.display_widget = QLabel("SAFE MODE: Web view désactivée (ou WebEngine indisponible)")
            self.display_widget.setMinimumHeight(500)
            self.display_widget.setStyleSheet("color: red;")

        self.vol_type_select = QComboBox()
        self.vol_type_select.addItems(["C2C", "O2C", "Les deux"])
        self.vol_type_select.currentIndexChanged.connect(self.update_plot)

        self.display_mode = QComboBox()
        self.display_mode.addItems(["Pourcentage", "Valeur absolue"])
        self.display_mode.currentIndexChanged.connect(self.update_plot)

        self.show_daily_checkbox = QCheckBox("Afficher la volatilité journalière")
        self.show_daily_checkbox.setChecked(True)
        self.show_daily_checkbox.stateChanged.connect(self.update_plot)

        update_button = QPushButton("Mettre à jour le graphique")
        update_button.clicked.connect(self.update_plot)

        self.form_layout = QFormLayout()
        self.form_layout.setSpacing(5)
        self.form_layout.addRow("Type de VR :", self.vol_type_select)
        self.form_layout.addRow("Affichage :", self.display_mode)
        self.form_layout.addRow("", self.show_daily_checkbox)
        self.form_layout.addRow("", update_button)

        self.group_box = QGroupBox("Options de volatilité")
        self.group_box.setCheckable(True)
        self.group_box.setChecked(True)
        self.group_box.setLayout(self.form_layout)
        self.group_box.toggled.connect(self.toggle_option_visibility)

        options_layout.addWidget(self.group_box)

        main_layout.addLayout(options_layout)
        main_layout.addWidget(self.display_widget, stretch=1)
        self.setLayout(main_layout)

    def toggle_option_visibility(self, checked):
        for i in range(self.form_layout.count()):
            item = self.form_layout.itemAt(i)
            if item and item.widget():
                item.widget().setVisible(checked)

    def set_parameters(self, ticker: str, period: str):
        self.ticker = ticker
        self.period = period
        self.update_plot()

    def update_plot(self):
        self.vol_type = self.vol_type_select.currentText()
        self.mode = self.display_mode.currentText()
        self.show_daily = self.show_daily_checkbox.isChecked()
        self.plot()

    def plot(self):
        ticker = self.ticker
        period = self.period
        vol_type = self.vol_type
        mode = self.mode

        conn = sqlite3.connect("db/market_data.db")
        df = pd.read_sql(f"SELECT * FROM {ticker.lower()}_data", conn, parse_dates=["date"])
        conn.close()

        df = df.sort_values("date")

        if period == "1 an":
            df = df[df["date"] >= datetime.now() - timedelta(days=365)]
        elif period == "5 ans":
            df = df[df["date"] >= datetime.now() - timedelta(days=5 * 365)]

        df.dropna(inplace=True)
        df["C2C"] = df["close"].pct_change()
        df["O2C"] = (df["close"] - df["open"]) / df["open"]

        if mode == "Valeur absolue":
            df["C2C"] = df["C2C"].abs()
            df["O2C"] = df["O2C"].abs()

        for window in [5, 20, 60, 120, 252]:
            df[f"C2C_MA_{window}"] = df["C2C"].rolling(window).mean()
            df[f"O2C_MA_{window}"] = df["O2C"].rolling(window).mean()

        fig = go.Figure()
        format_y = lambda y: y * 100
        suffix_y = "%"

        if self.show_daily:
            if vol_type in ["C2C", "Les deux"]:
                fig.add_trace(go.Scatter(
                    x=df['date'], y=format_y(df["C2C"]),
                    mode='lines', name="C2C journalier",
                    line=dict(width=1, dash='dot')
                ))
            if vol_type in ["O2C", "Les deux"]:
                fig.add_trace(go.Scatter(
                    x=df['date'], y=format_y(df["O2C"]),
                    mode='lines', name="O2C journalier",
                    line=dict(width=1, dash='dot')
                ))

        if vol_type in ["C2C", "Les deux"]:
            for window in [5, 20, 60, 120, 252]:
                col = f"C2C_MA_{window}"
                series = df[col].dropna()

                if len(series) < 30:
                    continue

                try:
                    stats = compute_rank_percentile(series, window=min(252, len(series)))

                    if stats.empty or len(stats) == 0:
                        continue

                    # IMPORTANT: Ne pas reset_index, garder l'index original pour alignment
                    # Trouver les dates correspondantes aux stats (même index que series)
                    stats_with_dates = df.loc[stats.index, ['date']].copy()
                    stats_with_dates = stats_with_dates.join(stats)

                    # Convert to percentage for display
                    rank_pct = stats_with_dates["rank"] * 100
                    percentile_pct = stats_with_dates["percentile"] * 100
                    slope_pct = stats_with_dates["slope"] * 100

                    fig.add_trace(go.Scatter(
                        x=stats_with_dates["date"],
                        y=format_y(stats_with_dates["value"]),
                        mode='lines',
                        name=f"C2C MM{window}",
                        line=dict(width=2),
                        customdata=np.stack([
                            rank_pct,
                            percentile_pct,
                            slope_pct
                        ], axis=-1),
                        hovertemplate=(
                            f"<b>Date</b> : %{{x|%Y-%m-%d}}<br>"
                            f"<b>C2C MM{window}</b> : %{{y:.2f}}%<br>"
                            "<b>Rank</b> : %{customdata[0]:.1f}%<br>"
                            "<b>Percentile</b> : %{customdata[1]:.1f}%<br>"
                            "<b>Tendance 5j</b> : %{customdata[2]:+.2f}%"
                            "<extra></extra>"
                        )
                    ))
                except Exception as e:
                    print(f"⚠️ Erreur enrichissement C2C MM{window} : {e}")
                    import traceback
                    traceback.print_exc()
                    # Fallback sans enrichissement
                    fig.add_trace(go.Scatter(
                        x=df["date"],
                        y=format_y(df[col]),
                        mode="lines",
                        name=f"C2C MM{window}",
                        line=dict(width=2)
                    ))

        if vol_type in ["O2C", "Les deux"]:
            for window in [5, 20, 60, 120, 252]:
                col = f"O2C_MA_{window}"
                series = df[col].dropna()

                if len(series) < 30:
                    continue

                try:
                    stats = compute_rank_percentile(series, window=min(252, len(series)))

                    if stats.empty or len(stats) == 0:
                        continue

                    # Même logique pour O2C
                    stats_with_dates = df.loc[stats.index, ['date']].copy()
                    stats_with_dates = stats_with_dates.join(stats)

                    rank_pct = stats_with_dates["rank"] * 100
                    percentile_pct = stats_with_dates["percentile"] * 100
                    slope_pct = stats_with_dates["slope"] * 100

                    fig.add_trace(go.Scatter(
                        x=stats_with_dates["date"],
                        y=format_y(stats_with_dates["value"]),
                        mode='lines',
                        name=f"O2C MM{window}",
                        line=dict(width=2),
                        customdata=np.stack([
                            rank_pct,
                            percentile_pct,
                            slope_pct
                        ], axis=-1),
                        hovertemplate=(
                            f"<b>Date</b> : %{{x|%Y-%m-%d}}<br>"
                            f"<b>O2C MM{window}</b> : %{{y:.2f}}%<br>"
                            "<b>Rank</b> : %{customdata[0]:.1f}%<br>"
                            "<b>Percentile</b> : %{customdata[1]:.1f}%<br>"
                            "<b>Tendance 5j</b> : %{customdata[2]:+.2f}%"
                            "<extra></extra>"
                        )
                    ))
                except Exception as e:
                    print(f"⚠️ Erreur enrichissement O2C MM{window} : {e}")
                    import traceback
                    traceback.print_exc()
                    # Fallback
                    fig.add_trace(go.Scatter(
                        x=df["date"],
                        y=format_y(df[col]),
                        mode="lines",
                        name=f"O2C MM{window}",
                        line=dict(width=2)
                    ))

        # Prix en arrière-plan
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["close"],
            mode="lines", name="Prix",
            yaxis="y2", line=dict(dash="dot")
        ))

        fig.update_layout(
            title=f"{ticker} - Volatilité Réalisée ({vol_type})",
            xaxis=dict(
                title="Date",
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128,128,128,0.2)'
            ),
            yaxis=dict(
                title=f"Volatilité réalisée {suffix_y}",
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128,128,128,0.2)'
            ),
            yaxis2=dict(
                title="Prix du sous-jacent",
                overlaying="y",
                side="right",
                showgrid=False
            ),
            template="plotly_dark",
            margin=dict(l=50, r=70, t=60, b=50),
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
            )
        )

        # Rendu: QWebEngine si dispo, sinon fallback avec chemin du HTML
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as f:
            fig.write_html(f.name, full_html=True, include_plotlyjs='inline', config={"responsive": True})
            html_path = f.name

        if self.web_view is not None:
            self.web_view.load(QUrl.fromLocalFile(html_path))
        else:
            if isinstance(self.display_widget, QLabel):
                self.display_widget.setText(
                    "SAFE MODE: graphique généré.\n"
                    f"Ouvre ce fichier dans ton navigateur :\n{html_path}"
                )
