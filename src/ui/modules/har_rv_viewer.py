import os
import sqlite3
import tempfile

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm
from pandas.tseries.offsets import BDay

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QComboBox, QPushButton
from PyQt6.QtCore import Qt, QUrl

# 🛡️ SAFE MODE: ne pas importer/créer QWebEngineView si SAFE_MODE=1
SAFE_MODE = os.getenv("SAFE_MODE", "0") == "1"
if not SAFE_MODE:
    try:
        from PyQt6.QtWebEngineWidgets import QWebEngineView
    except Exception as e:
        print(f"⚠️ WebEngine indisponible: {e}")
        QWebEngineView = None  # fallback
else:
    QWebEngineView = None


class HARVolatilityViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🤖 Prévision de la volatilité - Modèle HAR-RV")
        self.ticker = "SPX"

        layout = QVBoxLayout()

        self.period_select = QComboBox()
        self.period_select.addItems(["1 an", "5 ans", "Complet"])
        layout.addWidget(QLabel("Période :"))
        layout.addWidget(self.period_select)

        self.mode_select = QComboBox()
        self.mode_select.addItems(["C2C", "O2C"])
        layout.addWidget(QLabel("Type de volatilité réalisée :"))
        layout.addWidget(self.mode_select)

        self.refresh_button = QPushButton("Mettre à jour")
        self.refresh_button.clicked.connect(self.update_chart)
        layout.addWidget(self.refresh_button)

        # ✅ Zone d'affichage: WebEngine si dispo, sinon fallback QLabel
        if QWebEngineView is not None:
            self.web_view = QWebEngineView()
            self.web_view.setMinimumHeight(600)
            layout.addWidget(self.web_view)
            self.web_fallback_label = None
        else:
            self.web_view = None
            self.web_fallback_label = QLabel("SAFE MODE: Web view désactivée (ou WebEngine indisponible)")
            self.web_fallback_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.web_fallback_label.setMinimumHeight(600)
            self.web_fallback_label.setStyleSheet("color: red;")
            layout.addWidget(self.web_fallback_label)

        self.stats_label = QLabel()
        self.stats_label.setStyleSheet("""
            color: #eeeeee;
            font-family: Consolas;
            font-size: 12px;
            padding: 10px;
            background-color: #1e1e1e;
            border: 1px solid #444;
            border-radius: 5px;
        """)
        self.stats_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.stats_label)

        self.setLayout(layout)

    def set_parameters(self, ticker: str, period: str):
        self.ticker = ticker
        self.period_select.setCurrentText(period)
        self.update_chart()

    def update_chart(self):
        import numpy as np
        ticker = self.ticker
        period = self.period_select.currentText()
        mode = self.mode_select.currentText()
        table = f"{ticker.lower()}_data"

        # --- Load DB ---
        try:
            conn = sqlite3.connect("db/market_data.db")
            df = pd.read_sql(
                f"SELECT date, open, close FROM {table} ORDER BY date ASC",
                conn,
                parse_dates=['date']
            )
            conn.close()
        except Exception as e:
            self.stats_label.setText(f"❌ Erreur DB : {e}")
            return

        df.dropna(subset=['open', 'close'], inplace=True)

        if period == "1 an":
            df = df[df["date"] >= pd.Timestamp.today() - pd.Timedelta(days=365)]
        elif period == "5 ans":
            df = df[df["date"] >= pd.Timestamp.today() - pd.Timedelta(days=5 * 365)]

        # --- Realized Vol ---
        if mode == "C2C":
            df['rv'] = (df['close'] / df['close'].shift(1) - 1) ** 2
        else:
            df['rv'] = ((df['close'] - df['open']) / df['open']) ** 2

        df.dropna(inplace=True)
        df['rv5'] = df['rv'].rolling(window=5).mean()
        df['rv22'] = df['rv'].rolling(window=22).mean()
        df.dropna(subset=["rv", "rv5", "rv22"], inplace=True)

        if df.empty or df.shape[0] < 30:
            self.stats_label.setText("⚠️ Pas assez de données pour entraîner le modèle HAR-RV.")
            return

        df["target"] = df["rv"].shift(-1)
        last_valid_row = df.iloc[-1]
        next_date = last_valid_row["date"] + BDay(1)

        # --- HAR training ---
        train_df = df.dropna(subset=["target"])
        X = train_df[["rv", "rv5", "rv22"]]
        y = train_df["target"]
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()

        # --- In-sample predictions ---
        X_full = sm.add_constant(df[["rv", "rv5", "rv22"]])
        df["har_pred"] = model.predict(X_full)

        # --- Next-day prediction with CI ---
        future_X = pd.DataFrame([{
            "const": 1.0,
            "rv": last_valid_row["rv"],
            "rv5": last_valid_row["rv5"],
            "rv22": last_valid_row["rv22"]
        }])
        pred_result = model.get_prediction(future_X)
        pred_mean = pred_result.predicted_mean[0]
        lower, upper = pred_result.conf_int(alpha=0.05)[0]

        # --- Append next-day row for plotting ---
        df = pd.concat([df, pd.DataFrame([{
            "date": next_date,
            "rv": np.nan,
            "rv5": np.nan,
            "rv22": np.nan,
            "target": np.nan,
            "har_pred": pred_mean,
            "open": np.nan,
            "close": np.nan
        }])], ignore_index=True)

        # --- To annualized volatility (%) ---
        if pred_mean > 0:
            pred_vol = (pred_mean ** 0.5) * 100
            lower_vol = (lower ** 0.5) * 100 if lower > 0 else 0
            upper_vol = (upper ** 0.5) * 100 if upper > 0 else 0
        else:
            pred_vol = 0
            lower_vol = 0
            upper_vol = 0

        df['rv_pct'] = (df['rv'] ** 0.5) * 100
        df['har_pred_pct'] = (df['har_pred'] ** 0.5) * 100

        # --- Plot ---
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['rv_pct'],
            name=f"Volatilité réalisée {mode}",
            mode='lines'
        ))
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['har_pred_pct'],
            name="Prévision HAR-RV",
            mode='lines'
        ))
        fig.add_trace(go.Scatter(
            x=[next_date],
            y=[pred_vol],
            mode="markers",
            name="Prévision J+1",
            marker=dict(size=10, color="deepskyblue", symbol="circle"),
            error_y=dict(
                type='data',
                symmetric=False,
                array=[upper_vol - pred_vol],
                arrayminus=[pred_vol - lower_vol],
                color='deepskyblue',
                thickness=2,
                width=6
            ),
            hovertemplate=(
                "Prévision HAR-RV<br>"
                "Date : %{{x}}<br>"
                "Valeur : <b>%{{y:.2f}} %</b><br>"
                "Haut : <b>{:.2f} %</b><br>"
                "Bas : <b>{:.2f} %</b>"
            ).format(upper_vol, lower_vol),
            showlegend=True
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

        # --- Render: WebEngine si dispo, sinon fallback texte ---
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as f:
            fig.write_html(f.name, full_html=True, include_plotlyjs='inline', config={"responsive": True})
            html_path = f.name

        if self.web_view is not None:
            self.web_view.load(QUrl.fromLocalFile(html_path))
            if self.web_fallback_label is not None:
                self.web_fallback_label.hide()
        else:
            # Fallback: info + chemin du fichier généré
            self.web_fallback_label.setText(
                f"SAFE MODE: graphique généré.\n"
                f"Ouvre ce fichier dans ton navigateur :\n{html_path}"
            )

        # --- Stats model ---
        summary = [
            f"Observations: {len(train_df)}",
            f"Coef const: {model.params.get('const', np.nan):.6f}",
            f"Coef rv: {model.params.get('rv', np.nan):.6f}",
            f"Coef rv5: {model.params.get('rv5', np.nan):.6f}",
            f"Coef rv22: {model.params.get('rv22', np.nan):.6f}",
            f"R²: {model.rsquared:.4f}",
            f"R² ajusté: {model.rsquared_adj:.4f}",
            f"Prévision J+1 (vol%): {pred_vol:.2f}  (IC95%: {lower_vol:.2f} — {upper_vol:.2f})"
        ]
        self.stats_label.setText("\n".join(summary))
