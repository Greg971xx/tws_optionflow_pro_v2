"""
GEX Tab - Gamma Exposure Analysis with OI visualization
"""
import os
import tempfile
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QComboBox, QPushButton, QGroupBox)
from PyQt6.QtCore import Qt, QTimer, QUrl
from PyQt6.QtGui import QFont

SAFE_MODE = os.getenv("SAFE_MODE", "0") == "1"
if not SAFE_MODE:
    try:
        from PyQt6.QtWebEngineWidgets import QWebEngineView
    except Exception:
        QWebEngineView = None
else:
    QWebEngineView = None

from src.core.options_flow_analyzer import load_comprehensive_data, get_available_expiries
from src.core.gex_calculator import EnhancedGEXAnalyzer
from src.core.oi_fetcher import latest_oi_frame, get_oi_snapshot_meta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.config import OPTIONFLOW_DB


class GEXTab(QWidget):
    def __init__(self):
        super().__init__()
        self.db_path = OPTIONFLOW_DB
        self.current_expiry = "ALL"
        self.analyzer = EnhancedGEXAnalyzer(multiplier=100.0, prefer_ib_gamma=True)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)

        # Controls
        controls_layout = QHBoxLayout()

        expiry_label = QLabel("Expiration :")
        expiry_label.setStyleSheet("font-weight: bold; color: white;")

        self.expiry_combo = QComboBox()
        self.expiry_combo.setMinimumWidth(250)
        self.expiry_combo.setStyleSheet("""
            QComboBox {
                background: #2d2d2d;
                color: white;
                border: 1px solid #444;
                padding: 5px 10px;
                border-radius: 3px;
            }
        """)
        self.load_expiries()
        self.expiry_combo.currentTextChanged.connect(self.on_expiry_changed)

        refresh_btn = QPushButton("Rafraichir")
        refresh_btn.setStyleSheet("""
            QPushButton {
                background: #4CAF50;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 3px;
                font-weight: bold;
            }
            QPushButton:hover { background: #45a049; }
        """)
        refresh_btn.clicked.connect(self.refresh_data)

        controls_layout.addWidget(expiry_label)
        controls_layout.addWidget(self.expiry_combo)
        controls_layout.addStretch()
        controls_layout.addWidget(refresh_btn)

        layout.addLayout(controls_layout)

        # Data source indicator
        self.source_label = QLabel("Chargement...")
        self.source_label.setStyleSheet("""
            color: #FFA500;
            font-size: 12px;
            font-weight: bold;
            padding: 10px;
            background: #2d2d2d;
            border-radius: 3px;
            border-left: 4px solid #FFA500;
        """)
        layout.addWidget(self.source_label)

        # Metrics
        metrics_layout = QHBoxLayout()
        self.total_gex_card = self.create_metric_card("Total GEX", "0", "#2196F3")
        self.zero_gamma_card = self.create_metric_card("Zero-Gamma", "-", "#FF9800")
        self.spot_card = self.create_metric_card("Spot", "-", "#4CAF50")

        metrics_layout.addWidget(self.total_gex_card)
        metrics_layout.addWidget(self.zero_gamma_card)
        metrics_layout.addWidget(self.spot_card)

        layout.addLayout(metrics_layout)

        # Charts
        if QWebEngineView is not None:
            # GEX Chart
            gex_label = QLabel("GEX par Strike")
            gex_label.setStyleSheet("color: white; font-weight: bold; font-size: 13px; padding: 5px;")
            layout.addWidget(gex_label)

            self.gex_chart_view = QWebEngineView()
            self.gex_chart_view.setMinimumHeight(400)
            layout.addWidget(self.gex_chart_view)

            # OI Chart
            oi_label = QLabel("Open Interest par Strike")
            oi_label.setStyleSheet("color: white; font-weight: bold; font-size: 13px; padding: 5px;")
            layout.addWidget(oi_label)

            self.oi_chart_view = QWebEngineView()
            self.oi_chart_view.setMinimumHeight(350)
            layout.addWidget(self.oi_chart_view)
        else:
            self.gex_chart_view = None
            self.oi_chart_view = None
            fallback = QLabel("SAFE MODE: Charts disabled")
            fallback.setAlignment(Qt.AlignmentFlag.AlignCenter)
            fallback.setStyleSheet("color: red; padding: 50px;")
            layout.addWidget(fallback)

        self.setLayout(layout)
        self.refresh_data()

    def create_metric_card(self, title, value, color):
        card = QGroupBox(title)
        card.setStyleSheet(f"""
            QGroupBox {{
                font-weight: bold;
                font-size: 12px;
                color: white;
                border: 2px solid {color};
                border-radius: 5px;
                margin-top: 8px;
                padding-top: 8px;
                background: #2d2d2d;
            }}
        """)

        card_layout = QVBoxLayout()
        value_label = QLabel(value)
        value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        value_label.setFont(QFont("Arial", 15, QFont.Weight.Bold))
        value_label.setStyleSheet(f"color: {color}; padding: 15px;")
        value_label.setObjectName("value_label")

        card_layout.addWidget(value_label)
        card.setLayout(card_layout)
        card.setMinimumHeight(100)

        return card

    def load_expiries(self):
        expiries = get_available_expiries(self.db_path)
        self.expiry_combo.clear()
        self.expiry_combo.addItem("ALL", "ALL")
        for exp in expiries:
            self.expiry_combo.addItem(exp['label'], exp['value'])
        if expiries:
            self.expiry_combo.setCurrentIndex(1)

    def on_expiry_changed(self, text):
        idx = self.expiry_combo.currentIndex()
        self.current_expiry = self.expiry_combo.itemData(idx)
        self.refresh_data()

    def refresh_data(self):
        df = load_comprehensive_data(
            db_path=self.db_path,
            selected_expiry=self.current_expiry,
            sample_size=100000,
            min_volume_filter=0,
            confidence_threshold=0.6
        )

        if df.empty:
            self.source_label.setText("❌ Aucune donnée disponible pour cette expiry")
            return

        spot = float(df["spot"].replace(0, float('nan')).dropna().median()) if "spot" in df.columns else 6700.0

        # Check if OI available
        expiry_for_oi = self.current_expiry if self.current_expiry != "ALL" else None
        use_oi = False
        oi_df = None

        if expiry_for_oi:
            last_ts, n_total, n_calls, n_puts = get_oi_snapshot_meta(self.db_path, expiry_for_oi)
            if n_total > 0:
                use_oi = True
                oi_df = latest_oi_frame(self.db_path, expiry_for_oi)
                self.source_label.setText(
                    f"✅ GEX calculé sur OI réel | "
                    f"Snapshot: {last_ts} | "
                    f"{n_total} strikes ({n_calls} calls, {n_puts} puts)"
                )
                self.source_label.setStyleSheet("""
                    color: #4CAF50;
                    font-size: 12px;
                    font-weight: bold;
                    padding: 10px;
                    background: #2d2d2d;
                    border-radius: 3px;
                    border-left: 4px solid #4CAF50;
                """)
            else:
                self.source_label.setText(
                    "⚠️ GEX calculé sur flux estimés (OI indisponible) | "
                    "Fetch OI dans 'OI Manager' pour calcul précis"
                )
                self.source_label.setStyleSheet("""
                    color: #FFA500;
                    font-size: 12px;
                    font-weight: bold;
                    padding: 10px;
                    background: #2d2d2d;
                    border-radius: 3px;
                    border-left: 4px solid #FFA500;
                """)
        else:
            self.source_label.setText(
                "⚠️ GEX calculé sur flux estimés (mode ALL expiries) | "
                "Sélectionner une expiry spécifique pour utiliser OI"
            )
            self.source_label.setStyleSheet("""
                color: #FFA500;
                font-size: 12px;
                font-weight: bold;
                padding: 10px;
                background: #2d2d2d;
                border-radius: 3px;
                border-left: 4px solid #FFA500;
            """)

        # Calculate GEX
        gex_df, critical, detailed = self.analyzer.analyze(
            df, spot,
            use_real_oi=use_oi,
            db_path=self.db_path if use_oi else None,
            expiry=expiry_for_oi if use_oi else None
        )

        # Update metrics
        self.update_card_value(self.total_gex_card, f"{critical.get('total_gex', 0):,.0f}")
        zg = critical.get('zero_gamma')
        self.update_card_value(self.zero_gamma_card, f"{zg:.0f}" if zg else "-")
        self.update_card_value(self.spot_card, f"{detailed.get('spot_used', 0):.2f}")

        # Update charts
        if self.gex_chart_view is not None:
            self.update_gex_chart(gex_df, critical, spot)

        if self.oi_chart_view is not None:
            if use_oi and oi_df is not None and not oi_df.empty:
                self.update_oi_chart(oi_df, spot)
            else:
                self.show_no_oi_message()

    def update_card_value(self, card, value):
        label = card.findChild(QLabel, "value_label")
        if label:
            label.setText(value)

    def update_gex_chart(self, gex_df, critical, spot):
        if gex_df.empty:
            return

        fig = go.Figure()

        # GEX bars
        colors = ['green' if x > 0 else 'red' for x in gex_df["gex"]]
        fig.add_trace(go.Bar(
            x=gex_df["strike"],
            y=gex_df["gex"],
            name="GEX",
            marker_color=colors,
            hovertemplate="Strike: %{x}<br>GEX: %{y:,.0f}<extra></extra>"
        ))

        # Zero-Gamma line
        if critical.get("zero_gamma"):
            fig.add_vline(
                x=critical["zero_gamma"],
                line_color="yellow",
                line_dash="dash",
                line_width=2,
                annotation_text="Zero-Gamma",
                annotation_position="top"
            )

        # Spot line
        fig.add_vline(
            x=spot,
            line_color="white",
            line_dash="dot",
            line_width=2,
            annotation_text="Spot",
            annotation_position="bottom"
        )

        fig.update_layout(
            title="GEX by Strike (Green=Support, Red=Resistance)",
            xaxis_title="Strike",
            yaxis_title="GEX",
            template="plotly_dark",
            height=450,
            hovermode='x unified'
        )

        with tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode='w', encoding='utf-8') as f:
            fig.write_html(f, include_plotlyjs='inline', full_html=True)
            html_path = f.name

        self.gex_chart_view.load(QUrl.fromLocalFile(html_path))

    def update_oi_chart(self, oi_df, spot):
        """Display OI by strike with Call/Put separation"""
        if oi_df.empty:
            return

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("CALL Open Interest", "PUT Open Interest"),
            horizontal_spacing=0.1
        )

        # Calls
        calls = oi_df[oi_df['option_type'] == 'CALL'].sort_values('strike')
        if not calls.empty:
            fig.add_trace(
                go.Bar(
                    x=calls['strike'],
                    y=calls['open_interest'],
                    name='CALL OI',
                    marker_color='green',
                    hovertemplate="Strike: %{x}<br>OI: %{y:,}<extra></extra>"
                ),
                row=1, col=1
            )

        # Puts
        puts = oi_df[oi_df['option_type'] == 'PUT'].sort_values('strike')
        if not puts.empty:
            fig.add_trace(
                go.Bar(
                    x=puts['strike'],
                    y=puts['open_interest'],
                    name='PUT OI',
                    marker_color='red',
                    hovertemplate="Strike: %{x}<br>OI: %{y:,}<extra></extra>"
                ),
                row=1, col=2
            )

        # Add spot lines
        fig.add_vline(x=spot, line_color="white", line_dash="dot", row=1, col=1)
        fig.add_vline(x=spot, line_color="white", line_dash="dot", row=1, col=2)

        fig.update_xaxes(title_text="Strike", row=1, col=1)
        fig.update_xaxes(title_text="Strike", row=1, col=2)
        fig.update_yaxes(title_text="Open Interest", row=1, col=1)
        fig.update_yaxes(title_text="Open Interest", row=1, col=2)

        fig.update_layout(
            template="plotly_dark",
            height=400,
            showlegend=False,
            hovermode='x unified'
        )

        with tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode='w', encoding='utf-8') as f:
            fig.write_html(f, include_plotlyjs='inline', full_html=True)
            html_path = f.name

        self.oi_chart_view.load(QUrl.fromLocalFile(html_path))

    def show_no_oi_message(self):
        """Display message when no OI available"""
        fig = go.Figure()

        fig.add_annotation(
            text="Aucun OI disponible<br><br>Fetch OI dans 'OI Manager' pour afficher ce graphique",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16, color="#FFA500")
        )

        fig.update_layout(
            template="plotly_dark",
            height=400,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )

        with tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode='w', encoding='utf-8') as f:
            fig.write_html(f, include_plotlyjs='inline', full_html=True)
            html_path = f.name

        self.oi_chart_view.load(QUrl.fromLocalFile(html_path))