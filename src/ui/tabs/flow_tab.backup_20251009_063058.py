"""
Flow Tab - Options flow analysis (migrated from Streamlit)
"""
import os
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QComboBox, QPushButton, QGroupBox, QScrollArea)
from PyQt6.QtCore import Qt, QTimer, QUrl
from PyQt6.QtGui import QFont
import tempfile

SAFE_MODE = os.getenv("SAFE_MODE", "0") == "1"
if not SAFE_MODE:
    try:
        from PyQt6.QtWebEngineWidgets import QWebEngineView
    except Exception:
        QWebEngineView = None
else:
    QWebEngineView = None

from src.core.options_flow_analyzer import (
    load_comprehensive_data,
    compute_comprehensive_metrics,
    build_enhanced_strike_metrics,
    get_available_expiries
)

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from src.config import OPTIONFLOW_DB


class FlowTab(QWidget):
    def __init__(self):
        super().__init__()
        self.db_path = OPTIONFLOW_DB
        self.current_expiry = "ALL"
        self.setup_ui()

        # Auto-refresh timer (optional)
        self.timer = QTimer()
        self.timer.timeout.connect(self.refresh_data)
        # Start disabled by default
        # self.timer.start(30000)  # 30 seconds

    def setup_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # Controls bar
        controls_layout = QHBoxLayout()

        # Expiry selector
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

        # Refresh button
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

        main_layout.addLayout(controls_layout)

        # Metrics cards
        # Metrics row avec espacement amélioré
        metrics_layout = QHBoxLayout()
        metrics_layout.setSpacing(15)
        metrics_layout.setContentsMargins(10, 10, 10, 10)

        # Call metrics
        self.call_buys_card = self.create_metric_card("CALL Buys", "0", "#4CAF50")
        self.call_sells_card = self.create_metric_card("CALL Sells", "0", "#f44336")
        self.call_net_card = self.create_metric_card("CALL Net", "0", "#2196F3")

        metrics_layout.addWidget(self.call_buys_card)
        metrics_layout.addWidget(self.call_sells_card)
        metrics_layout.addWidget(self.call_net_card)

        # Put metrics
        self.put_buys_card = self.create_metric_card("PUT Buys", "0", "#4CAF50")
        self.put_sells_card = self.create_metric_card("PUT Sells", "0", "#f44336")
        self.put_net_card = self.create_metric_card("PUT Net", "0", "#2196F3")

        metrics_layout.addWidget(self.put_buys_card)
        metrics_layout.addWidget(self.put_sells_card)
        metrics_layout.addWidget(self.put_net_card)

        # Add stretch to center cards if window is wider
        metrics_layout.addStretch()

        main_layout.addLayout(metrics_layout)


        # Charts area
        if QWebEngineView is not None:
            self.heatmap_view = QWebEngineView()
            self.heatmap_view.setMinimumHeight(400)
            main_layout.addWidget(QLabel("Flow Heatmap:"))
            main_layout.addWidget(self.heatmap_view)

            self.strike_view = QWebEngineView()
            self.strike_view.setMinimumHeight(400)
            main_layout.addWidget(QLabel("Strike Analysis:"))
            main_layout.addWidget(self.strike_view)
        else:
            self.heatmap_view = None
            self.strike_view = None
            fallback = QLabel("SAFE MODE: Charts disabled")
            fallback.setAlignment(Qt.AlignmentFlag.AlignCenter)
            fallback.setStyleSheet("color: red; padding: 50px;")
            main_layout.addWidget(fallback)

        self.setLayout(main_layout)

        # Initial load
        self.refresh_data()

    def create_metric_card(self, title: str, value: str, color: str) -> QGroupBox:
        """Create a metric display card with improved sizing"""
        card = QGroupBox(title)
        card.setStyleSheet(f"""
            QGroupBox {{
                font-weight: bold;
                font-size: 13px;
                color: white;
                border: 2px solid {color};
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 15px;
                background: #2d2d2d;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 5px 10px;
                background: {color};
                border-radius: 4px;
            }}
        """)

        card_layout = QVBoxLayout()
        card_layout.setContentsMargins(10, 20, 10, 15)
        card_layout.setSpacing(8)

        # Value label
        value_label = QLabel(value)
        value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        value_label.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        value_label.setStyleSheet(f"color: {color}; padding: 10px;")
        value_label.setObjectName("value_label")
        value_label.setWordWrap(True)

        card_layout.addWidget(value_label)
        card.setLayout(card_layout)

        # Increased minimum size
        card.setMinimumHeight(120)
        card.setMinimumWidth(180)

        return card

    def load_expiries(self):
        """Load available expiries from database"""
        expiries = get_available_expiries(self.db_path)

        self.expiry_combo.clear()
        self.expiry_combo.addItem("ALL", "ALL")

        for exp in expiries:
            self.expiry_combo.addItem(exp['label'], exp['value'])

        if expiries:
            self.expiry_combo.setCurrentIndex(1)  # Select first real expiry

    def on_expiry_changed(self, text):
        """Handle expiry selection change"""
        idx = self.expiry_combo.currentIndex()
        self.current_expiry = self.expiry_combo.itemData(idx)
        self.refresh_data()

    def refresh_data(self):
        """Load data and update all displays"""
        # Load data
        df = load_comprehensive_data(
            db_path=self.db_path,
            selected_expiry=self.current_expiry,
            sample_size=100000,
            min_volume_filter=0,
            confidence_threshold=0.6
        )

        print(f"[DEBUG] Loaded {len(df)} rows for expiry {self.current_expiry}")  # ADD

        if df.empty:
            print("[DEBUG] DataFrame is empty!")  # ADD
            self.update_metrics({})
            return

        # Compute metrics
        metrics = compute_comprehensive_metrics(df)
        strike_metrics = build_enhanced_strike_metrics(df)

        print(f"[DEBUG] Metrics: {metrics}")  # ADD
        print(f"[DEBUG] Strike metrics rows: {len(strike_metrics)}")  # ADD

        # Update displays
        self.update_metrics(metrics)

        if self.heatmap_view is not None:
            print("[DEBUG] Updating heatmap...")  # ADD
            self.update_heatmap(df)

        if self.strike_view is not None:
            print("[DEBUG] Updating strike chart...")  # ADD
            self.update_strike_chart(strike_metrics)

    def update_metrics(self, metrics):
        """Update metric cards"""
        self.update_card_value(self.call_buys_card,
                               f"{metrics.get('call_buys', 0):,}\n{metrics.get('agg_call_buys', 0):,} agg")
        self.update_card_value(self.call_sells_card,
                               f"{metrics.get('call_sells', 0):,}\n{metrics.get('agg_call_sells', 0):,} agg")
        self.update_card_value(self.call_net_card,
                               f"{metrics.get('call_net_flow', 0):+,}\n{metrics.get('agg_call_net', 0):+,} agg")
        self.update_card_value(self.put_buys_card,
                               f"{metrics.get('put_buys', 0):,}\n{metrics.get('agg_put_buys', 0):,} agg")
        self.update_card_value(self.put_sells_card,
                               f"{metrics.get('put_sells', 0):,}\n{metrics.get('agg_put_sells', 0):,} agg")
        self.update_card_value(self.put_net_card,
                               f"{metrics.get('put_net_flow', 0):+,}\n{metrics.get('agg_put_net', 0):+,} agg")

    def update_card_value(self, card, value):
        label = card.findChild(QLabel, "value_label")
        if label:
            label.setText(value)

    def update_heatmap(self, df):
        """Create and display flow heatmap"""
        if df.empty or 'minute' not in df.columns or df['minute'].isna().all():
            return

        heat_data = (
            df.groupby(['minute', 'strike', 'option_type'])
            .agg({'aggressive_net_volume': 'sum'})
            .reset_index()
        )

        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("CALL Aggressive Net Flow", "PUT Aggressive Net Flow"),
            shared_xaxes=True,
            vertical_spacing=0.1
        )

        for i, option_type in enumerate(['CALL', 'PUT'], 1):
            data = heat_data[heat_data['option_type'] == option_type]
            if not data.empty:
                pivot_data = data.pivot_table(
                    index='strike', columns='minute',
                    values='aggressive_net_volume', fill_value=0
                )
                fig.add_trace(
                    go.Heatmap(
                        z=pivot_data.values,
                        x=pivot_data.columns.astype(str),
                        y=pivot_data.index.astype(str),
                        colorscale='RdBu',
                        zmid=0,
                        name=f'{option_type} Flow',
                        showscale=True
                    ),
                    row=i, col=1
                )

        fig.update_layout(
            title="Flow Heatmap (Green=Buy, Red=Sell)",
            height=600,
            template="plotly_dark"
        )

        # CHANGEMENT : Utiliser encoding='utf-8'
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode='w', encoding='utf-8') as f:
            fig.write_html(f, include_plotlyjs='inline', full_html=True)
            html_path = f.name

        self.heatmap_view.load(QUrl.fromLocalFile(html_path))

    def update_strike_chart(self, strike_metrics):
        """Create and display strike analysis chart"""
        if strike_metrics.empty:
            return

        top_strikes = strike_metrics.nlargest(15, 'total_aggressive').sort_values('strike')

        fig = go.Figure()

        for option_type in ['CALL', 'PUT']:
            data = top_strikes[top_strikes['option_type'] == option_type]
            if data.empty:
                continue
            color = 'green' if option_type == 'CALL' else 'red'

            fig.add_trace(go.Bar(
                x=data['strike'],
                y=data['aggressive_buy_volume'],
                name=f'{option_type} Buys',
                marker_color=color,
                opacity=0.7
            ))
            fig.add_trace(go.Bar(
                x=data['strike'],
                y=-data['aggressive_sell_volume'],
                name=f'{option_type} Sells',
                marker_color=color,
                opacity=0.5
            ))

        fig.update_layout(
            title="Strike Analysis - Top 15 Most Active",
            barmode='relative',
            height=500,
            template="plotly_dark",
            xaxis_title="Strike",
            yaxis_title="Volume"
        )

        # CHANGEMENT : Utiliser encoding='utf-8'
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode='w', encoding='utf-8') as f:
            fig.write_html(f, include_plotlyjs='inline', full_html=True)
            html_path = f.name

        self.strike_view.load(QUrl.fromLocalFile(html_path))