"""
Flow Tab - Options flow analysis (migrated from Streamlit)
"""
import tempfile
import pandas as pd
import plotly.graph_objects as go
from PyQt6.QtCore import QUrl, Qt, QTimer
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QComboBox, QGroupBox
)
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtGui import QFont

from src.core.options_flow_analyzer import (
    load_comprehensive_data,
    compute_comprehensive_metrics,
    build_enhanced_strike_metrics,
    get_available_expiries
)
from src.config import OPTIONFLOW_DB


class FlowTab(QWidget):
    def __init__(self):
        super().__init__()
        self.db_path = OPTIONFLOW_DB
        self.current_expiry = "ALL"
        self._last_data_hash = None
        self._cached_metrics = None

        self.setup_ui()

        # Auto-refresh timer (disabled by default)
        self.timer = QTimer()
        self.timer.timeout.connect(self.refresh_data)
        # Uncomment to enable 30s auto-refresh:
        # self.timer.start(30000)

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
        refresh_btn = QPushButton("Rafraîchir")
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

        metrics_layout.addStretch()
        main_layout.addLayout(metrics_layout)

        # Charts area
        self.heatmap_view = QWebEngineView()
        self.heatmap_view.setMinimumHeight(400)
        main_layout.addWidget(QLabel("Flow Heatmap:"))
        main_layout.addWidget(self.heatmap_view)

        self.strike_view = QWebEngineView()
        self.strike_view.setMinimumHeight(400)
        main_layout.addWidget(QLabel("Strike Analysis:"))
        main_layout.addWidget(self.strike_view)

        self.setLayout(main_layout)

        # Initial load (delayed to ensure UI is ready)
        QTimer.singleShot(500, self.refresh_data)

    def create_metric_card(self, title: str, value: str, color: str) -> QGroupBox:
        """Create a metric display card"""
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

        value_label = QLabel(value)
        value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        value_label.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        value_label.setStyleSheet(f"color: {color}; padding: 10px;")
        value_label.setObjectName("value_label")
        value_label.setWordWrap(True)

        card_layout.addWidget(value_label)
        card.setLayout(card_layout)

        card.setMinimumHeight(120)
        card.setMinimumWidth(180)

        return card

    def load_expiries(self):
        """Load available expiries from database"""
        try:
            expiries = get_available_expiries(self.db_path)

            self.expiry_combo.clear()
            self.expiry_combo.addItem("ALL", "ALL")

            for exp in expiries:
                self.expiry_combo.addItem(exp['label'], exp['value'])

            if expiries:
                self.expiry_combo.setCurrentIndex(1)
        except Exception as e:
            print(f"[ERROR] Loading expiries: {e}")
            self.expiry_combo.addItem("Error loading expiries", None)

    def on_expiry_changed(self, text):
        """Handle expiry selection change"""
        idx = self.expiry_combo.currentIndex()
        self.current_expiry = self.expiry_combo.itemData(idx)
        if self.current_expiry:
            self.refresh_data()

    def refresh_data(self):
        """Load data and update all displays"""
        try:
            # Load data
            df = load_comprehensive_data(
                db_path=self.db_path,
                selected_expiry=self.current_expiry,
                sample_size=50000,
                min_volume_filter=0,
                confidence_threshold=0.6
            )

            print(f"[DEBUG] Loaded {len(df)} rows for expiry {self.current_expiry}")

            if df.empty:
                print("[DEBUG] DataFrame is empty!")
                self.update_metrics({})
                return

            # Compute metrics
            metrics = compute_comprehensive_metrics(df)
            strike_metrics = build_enhanced_strike_metrics(df)

            print(f"[DEBUG] Metrics: {metrics}")
            print(f"[DEBUG] Strike metrics rows: {len(strike_metrics)}")

            # Update displays
            self.update_metrics(metrics)
            self.update_heatmap(df)
            self.update_strike_chart(strike_metrics)

        except Exception as e:
            print(f"[ERROR] Refresh data: {e}")
            import traceback
            traceback.print_exc()

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
        """Update card value label"""
        label = card.findChild(QLabel, "value_label")
        if label:
            label.setText(value)

    def update_heatmap(self, df):
        """Update flow heatmap"""
        if df.empty:
            print("[DEBUG] DataFrame empty, skipping heatmap")
            return

        try:
            print(f"[DEBUG] Creating heatmap with {len(df)} rows")

            # Create pivot table
            pivot = df.groupby(['market_session', 'money_class']).agg({
                'qty': 'sum',
                'is_buy': 'sum',
                'is_sell': 'sum'
            }).reset_index()

            if pivot.empty:
                print("[DEBUG] Pivot table empty")
                return

            pivot['net_flow'] = pivot['is_buy'] - pivot['is_sell']

            heatmap_data = pivot.pivot(
                index='money_class',
                columns='market_session',
                values='net_flow'
            ).fillna(0)

            print(f"[DEBUG] Heatmap shape: {heatmap_data.shape}")

            # Create figure with inline plotly
            fig = go.Figure(data=go.Heatmap(
                z=heatmap_data.values,
                x=heatmap_data.columns,
                y=heatmap_data.index,
                colorscale='RdYlGn',
                zmid=0,
                text=heatmap_data.values,
                texttemplate='%{text:.0f}',
                textfont={"size": 12},
                colorbar=dict(title="Net Flow")
            ))

            fig.update_layout(
                title="Flow Heatmap - Net Buy/Sell by Session & Moneyness",
                xaxis_title="Market Session",
                yaxis_title="Money Class",
                height=400,
                template="plotly_dark"
            )

            # ✅ FIX: Use 'inline' instead of 'cdn' for guaranteed loading
            with tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode='w', encoding='utf-8') as f:
                fig.write_html(f, include_plotlyjs='inline', full_html=True)
                html_path = f.name

            self.heatmap_view.setUrl(QUrl.fromLocalFile(html_path))
            print(f"[DEBUG] Heatmap loaded: {html_path}")

        except Exception as e:
            print(f"[ERROR] Creating heatmap: {e}")
            import traceback
            traceback.print_exc()

    def update_strike_chart(self, strike_metrics):
        """Create and display strike analysis chart"""
        if strike_metrics.empty:
            print("[DEBUG] Strike metrics empty")
            return

        try:
            print(f"[DEBUG] Creating strike chart with {len(strike_metrics)} strikes")

            top_strikes = strike_metrics.nlargest(15, 'total_qty').sort_values('strike')

            if top_strikes.empty:
                print("[DEBUG] No top strikes")
                return

            fig = go.Figure()

            for option_type in ['CALL', 'PUT']:
                data = top_strikes[top_strikes['option_type'] == option_type]
                if data.empty:
                    continue

                color = 'green' if option_type == 'CALL' else 'red'

                fig.add_trace(go.Bar(
                    x=data['strike'],
                    y=data['buys'],
                    name=f'{option_type} Buys',
                    marker_color=color,
                    opacity=0.7
                ))

                fig.add_trace(go.Bar(
                    x=data['strike'],
                    y=-data['sells'],
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
                yaxis_title="Volume",
                showlegend=True
            )

            # ✅ FIX: Use 'inline' instead of 'cdn'
            with tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode='w', encoding='utf-8') as f:
                fig.write_html(f, include_plotlyjs='inline', full_html=True)
                html_path = f.name

            self.strike_view.setUrl(QUrl.fromLocalFile(html_path))
            print(f"[DEBUG] Strike chart loaded: {html_path}")

        except Exception as e:
            print(f"[ERROR] Creating strike chart: {e}")
            import traceback
            traceback.print_exc()