"""
IV Skew Analysis Tab - Analyze implied volatility patterns
"""
import sqlite3
from datetime import datetime
from typing import List, Dict, Tuple, Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tempfile

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QComboBox, QGroupBox, QCheckBox
)
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtGui import QFont

from src.config import OPTIONFLOW_DB


class IVSkewTab(QWidget):
    def __init__(self):
        super().__init__()
        self.db_path = OPTIONFLOW_DB
        self.current_expiry = None
        self.selected_snapshot1 = None
        self.selected_snapshot2 = None

        self.setup_ui()
        self.load_expiries()

    def setup_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(15, 15, 15, 15)

        # Title
        title = QLabel("IV Skew Analysis")
        title.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        title.setStyleSheet("color: white; padding: 10px;")
        main_layout.addWidget(title)

        subtitle = QLabel("Analyze implied volatility patterns and evolution")
        subtitle.setStyleSheet("color: #aaa; font-size: 12px; padding: 5px;")
        main_layout.addWidget(subtitle)

        # Controls
        controls_group = QGroupBox("Controls")
        controls_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                color: white;
                border: 2px solid #2196F3;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
        """)
        controls_layout = QVBoxLayout()

        # Expiry selection
        expiry_layout = QHBoxLayout()
        expiry_layout.addWidget(QLabel("Expiry:"))

        self.expiry_combo = QComboBox()
        self.expiry_combo.setMinimumWidth(300)
        self.expiry_combo.setStyleSheet("""
            QComboBox {
                background: #2d2d2d;
                color: white;
                border: 1px solid #444;
                padding: 5px 10px;
                border-radius: 3px;
            }
        """)
        self.expiry_combo.currentIndexChanged.connect(self.on_expiry_changed)
        expiry_layout.addWidget(self.expiry_combo)

        refresh_btn = QPushButton("🔄 Refresh")
        refresh_btn.setStyleSheet("""
            QPushButton {
                background: #2196F3;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 3px;
                font-weight: bold;
            }
            QPushButton:hover { background: #1976D2; }
        """)
        refresh_btn.clicked.connect(self.load_expiries)
        expiry_layout.addWidget(refresh_btn)
        expiry_layout.addStretch()

        controls_layout.addLayout(expiry_layout)

        # Snapshot comparison
        snapshot_layout = QHBoxLayout()

        snapshot_layout.addWidget(QLabel("Compare Snapshots:"))

        self.snapshot1_combo = QComboBox()
        self.snapshot1_combo.setMinimumWidth(200)
        self.snapshot1_combo.setStyleSheet("""
            QComboBox {
                background: #2d2d2d;
                color: white;
                border: 1px solid #444;
                padding: 5px 10px;
                border-radius: 3px;
            }
        """)
        snapshot_layout.addWidget(QLabel("Snapshot 1:"))
        snapshot_layout.addWidget(self.snapshot1_combo)

        self.snapshot2_combo = QComboBox()
        self.snapshot2_combo.setMinimumWidth(200)
        self.snapshot2_combo.setStyleSheet("""
            QComboBox {
                background: #2d2d2d;
                color: white;
                border: 1px solid #444;
                padding: 5px 10px;
                border-radius: 3px;
            }
        """)
        snapshot_layout.addWidget(QLabel("Snapshot 2:"))
        snapshot_layout.addWidget(self.snapshot2_combo)

        compare_btn = QPushButton("Compare IV Evolution")
        compare_btn.setStyleSheet("""
            QPushButton {
                background: #9C27B0;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 3px;
                font-weight: bold;
            }
            QPushButton:hover { background: #7B1FA2; }
        """)
        compare_btn.clicked.connect(self.show_iv_comparison)
        snapshot_layout.addWidget(compare_btn)

        snapshot_layout.addStretch()
        controls_layout.addLayout(snapshot_layout)

        # Chart options
        options_layout = QHBoxLayout()

        self.show_calls_check = QCheckBox("Show Calls")
        self.show_calls_check.setChecked(True)
        self.show_calls_check.setStyleSheet("color: white;")
        self.show_calls_check.stateChanged.connect(self.update_all_charts)
        options_layout.addWidget(self.show_calls_check)

        self.show_puts_check = QCheckBox("Show Puts")
        self.show_puts_check.setChecked(True)
        self.show_puts_check.setStyleSheet("color: white;")
        self.show_puts_check.stateChanged.connect(self.update_all_charts)
        options_layout.addWidget(self.show_puts_check)

        options_layout.addStretch()
        controls_layout.addLayout(options_layout)

        controls_group.setLayout(controls_layout)
        main_layout.addWidget(controls_group)

        # Status label
        self.status_label = QLabel("Select an expiry to view IV analysis")
        self.status_label.setStyleSheet("""
            color: #aaa;
            font-size: 11px;
            padding: 10px;
            background: #2d2d2d;
            border-radius: 3px;
            font-family: Consolas;
        """)
        main_layout.addWidget(self.status_label)

        # Charts
        self.skew_view = QWebEngineView()
        self.skew_view.setMinimumHeight(400)
        main_layout.addWidget(QLabel("IV Skew (by Strike):"))
        main_layout.addWidget(self.skew_view)

        self.evolution_view = QWebEngineView()
        self.evolution_view.setMinimumHeight(400)
        main_layout.addWidget(QLabel("IV Evolution (Comparison):"))
        main_layout.addWidget(self.evolution_view)

        main_layout.addStretch()
        self.setLayout(main_layout)

    def load_expiries(self):
        """Load expiries that have IV data"""
        try:
            conn = sqlite3.connect(self.db_path)

            query = """
                SELECT DISTINCT expiry, COUNT(*) as count
                FROM oi_snapshots
                WHERE iv IS NOT NULL
                GROUP BY expiry
                ORDER BY expiry DESC
            """

            df = pd.read_sql(query, conn)
            conn.close()

            self.expiry_combo.clear()

            if df.empty:
                self.expiry_combo.addItem("No IV data available", None)
                self.status_label.setText("❌ No IV data found. Fetch OI with IV first.")
                self.status_label.setStyleSheet("""
                    color: #f44336;
                    font-size: 11px;
                    padding: 10px;
                    background: #2d2d2d;
                    border-radius: 3px;
                    font-family: Consolas;
                """)
                return

            today = datetime.now().date()

            for _, row in df.iterrows():
                exp_str = row['expiry']
                count = row['count']

                try:
                    exp_date = datetime.strptime(exp_str, '%Y%m%d').date()
                    dte = (exp_date - today).days

                    if dte == 0:
                        dte_label = "0DTE"
                    elif dte < 0:
                        dte_label = "Expired"
                    else:
                        dte_label = f"{dte}DTE"

                    label = f"{exp_date} ({dte_label}) - {count} strikes"
                    self.expiry_combo.addItem(label, exp_str)

                except Exception:
                    self.expiry_combo.addItem(f"{exp_str} - {count} strikes", exp_str)

            if not df.empty:
                self.expiry_combo.setCurrentIndex(0)

        except Exception as e:
            print(f"Error loading expiries: {e}")

    def on_expiry_changed(self, index):
        """Handle expiry selection"""
        expiry = self.expiry_combo.currentData()

        if not expiry:
            return

        self.current_expiry = expiry

        # Load available snapshots
        self.load_snapshots()

        # Update charts
        self.update_all_charts()

    def load_snapshots(self):
        """Load available snapshots for selected expiry"""
        if not self.current_expiry:
            return

        try:
            conn = sqlite3.connect(self.db_path)

            query = """
                SELECT DISTINCT ts
                FROM oi_snapshots
                WHERE expiry = ? AND iv IS NOT NULL
                ORDER BY ts DESC
            """

            df = pd.read_sql(query, conn, params=[self.current_expiry])
            conn.close()

            self.snapshot1_combo.clear()
            self.snapshot2_combo.clear()

            if df.empty:
                self.snapshot1_combo.addItem("No snapshots", None)
                self.snapshot2_combo.addItem("No snapshots", None)
                return

            for ts in df['ts']:
                self.snapshot1_combo.addItem(ts, ts)
                self.snapshot2_combo.addItem(ts, ts)

            # Auto-select most recent and previous
            if len(df) >= 2:
                self.snapshot1_combo.setCurrentIndex(1)  # Previous
                self.snapshot2_combo.setCurrentIndex(0)  # Most recent

        except Exception as e:
            print(f"Error loading snapshots: {e}")

    def get_iv_data(self, expiry: str, snapshot_ts: Optional[str] = None) -> pd.DataFrame:
        """Get IV data for an expiry"""
        try:
            conn = sqlite3.connect(self.db_path)

            if snapshot_ts:
                query = """
                    SELECT strike, right, iv, delta, gamma, vega, open_interest, ts
                    FROM oi_snapshots
                    WHERE expiry = ? AND ts = ? AND iv IS NOT NULL
                    ORDER BY strike
                """
                df = pd.read_sql(query, conn, params=[expiry, snapshot_ts])
            else:
                # Get latest snapshot
                query = """
                    WITH latest AS (
                        SELECT MAX(ts) as max_ts
                        FROM oi_snapshots
                        WHERE expiry = ? AND iv IS NOT NULL
                    )
                    SELECT strike, right, iv, delta, gamma, vega, open_interest, ts
                    FROM oi_snapshots
                    WHERE expiry = ? AND ts = (SELECT max_ts FROM latest) AND iv IS NOT NULL
                    ORDER BY strike
                """
                df = pd.read_sql(query, conn, params=[expiry, expiry])

            conn.close()

            if not df.empty:
                df['option_type'] = df['right'].map({'C': 'CALL', 'P': 'PUT'})

            return df

        except Exception as e:
            print(f"Error getting IV data: {e}")
            return pd.DataFrame()

    def update_all_charts(self):
        """Update all charts"""
        if not self.current_expiry:
            return

        self.show_iv_skew()
        self.show_iv_comparison()

    def show_iv_skew(self):
        """Show IV skew chart"""
        try:
            df = self.get_iv_data(self.current_expiry)

            if df.empty:
                self.status_label.setText(f"❌ No IV data for {self.current_expiry}")
                return

            self.status_label.setText(
                f"✓ Loaded {len(df)} strikes | "
                f"Last update: {df['ts'].iloc[0]}"
            )
            self.status_label.setStyleSheet("""
                color: #4CAF50;
                font-size: 11px;
                padding: 10px;
                background: #2d2d2d;
                border-radius: 3px;
                font-family: Consolas;
            """)

            # Create figure
            fig = go.Figure()

            show_calls = self.show_calls_check.isChecked()
            show_puts = self.show_puts_check.isChecked()

            if show_calls:
                calls = df[df['option_type'] == 'CALL']
                fig.add_trace(go.Scatter(
                    x=calls['strike'],
                    y=calls['iv'] * 100,  # Convert to percentage
                    mode='lines+markers',
                    name='Calls IV',
                    line=dict(color='green', width=2),
                    marker=dict(size=8)
                ))

            if show_puts:
                puts = df[df['option_type'] == 'PUT']
                fig.add_trace(go.Scatter(
                    x=puts['strike'],
                    y=puts['iv'] * 100,
                    mode='lines+markers',
                    name='Puts IV',
                    line=dict(color='red', width=2),
                    marker=dict(size=8)
                ))

            fig.update_layout(
                title=f"IV Skew - {self.current_expiry}",
                xaxis_title="Strike",
                yaxis_title="Implied Volatility (%)",
                template="plotly_dark",
                height=400,
                hovermode='x unified',
                showlegend=True
            )

            # Save and display
            with tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode='w', encoding='utf-8') as f:
                fig.write_html(f, include_plotlyjs='inline', full_html=True)
                html_path = f.name

            self.skew_view.setUrl(QUrl.fromLocalFile(html_path))

        except Exception as e:
            print(f"Error creating IV skew: {e}")
            import traceback
            traceback.print_exc()

    def show_iv_comparison(self):
        """Show IV evolution comparison"""
        try:
            snapshot1_ts = self.snapshot1_combo.currentData()
            snapshot2_ts = self.snapshot2_combo.currentData()

            if not snapshot1_ts or not snapshot2_ts:
                return

            df1 = self.get_iv_data(self.current_expiry, snapshot1_ts)
            df2 = self.get_iv_data(self.current_expiry, snapshot2_ts)

            if df1.empty or df2.empty:
                return

            # Create subplots
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=(
                    f"IV Comparison: {snapshot1_ts} vs {snapshot2_ts}",
                    "IV Change (Δ)"
                ),
                vertical_spacing=0.12
            )

            show_calls = self.show_calls_check.isChecked()
            show_puts = self.show_puts_check.isChecked()

            # Plot 1: IV comparison
            if show_calls:
                calls1 = df1[df1['option_type'] == 'CALL']
                calls2 = df2[df2['option_type'] == 'CALL']

                fig.add_trace(go.Scatter(
                    x=calls1['strike'],
                    y=calls1['iv'] * 100,
                    mode='lines',
                    name=f'Calls {snapshot1_ts}',
                    line=dict(color='lightgreen', width=2, dash='dash')
                ), row=1, col=1)

                fig.add_trace(go.Scatter(
                    x=calls2['strike'],
                    y=calls2['iv'] * 100,
                    mode='lines+markers',
                    name=f'Calls {snapshot2_ts}',
                    line=dict(color='green', width=2)
                ), row=1, col=1)

            if show_puts:
                puts1 = df1[df1['option_type'] == 'PUT']
                puts2 = df2[df2['option_type'] == 'PUT']

                fig.add_trace(go.Scatter(
                    x=puts1['strike'],
                    y=puts1['iv'] * 100,
                    mode='lines',
                    name=f'Puts {snapshot1_ts}',
                    line=dict(color='lightcoral', width=2, dash='dash')
                ), row=1, col=1)

                fig.add_trace(go.Scatter(
                    x=puts2['strike'],
                    y=puts2['iv'] * 100,
                    mode='lines+markers',
                    name=f'Puts {snapshot2_ts}',
                    line=dict(color='red', width=2)
                ), row=1, col=1)

            # Plot 2: IV change
            # Merge dataframes
            if show_calls:
                calls_merged = pd.merge(
                    calls1[['strike', 'iv']],
                    calls2[['strike', 'iv']],
                    on='strike',
                    suffixes=('_old', '_new')
                )
                calls_merged['iv_change'] = (calls_merged['iv_new'] - calls_merged['iv_old']) * 100

                fig.add_trace(go.Bar(
                    x=calls_merged['strike'],
                    y=calls_merged['iv_change'],
                    name='Calls Δ IV',
                    marker_color='green',
                    opacity=0.7
                ), row=2, col=1)

            if show_puts:
                puts_merged = pd.merge(
                    puts1[['strike', 'iv']],
                    puts2[['strike', 'iv']],
                    on='strike',
                    suffixes=('_old', '_new')
                )
                puts_merged['iv_change'] = (puts_merged['iv_new'] - puts_merged['iv_old']) * 100

                fig.add_trace(go.Bar(
                    x=puts_merged['strike'],
                    y=puts_merged['iv_change'],
                    name='Puts Δ IV',
                    marker_color='red',
                    opacity=0.7
                ), row=2, col=1)

            # Update axes
            fig.update_xaxes(title_text="Strike", row=1, col=1)
            fig.update_xaxes(title_text="Strike", row=2, col=1)
            fig.update_yaxes(title_text="IV (%)", row=1, col=1)
            fig.update_yaxes(title_text="Δ IV (%)", row=2, col=1)

            fig.update_layout(
                template="plotly_dark",
                height=800,
                showlegend=True,
                hovermode='x unified'
            )

            # Save and display
            with tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode='w', encoding='utf-8') as f:
                fig.write_html(f, include_plotlyjs='inline', full_html=True)
                html_path = f.name

            self.evolution_view.setUrl(QUrl.fromLocalFile(html_path))

        except Exception as e:
            print(f"Error creating IV comparison: {e}")
            import traceback
            traceback.print_exc()