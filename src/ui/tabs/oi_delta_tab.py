"""
OI Delta Analysis Tab - Track open interest changes between snapshots
"""
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QGroupBox, QComboBox, QTableWidget,
                             QTableWidgetItem, QHeaderView, QAbstractItemView)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QColor

from src.core.oi_fetcher import (
    get_oi_delta_between_snapshots,
    get_available_oi_snapshots
)
from src.core.options_flow_analyzer import get_available_expiries
from src.config import OPTIONFLOW_DB



class OIDeltaTab(QWidget):
    def __init__(self):
        super().__init__()
        self.db_path = OPTIONFLOW_DB
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(15, 15, 15, 15)

        # Title
        title = QLabel("OI Delta Analysis")
        title.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        title.setStyleSheet("color: white; padding: 10px;")
        layout.addWidget(title)

        subtitle = QLabel("Track Open Interest changes - Detect institutional positioning")
        subtitle.setStyleSheet("color: #aaa; font-size: 12px; padding: 5px;")
        layout.addWidget(subtitle)

        # Selection controls
        controls_group = QGroupBox("Snapshot Selection")
        controls_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                color: white;
                border: 2px solid #444;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
        """)
        controls_layout = QVBoxLayout()

        # Expiry selection
        exp_layout = QHBoxLayout()
        exp_layout.addWidget(QLabel("Expiry:"))

        self.expiry_combo = QComboBox()
        self.expiry_combo.setMinimumWidth(300)
        self.expiry_combo.currentIndexChanged.connect(self.on_expiry_changed)
        exp_layout.addWidget(self.expiry_combo)
        exp_layout.addStretch()

        controls_layout.addLayout(exp_layout)

        # Snapshot selection
        snap_layout = QHBoxLayout()

        snap_layout.addWidget(QLabel("Compare:"))
        self.snap1_combo = QComboBox()
        self.snap1_combo.setMinimumWidth(200)
        snap_layout.addWidget(self.snap1_combo)

        snap_layout.addWidget(QLabel("vs"))
        self.snap2_combo = QComboBox()
        self.snap2_combo.setMinimumWidth(200)
        snap_layout.addWidget(self.snap2_combo)

        snap_layout.addStretch()
        controls_layout.addLayout(snap_layout)

        # Analyze button
        self.analyze_btn = QPushButton("🔍 Analyze Delta OI")
        self.analyze_btn.setStyleSheet("""
            QPushButton {
                background: #4CAF50;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover { background: #45a049; }
        """)
        self.analyze_btn.clicked.connect(self.analyze_delta)
        controls_layout.addWidget(self.analyze_btn)

        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)

        # Summary stats
        self.summary_label = QLabel("Select expiry and snapshots to analyze")
        self.summary_label.setStyleSheet("""
            color: #ddd;
            font-size: 12px;
            padding: 15px;
            background: #2d2d2d;
            border-radius: 5px;
            font-family: Consolas;
        """)
        layout.addWidget(self.summary_label)

        # Results table
        results_group = QGroupBox("OI Changes (Top Movers)")
        results_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                color: white;
                border: 2px solid #444;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
        """)
        results_layout = QVBoxLayout()

        self.results_table = QTableWidget()
        self.results_table.setColumnCount(7)
        self.results_table.setHorizontalHeaderLabels([
            "Type", "Strike", "OI Old", "OI New", "Delta OI", "Delta %", "Signal"
        ])
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.results_table.setAlternatingRowColors(True)
        self.results_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.results_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.results_table.setStyleSheet("""
            QTableWidget {
                background: #2d2d2d;
                color: white;
                gridline-color: #444;
                font-size: 11px;
            }
            QHeaderView::section {
                background: #3d3d3d;
                color: white;
                padding: 5px;
                border: 1px solid #444;
                font-weight: bold;
            }
            QTableWidget::item:selected {
                background: #4CAF50;
            }
        """)

        results_layout.addWidget(self.results_table)
        results_group.setLayout(results_layout)
        layout.addWidget(results_group)

        self.setLayout(layout)

        # Load initial data
        self.load_expiries()

    def load_expiries(self):
        """Load available expiries"""
        from src.core.options_flow_analyzer import get_available_expiries

        expiries = get_available_expiries(self.db_path)
        self.expiry_combo.clear()

        for exp in expiries:
            self.expiry_combo.addItem(exp['label'], exp['value'])

    def on_expiry_changed(self):
        """Load snapshots when expiry changes"""
        expiry = self.expiry_combo.currentData()
        if not expiry:
            return

        snapshots = get_available_oi_snapshots(self.db_path, expiry)

        self.snap1_combo.clear()
        self.snap2_combo.clear()

        if not snapshots:
            self.summary_label.setText(f"No OI snapshots available for {expiry}")
            return

        for snap in snapshots:
            self.snap1_combo.addItem(snap)
            self.snap2_combo.addItem(snap)

        # Auto-select last two if available
        if len(snapshots) >= 2:
            self.snap1_combo.setCurrentIndex(1)  # Older
            self.snap2_combo.setCurrentIndex(0)  # Newer

        self.summary_label.setText(f"Found {len(snapshots)} snapshots for {expiry}")

    def analyze_delta(self):
        """Analyze OI delta between selected snapshots"""
        expiry = self.expiry_combo.currentData()
        snap1 = self.snap1_combo.currentText()
        snap2 = self.snap2_combo.currentText()

        if not all([expiry, snap1, snap2]):
            self.summary_label.setText("❌ Select expiry and both snapshots")
            return

        if snap1 == snap2:
            self.summary_label.setText("❌ Select two different snapshots")
            return

        # Get delta
        delta_df = get_oi_delta_between_snapshots(self.db_path, expiry, snap1, snap2)

        if delta_df.empty:
            self.summary_label.setText("❌ No data found")
            return

        # Calculate summary
        total_delta = delta_df['delta_oi'].sum()
        call_delta = delta_df[delta_df['option_type'] == 'CALL']['delta_oi'].sum()
        put_delta = delta_df[delta_df['option_type'] == 'PUT']['delta_oi'].sum()

        top_increases = delta_df[delta_df['delta_oi'] > 0].head(10)
        top_decreases = delta_df[delta_df['delta_oi'] < 0].head(10)

        summary = (
            f"📊 OI Delta Summary:\n"
            f"   Total OI change: {total_delta:+,} contracts\n"
            f"   Calls: {call_delta:+,} | Puts: {put_delta:+,}\n"
            f"   Top increases: {len(top_increases)} strikes\n"
            f"   Top decreases: {len(top_decreases)} strikes"
        )
        self.summary_label.setText(summary)

        # Populate table (top 50 movers)
        display_df = delta_df.head(50)
        self.results_table.setRowCount(len(display_df))

        for row_idx, (_, row) in enumerate(display_df.iterrows()):
            # Type
            type_item = QTableWidgetItem(row['option_type'])
            type_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.results_table.setItem(row_idx, 0, type_item)

            # Strike
            strike_item = QTableWidgetItem(f"{row['strike']:.0f}")
            strike_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.results_table.setItem(row_idx, 1, strike_item)

            # OI Old
            oi_old_item = QTableWidgetItem(f"{int(row['oi_old']):,}")
            oi_old_item.setTextAlignment(Qt.AlignmentFlag.AlignRight)
            self.results_table.setItem(row_idx, 2, oi_old_item)

            # OI New
            oi_new_item = QTableWidgetItem(f"{int(row['oi_new']):,}")
            oi_new_item.setTextAlignment(Qt.AlignmentFlag.AlignRight)
            self.results_table.setItem(row_idx, 3, oi_new_item)

            # Delta OI
            delta_oi = int(row['delta_oi'])
            delta_item = QTableWidgetItem(f"{delta_oi:+,}")
            delta_item.setTextAlignment(Qt.AlignmentFlag.AlignRight)

            # Color coding
            if delta_oi > 0:
                delta_item.setForeground(QColor("#4CAF50"))  # Green
            elif delta_oi < 0:
                delta_item.setForeground(QColor("#f44336"))  # Red

            self.results_table.setItem(row_idx, 4, delta_item)

            # Delta %
            delta_pct = row['delta_oi_pct']
            pct_item = QTableWidgetItem(f"{delta_pct:+.1f}%")
            pct_item.setTextAlignment(Qt.AlignmentFlag.AlignRight)
            self.results_table.setItem(row_idx, 5, pct_item)

            # Signal interpretation
            signal = self._interpret_signal(row)
            signal_item = QTableWidgetItem(signal)
            signal_item.setTextAlignment(Qt.AlignmentFlag.AlignLeft)
            self.results_table.setItem(row_idx, 6, signal_item)

    def _interpret_signal(self, row) -> str:
        """Interpret what the OI change means"""
        delta = int(row['delta_oi'])
        option_type = row['option_type']

        if abs(delta) < 50:
            return "Neutral"

        if delta > 0:
            if option_type == "PUT":
                if delta > 500:
                    return "🛡️ Heavy hedging"
                return "🔒 Protection buying"
            else:  # CALL
                if delta > 500:
                    return "🚀 Bullish positioning"
                return "📈 Call accumulation"
        else:  # delta < 0
            if option_type == "PUT":
                return "✅ Hedge unwinding"
            else:  # CALL
                return "📉 Position closing"