"""
OI Manager Tab - Fetch and manage Open Interest data from IBKR
"""
import pandas as pd  # ✅ AJOUT

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QGroupBox, QComboBox, QTextEdit,
    QLineEdit, QSpinBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QColor  # ✅ AJOUT QColor

from src.core.oi_fetcher import (
    fetch_oi_snapshot_for_expiry,
    get_oi_snapshot_meta,
    ensure_oi_schema
)
from src.config import OPTIONFLOW_DB  # ✅ Utilise config


class OIFetchWorker(QThread):
    """Worker thread for fetching OI (prevents UI freeze)"""
    progress = pyqtSignal(str)
    finished = pyqtSignal(bool, int)  # (success, count)

    def __init__(self, db_path, expiry, host, port):
        super().__init__()
        self.db_path = db_path
        self.expiry = expiry
        self.host = host
        self.port = port

    def run(self):
        try:
            success, count = fetch_oi_snapshot_for_expiry(
                db_path=self.db_path,
                expiry=self.expiry,
                host=self.host,
                port=self.port,
                client_id=21,
                symbol_hint="SPX",
                tclass_hint="SPXW",
                exchange="CBOE",
                pause=0.35,
                timeout_s=8.0,
                try_smart_on_empty=True,
                debug=True,
                progress_callback=self.progress.emit
            )

            self.finished.emit(success, count)

        except Exception as e:
            self.progress.emit(f"Error: {str(e)}")
            self.finished.emit(False, 0)


class OIManagerTab(QWidget):
    def __init__(self):
        super().__init__()
        self.db_path = OPTIONFLOW_DB  # ✅ Utilise config
        self.worker = None

        # Initialize schema
        ensure_oi_schema(self.db_path)

        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(15, 15, 15, 15)

        # Title
        title = QLabel("Open Interest Manager")
        title.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        title.setStyleSheet("color: white; padding: 10px;")
        layout.addWidget(title)

        subtitle = QLabel("Fetch real OI data from IBKR for accurate GEX calculations")
        subtitle.setStyleSheet("color: #aaa; font-size: 12px; padding: 5px;")
        layout.addWidget(subtitle)

        # Connection settings
        conn_group = QGroupBox("IB Connection")
        conn_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                color: white;
                border: 2px solid #444;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
        """)
        conn_layout = QHBoxLayout()

        conn_layout.addWidget(QLabel("Host:"))
        self.host_input = QLineEdit("127.0.0.1")
        self.host_input.setMaximumWidth(150)
        conn_layout.addWidget(self.host_input)

        conn_layout.addWidget(QLabel("Port:"))
        self.port_input = QSpinBox()
        self.port_input.setRange(1000, 9999)
        self.port_input.setValue(7497)
        self.port_input.setMaximumWidth(100)
        conn_layout.addWidget(self.port_input)
        conn_layout.addStretch()

        conn_group.setLayout(conn_layout)
        layout.addWidget(conn_group)

        # Expiry selection
        expiry_group = QGroupBox("Expiry Selection")
        expiry_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                color: white;
                border: 2px solid #444;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
        """)
        expiry_layout = QVBoxLayout()

        select_layout = QHBoxLayout()
        select_layout.addWidget(QLabel("Expiry:"))

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
        select_layout.addWidget(self.expiry_combo)
        select_layout.addStretch()

        expiry_layout.addLayout(select_layout)

        refresh_expiries_btn = QPushButton("🔄 Refresh Expiries")
        refresh_expiries_btn.setStyleSheet("""
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
        refresh_expiries_btn.clicked.connect(self.refresh_expiries)  # ✅ CORRIGÉ
        expiry_layout.addWidget(refresh_expiries_btn)

        # Current OI status
        self.oi_status_label = QLabel("No OI data for this expiry")
        self.oi_status_label.setStyleSheet("""
            color: #aaa;
            font-size: 11px;
            padding: 10px;
            background: #2d2d2d;
            border-radius: 3px;
            font-family: Consolas;
        """)
        expiry_layout.addWidget(self.oi_status_label)

        expiry_group.setLayout(expiry_layout)
        layout.addWidget(expiry_group)

        # Fetch controls
        fetch_group = QGroupBox("Fetch OI Data")
        fetch_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                color: white;
                border: 2px solid #4CAF50;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
        """)
        fetch_layout = QVBoxLayout()

        info_label = QLabel(
            "⚠️ Fetching OI takes 5-15 minutes depending on number of strikes.\n"
            "Make sure TWS/Gateway is running and market data subscriptions are active."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #FFA500; font-size: 11px; padding: 10px;")
        fetch_layout.addWidget(info_label)

        self.fetch_button = QPushButton("Fetch OI for Selected Expiry")
        self.fetch_button.setStyleSheet("""
            QPushButton {
                background: #4CAF50;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 5px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover { background: #45a049; }
            QPushButton:disabled { background: #666; }
        """)
        self.fetch_button.clicked.connect(self.fetch_oi)
        fetch_layout.addWidget(self.fetch_button)

        fetch_group.setLayout(fetch_layout)
        layout.addWidget(fetch_group)

        # Progress log
        log_group = QGroupBox("Progress Log")
        log_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                color: white;
                border: 2px solid #444;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
        """)
        log_layout = QVBoxLayout()

        self.progress_log = QTextEdit()
        self.progress_log.setReadOnly(True)
        self.progress_log.setMaximumHeight(250)
        self.progress_log.setStyleSheet("""
            QTextEdit {
                background: #1e1e1e;
                color: #ddd;
                font-family: Consolas;
                font-size: 11px;
                border: 1px solid #444;
                border-radius: 3px;
            }
        """)
        log_layout.addWidget(self.progress_log)

        log_group.setLayout(log_layout)
        layout.addWidget(log_group)

        layout.addStretch()
        self.setLayout(layout)

        # ✅ Load expiries after UI setup
        self.refresh_expiries()

    def refresh_expiries(self):
        """
        Refresh expiry dropdown from TRADES table
        Shows all expiries with options flow data
        """
        try:
            import sqlite3
            from datetime import datetime

            conn = sqlite3.connect(self.db_path)

            # Read from TRADES not oi_snapshots
            query = """
                SELECT DISTINCT expiry, COUNT(*) as trade_count
                FROM trades
                WHERE expiry IS NOT NULL
                  AND right IN ('C', 'P')
                  AND qty > 0
                GROUP BY expiry
                ORDER BY expiry DESC
                LIMIT 30
            """

            df = pd.read_sql(query, conn)

            if df.empty:
                self.expiry_combo.clear()
                self.expiry_combo.addItem("No expiries available", None)
                conn.close()
                return

            # Clear combo
            self.expiry_combo.clear()

            today = datetime.now().date()

            for _, row in df.iterrows():
                exp_str = row['expiry']
                trade_count = row['trade_count']

                try:
                    # Parse expiry date
                    exp_date = datetime.strptime(exp_str, '%Y%m%d').date()
                    dte = (exp_date - today).days

                    # DTE label
                    if dte == 0:
                        dte_label = "0DTE"
                    elif dte < 0:
                        dte_label = "Expired"
                    else:
                        dte_label = f"{dte}DTE"

                    # Check if OI snapshot exists
                    cursor = conn.cursor()
                    cursor.execute(
                        "SELECT COUNT(*) FROM oi_snapshots WHERE expiry = ?",
                        [exp_str]
                    )
                    oi_count = cursor.fetchone()[0]

                    # Build label
                    if oi_count > 0:
                        label = f"{exp_date} ({dte_label}) - {trade_count:,} trades | ✓ OI ({oi_count})"
                    else:
                        label = f"{exp_date} ({dte_label}) - {trade_count:,} trades | ⊘ No OI"

                    self.expiry_combo.addItem(label, exp_str)

                except Exception as e:
                    # Fallback for malformed dates
                    self.expiry_combo.addItem(f"{exp_str} - {trade_count:,} trades", exp_str)

            conn.close()

            print(f"✓ Refreshed: {len(df)} expiries loaded from trades table")

            # ✅ Apply color styling
            self.update_expiry_combo_style()

            # Auto-select most recent expiry without OI
            for i in range(self.expiry_combo.count()):
                if "⊘ No OI" in self.expiry_combo.itemText(i):
                    self.expiry_combo.setCurrentIndex(i)
                    break

        except Exception as e:
            print(f"❌ Error refreshing expiries: {e}")
            import traceback
            traceback.print_exc()

            self.expiry_combo.clear()
            self.expiry_combo.addItem("Error loading expiries", None)

    def update_expiry_combo_style(self):
        """
        Style the combo to highlight expiries without OI
        """
        for i in range(self.expiry_combo.count()):
            text = self.expiry_combo.itemText(i)
            if "⊘ No OI" in text:
                # Red for missing OI
                self.expiry_combo.setItemData(i, QColor("#f44336"), Qt.ItemDataRole.ForegroundRole)
            elif "✓ OI" in text:
                # Green for existing OI
                self.expiry_combo.setItemData(i, QColor("#4CAF50"), Qt.ItemDataRole.ForegroundRole)

    def on_expiry_changed(self, index):
        """Handle expiry selection change"""
        self.update_oi_status()

    def update_oi_status(self):
        """Update OI status for selected expiry"""
        expiry = self.expiry_combo.currentData()

        if not expiry:
            self.oi_status_label.setText("No expiry selected")
            return

        last_ts, n_total, n_calls, n_puts = get_oi_snapshot_meta(self.db_path, expiry)

        if last_ts is None or n_total == 0:
            self.oi_status_label.setText(
                f"Expiry: {expiry}\n"
                f"Status: No OI data available\n"
                f"Action: Click 'Fetch OI' to download from IBKR"
            )
            self.oi_status_label.setStyleSheet("""
                color: #f44336;
                font-size: 11px;
                padding: 10px;
                background: #2d2d2d;
                border-radius: 3px;
                font-family: Consolas;
            """)
        else:
            self.oi_status_label.setText(
                f"Expiry: {expiry}\n"
                f"Last update: {last_ts}\n"
                f"Total strikes: {n_total} ({n_calls} calls, {n_puts} puts)\n"
                f"Status: ✓ OI data available"
            )
            self.oi_status_label.setStyleSheet("""
                color: #4CAF50;
                font-size: 11px;
                padding: 10px;
                background: #2d2d2d;
                border-radius: 3px;
                font-family: Consolas;
            """)

    def fetch_oi(self):
        """Start OI fetch in background thread"""
        expiry = self.expiry_combo.currentData()

        if not expiry:
            self.progress_log.append("❌ No expiry selected")
            return

        self.fetch_button.setEnabled(False)
        self.progress_log.clear()
        self.progress_log.append(f"Starting OI fetch for expiry: {expiry}")
        self.progress_log.append(f"Host: {self.host_input.text()}, Port: {self.port_input.value()}\n")

        # Start worker
        self.worker = OIFetchWorker(
            db_path=self.db_path,
            expiry=expiry,
            host=self.host_input.text(),
            port=self.port_input.value()
        )
        self.worker.progress.connect(self.on_progress)
        self.worker.finished.connect(self.on_finished)
        self.worker.start()

    def on_progress(self, message):
        """Handle progress updates"""
        self.progress_log.append(message)
        # Auto-scroll
        self.progress_log.verticalScrollBar().setValue(
            self.progress_log.verticalScrollBar().maximum()
        )

    def on_finished(self, success, count):
        """Handle fetch completion"""
        self.fetch_button.setEnabled(True)

        if success:
            self.progress_log.append(f"\n✅ Success! {count} OI values inserted")
            self.refresh_expiries()  # ✅ Refresh to show new OI
            self.update_oi_status()
        else:
            self.progress_log.append("\n❌ Fetch failed or no data inserted")

        self.progress_log.append("\n=== Fetch completed ===")