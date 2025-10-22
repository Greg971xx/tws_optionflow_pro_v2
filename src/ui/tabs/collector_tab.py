"""
Collector Tab - Control options flow data collector
"""
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QGroupBox, QLineEdit, QSpinBox,
                             QTextEdit, QComboBox)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont

from src.core.flow_collector import FlowCollector, CollectorConfig


class CollectorTab(QWidget):
    def __init__(self):
        super().__init__()
        self.collector = None
        self.setup_ui()

        # Status update timer
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_status)
        self.status_timer.start(2000)  # Update every 2 seconds

    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(15, 15, 15, 15)

        # Title
        title = QLabel("Options Flow Data Collector")
        title.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        title.setStyleSheet("color: white; padding: 10px;")
        layout.addWidget(title)

        subtitle = QLabel("Collect real-time SPX options flow data from IBKR")
        subtitle.setStyleSheet("color: #aaa; font-size: 12px; padding: 5px;")
        layout.addWidget(subtitle)

        # Connection Settings
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

        conn_layout.addWidget(QLabel("Client ID:"))
        self.client_id_input = QSpinBox()
        self.client_id_input.setRange(1, 999)
        self.client_id_input.setValue(40)
        self.client_id_input.setMaximumWidth(100)
        conn_layout.addWidget(self.client_id_input)

        conn_layout.addStretch()
        conn_group.setLayout(conn_layout)
        layout.addWidget(conn_group)

        # Collector Settings
        settings_group = QGroupBox("Collector Settings")
        settings_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                color: white;
                border: 2px solid #444;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
        """)
        settings_layout = QVBoxLayout()

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Expiry:"))
        self.expiry_input = QComboBox()
        self.expiry_input.addItems(["0DTE","auto", "20251010", "20251017", "20251024", "20251031"])
        self.expiry_input.setEditable(True)
        self.expiry_input.setMaximumWidth(150)
        row1.addWidget(self.expiry_input)

        row1.addWidget(QLabel("Strikes:"))
        self.strikes_input = QSpinBox()
        self.strikes_input.setRange(10, 100)
        self.strikes_input.setValue(30)
        self.strikes_input.setMaximumWidth(80)
        row1.addWidget(self.strikes_input)

        row1.addWidget(QLabel("Trading Class:"))
        self.tclass_input = QComboBox()
        self.tclass_input.addItems(["auto", "SPX", "SPXW"])
        self.tclass_input.setMaximumWidth(100)
        row1.addWidget(self.tclass_input)

        row1.addStretch()
        settings_layout.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Throttle (ms):"))
        self.throttle_input = QSpinBox()
        self.throttle_input.setRange(100, 5000)
        self.throttle_input.setValue(400)
        self.throttle_input.setMaximumWidth(100)
        row2.addWidget(self.throttle_input)

        row2.addWidget(QLabel("OI Update (sec):"))
        self.oi_interval_input = QSpinBox()
        self.oi_interval_input.setRange(300, 7200)
        self.oi_interval_input.setValue(3600)
        self.oi_interval_input.setSingleStep(300)
        self.oi_interval_input.setMaximumWidth(100)
        row2.addWidget(self.oi_interval_input)

        row2.addStretch()
        settings_layout.addLayout(row2)

        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)

        # Status Display
        status_group = QGroupBox("Collector Status")
        status_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                color: white;
                border: 2px solid #4CAF50;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
        """)
        status_layout = QVBoxLayout()

        self.status_label = QLabel("Status: Stopped")
        self.status_label.setStyleSheet("""
            color: #f44336;
            font-size: 14px;
            padding: 15px;
            background: #2d2d2d;
            border-radius: 3px;
            font-family: Consolas;
        """)
        self.status_label.setWordWrap(True)
        status_layout.addWidget(self.status_label)

        status_group.setLayout(status_layout)
        layout.addWidget(status_group)

        # Controls
        controls_layout = QHBoxLayout()

        self.start_button = QPushButton("Start Collector")
        self.start_button.setStyleSheet("""
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
        self.start_button.clicked.connect(self.start_collector)
        controls_layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop Collector")
        self.stop_button.setStyleSheet("""
            QPushButton {
                background: #f44336;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 5px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover { background: #da190b; }
            QPushButton:disabled { background: #666; }
        """)
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_collector)
        controls_layout.addWidget(self.stop_button)

        controls_layout.addStretch()
        layout.addLayout(controls_layout)

        # Log Display
        log_group = QGroupBox("Collector Log")
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

        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setMaximumHeight(300)
        self.log_display.setStyleSheet("""
            QTextEdit {
                background: #1e1e1e;
                color: #ddd;
                font-family: Consolas;
                font-size: 10px;
                border: 1px solid #444;
                border-radius: 3px;
            }
        """)
        log_layout.addWidget(self.log_display)

        log_group.setLayout(log_layout)
        layout.addWidget(log_group)

        layout.addStretch()
        self.setLayout(layout)

    def start_collector(self):
        """Start the collector"""
        if self.collector and self.collector.running:
            self.log("Collector already running")
            return

        # Determine force_0dte based on expiry selection
        expiry_value = self.expiry_input.currentText()
        force_0dte = (expiry_value == "0DTE" or expiry_value == "auto")


        # Build config from UI
        config = CollectorConfig(
            host=self.host_input.text(),
            port=self.port_input.value(),
            client_id=self.client_id_input.value(),
            expiry=self.expiry_input.currentText(),
            n_strikes=self.strikes_input.value(),
            trading_class=None if self.tclass_input.currentText() == "auto" else self.tclass_input.currentText(),
            throttle_ms=self.throttle_input.value(),
            oi_update_interval=self.oi_interval_input.value(),
            force_0dte=force_0dte,
        )

        self.log("Starting collector...")
        self.log_display.clear()

        # Create and start collector
        self.collector = FlowCollector(config, progress_callback=self.log)

        if self.collector.start():
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.log("Collector started successfully")

            # Disable settings while running
            self.host_input.setEnabled(False)
            self.port_input.setEnabled(False)
            self.client_id_input.setEnabled(False)
            self.expiry_input.setEnabled(False)
            self.strikes_input.setEnabled(False)
            self.tclass_input.setEnabled(False)
            self.throttle_input.setEnabled(False)
            self.oi_interval_input.setEnabled(False)
        else:
            self.log("Failed to start collector")

    def stop_collector(self):
        """Stop the collector"""
        if not self.collector or not self.collector.running:
            self.log("Collector not running")
            return

        self.log("Stopping collector...")
        self.collector.stop()

        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

        # Re-enable settings
        self.host_input.setEnabled(True)
        self.port_input.setEnabled(True)
        self.client_id_input.setEnabled(True)
        self.expiry_input.setEnabled(True)
        self.strikes_input.setEnabled(True)
        self.tclass_input.setEnabled(True)
        self.throttle_input.setEnabled(True)
        self.oi_interval_input.setEnabled(True)

        self.log("Collector stopped")

    def update_status(self):
        """Update status display"""
        if not self.collector or not self.collector.running:
            self.status_label.setText("Status: Stopped")
            self.status_label.setStyleSheet("""
                color: #f44336;
                font-size: 14px;
                padding: 15px;
                background: #2d2d2d;
                border-radius: 3px;
                font-family: Consolas;
            """)
            return

        stats = self.collector.get_stats()

        # Safe formatting with fallbacks
        expiry = stats.get('expiry') or 'N/A'
        spot = stats.get('spot')
        spot_str = f"{spot:.2f}" if spot else 'N/A'
        contracts = stats.get('contracts_subscribed', 0)
        trades = stats.get('trades_collected', 0)
        last_update = stats.get('last_update') or 'N/A'

        status_text = (
            f"Status: Running\n"
            f"Expiry: {expiry}\n"
            f"Spot: {spot_str}\n"
            f"Contracts subscribed: {contracts}\n"
            f"Trades collected: {trades}\n"
            f"Last update: {last_update}"
        )

        self.status_label.setText(status_text)
        self.status_label.setStyleSheet("""
            color: #4CAF50;
            font-size: 12px;
            padding: 15px;
            background: #2d2d2d;
            border-radius: 3px;
            font-family: Consolas;
        """)

    def log(self, message: str):
        """Add message to log"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_display.append(f"[{timestamp}] {message}")

        # Auto-scroll
        self.log_display.verticalScrollBar().setValue(
            self.log_display.verticalScrollBar().maximum()
        )

    def closeEvent(self, event):
        """Handle tab close - stop collector"""
        if self.collector and self.collector.running:
            self.collector.stop()
        event.accept()