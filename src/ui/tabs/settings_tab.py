"""
Settings Tab - IB Connection & Database Management
"""
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QCheckBox, QLineEdit, QSpinBox, QProgressBar,
    QMessageBox, QGridLayout
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont

from src.config import MARKET_DATA_DB, IB_HOST, IB_PORT, IB_CLIENT_ID


class UpdateWorker(QThread):
    """Background worker for database updates"""
    progress = pyqtSignal(str)
    finished = pyqtSignal(bool, str)

    def __init__(self, symbols: list, force_full: bool, host: str, port: int, client_id: int):
        super().__init__()
        self.symbols = symbols
        self.force_full = force_full
        self.host = host
        self.port = port
        self.client_id = client_id

    def run(self):
        """Execute the update"""
        try:
            from src.core.data_manager import DataManager

            self.progress.emit("Initializing data manager...")

            # Create data manager
            dm = DataManager(market_db_path=MARKET_DATA_DB)

            # Connect to IB
            self.progress.emit(f"Connecting to IB @ {self.host}:{self.port}...")

            if not dm.connect_ib(self.host, self.port, self.client_id):
                self.finished.emit(False, "Failed to connect to Interactive Brokers")
                return

            self.progress.emit(f"Connected! Updating {len(self.symbols)} symbol(s)...")

            # Determine duration based on force_full
            duration = '5 Y' if self.force_full else '1 M'

            # Update each symbol
            success_count = 0
            failed_symbols = []

            for i, symbol in enumerate(self.symbols, 1):
                self.progress.emit(f"[{i}/{len(self.symbols)}] Updating {symbol}...")

                if dm.update_historical_data(symbol, duration=duration, bar_size='1 day'):
                    self.progress.emit(f"✓ {symbol} updated")
                    success_count += 1
                else:
                    self.progress.emit(f"✗ {symbol} failed")
                    failed_symbols.append(symbol)

            # Disconnect
            dm.disconnect_ib()

            # Final message
            if success_count == len(self.symbols):
                self.finished.emit(True, f"✓ Successfully updated all {success_count} symbol(s)")
            elif success_count > 0:
                self.finished.emit(True,
                                   f"⚠️  Updated {success_count}/{len(self.symbols)} symbol(s)\n"
                                   f"Failed: {', '.join(failed_symbols)}"
                                   )
            else:
                self.finished.emit(False, "✗ All updates failed")

        except Exception as e:
            import traceback
            error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
            self.finished.emit(False, error_msg)


class SettingsTab(QWidget):
    def __init__(self):
        super().__init__()
        self.worker = None
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(15, 15, 15, 15)

        # Title
        title = QLabel("Settings & Data Management")
        title.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        title.setStyleSheet("color: white; padding: 10px;")
        layout.addWidget(title)

        # Market Data Update Section
        self.setup_market_data_section(layout)

        layout.addStretch()
        self.setLayout(layout)

    def setup_market_data_section(self, parent_layout):
        """Market data update section"""
        market_group = QGroupBox("Market Data Update")
        market_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                color: white;
                border: 2px solid #2196F3;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
        """)
        market_layout = QVBoxLayout()

        # Info
        info_label = QLabel("Update historical data from Interactive Brokers")
        info_label.setStyleSheet("color: #aaa; font-size: 11px;")
        market_layout.addWidget(info_label)

        # Connection parameters
        conn_layout = QGridLayout()

        conn_layout.addWidget(QLabel("Host:"), 0, 0)
        self.host_input = QLineEdit()
        self.host_input.setText(IB_HOST)
        self.host_input.setPlaceholderText("127.0.0.1")
        self.host_input.setStyleSheet("background: #2d2d2d; color: white; border: 1px solid #444; padding: 5px;")
        conn_layout.addWidget(self.host_input, 0, 1)

        conn_layout.addWidget(QLabel("Port:"), 0, 2)
        self.port_input = QSpinBox()
        self.port_input.setRange(1000, 9999)
        self.port_input.setValue(IB_PORT)
        self.port_input.setStyleSheet("background: #2d2d2d; color: white; border: 1px solid #444; padding: 5px;")
        conn_layout.addWidget(self.port_input, 0, 3)

        conn_layout.addWidget(QLabel("Client ID:"), 1, 0)
        self.client_id_input = QSpinBox()
        self.client_id_input.setRange(0, 9999)
        self.client_id_input.setValue(IB_CLIENT_ID)
        self.client_id_input.setStyleSheet("background: #2d2d2d; color: white; border: 1px solid #444; padding: 5px;")
        conn_layout.addWidget(self.client_id_input, 1, 1)

        market_layout.addLayout(conn_layout)

        # Symbol selection
        symbols_label = QLabel("Select symbols to update:")
        symbols_label.setStyleSheet("color: white; font-weight: bold; margin-top: 10px;")
        market_layout.addWidget(symbols_label)

        symbols_layout = QHBoxLayout()

        self.spx_check = QCheckBox("SPX")
        self.spx_check.setChecked(True)
        self.spx_check.setStyleSheet("color: white;")
        symbols_layout.addWidget(self.spx_check)

        self.vix_check = QCheckBox("VIX")
        self.vix_check.setChecked(True)
        self.vix_check.setStyleSheet("color: white;")
        symbols_layout.addWidget(self.vix_check)

        self.spy_check = QCheckBox("SPY")
        self.spy_check.setChecked(False)
        self.spy_check.setStyleSheet("color: white;")
        symbols_layout.addWidget(self.spy_check)

        self.ndx_check = QCheckBox("NDX")
        self.ndx_check.setChecked(False)
        self.ndx_check.setStyleSheet("color: white;")
        symbols_layout.addWidget(self.ndx_check)

        self.rut_check = QCheckBox("RUT")
        self.rut_check.setChecked(False)
        self.rut_check.setStyleSheet("color: white;")
        symbols_layout.addWidget(self.rut_check)

        symbols_layout.addStretch()
        market_layout.addLayout(symbols_layout)

        # Update options
        options_layout = QHBoxLayout()

        self.force_full_check = QCheckBox("Force full update (5 years)")
        self.force_full_check.setStyleSheet("color: #FFA500;")
        self.force_full_check.setToolTip("Download 5 years of data instead of 1 month incremental")
        options_layout.addWidget(self.force_full_check)

        options_layout.addStretch()
        market_layout.addLayout(options_layout)

        # Update button
        self.update_btn = QPushButton("🔄 Update Database")
        self.update_btn.setStyleSheet("""
            QPushButton {
                background: #2196F3;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 5px;
                font-weight: bold;
                font-size: 13px;
            }
            QPushButton:hover {
                background: #1976D2;
            }
            QPushButton:disabled {
                background: #555;
            }
        """)
        self.update_btn.clicked.connect(self.update_database)
        market_layout.addWidget(self.update_btn)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #444;
                border-radius: 5px;
                text-align: center;
                background: #2d2d2d;
            }
            QProgressBar::chunk {
                background: #2196F3;
            }
        """)
        market_layout.addWidget(self.progress_bar)

        # Status label
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #ddd; padding: 5px;")
        self.status_label.setWordWrap(True)
        market_layout.addWidget(self.status_label)

        market_group.setLayout(market_layout)
        parent_layout.addWidget(market_group)

    def update_database(self):
        """Start database update in background"""
        # Get selected symbols
        symbols = []

        if self.spx_check.isChecked():
            symbols.append('SPX')
        if self.vix_check.isChecked():
            symbols.append('VIX')
        if self.spy_check.isChecked():
            symbols.append('SPY')
        if self.ndx_check.isChecked():
            symbols.append('NDX')
        if self.rut_check.isChecked():
            symbols.append('RUT')

        if not symbols:
            QMessageBox.warning(self, "No Selection", "Please select at least one symbol to update")
            return

        # Get connection params
        host = self.host_input.text().strip() or "127.0.0.1"
        port = self.port_input.value()
        client_id = self.client_id_input.value()

        # Get force_full option
        force_full = self.force_full_check.isChecked()

        # Confirm if force_full
        if force_full:
            reply = QMessageBox.question(
                self,
                "Confirm Full Update",
                f"Download 5 years of data for {len(symbols)} symbol(s)?\n\n"
                f"This may take several minutes.\n\n"
                f"Symbols: {', '.join(symbols)}",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )

            if reply != QMessageBox.StandardButton.Yes:
                return

        # Create and start worker
        self.worker = UpdateWorker(symbols, force_full, host, port, client_id)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.update_finished)
        self.worker.start()

        # Disable UI during update
        self.update_btn.setEnabled(False)
        self.spx_check.setEnabled(False)
        self.vix_check.setEnabled(False)
        self.spy_check.setEnabled(False)
        self.ndx_check.setEnabled(False)
        self.rut_check.setEnabled(False)
        self.force_full_check.setEnabled(False)

        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        self.status_label.setText("Starting update...")

    def update_progress(self, message: str):
        """Update progress label"""
        self.status_label.setText(message)
        self.status_label.setStyleSheet("color: #2196F3; padding: 5px;")

    def update_finished(self, success: bool, message: str):
        """Handle update completion"""
        # Re-enable UI
        self.update_btn.setEnabled(True)
        self.spx_check.setEnabled(True)
        self.vix_check.setEnabled(True)
        self.spy_check.setEnabled(True)
        self.ndx_check.setEnabled(True)
        self.rut_check.setEnabled(True)
        self.force_full_check.setEnabled(True)

        self.progress_bar.setVisible(False)

        if success:
            QMessageBox.information(self, "Update Complete", message)
            self.status_label.setText("✓ " + message)
            self.status_label.setStyleSheet("color: #4CAF50; padding: 5px;")
        else:
            QMessageBox.critical(self, "Update Failed", message)
            self.status_label.setText("✗ " + message)
            self.status_label.setStyleSheet("color: #f44336; padding: 5px;")