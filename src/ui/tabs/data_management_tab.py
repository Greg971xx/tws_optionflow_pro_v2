"""
Data Management Tab - Archive and cleanup old data
"""
import os
import sqlite3
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QTableWidget, QTableWidgetItem, QHeaderView,
    QCheckBox, QSpinBox, QComboBox, QProgressBar, QMessageBox,
    QFileDialog, QAbstractItemView
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QColor

from src.config import DATABASE_PATH


class ArchiveWorker(QThread):
    """Background worker for archiving data"""
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(bool, str)

    def __init__(self, db_path, table, condition, archive_path):
        super().__init__()
        self.db_path = db_path
        self.table = table
        self.condition = condition
        self.archive_path = archive_path

    def run(self):
        """Archive and delete data"""
        try:
            self.progress.emit(10, f"Connecting to database...")
            conn = sqlite3.connect(self.db_path)

            # Extract data to archive
            self.progress.emit(20, f"Extracting data from {self.table}...")
            query = f"SELECT * FROM {self.table} WHERE {self.condition}"
            df = pd.read_sql(query, conn)

            if df.empty:
                self.finished.emit(True, "No data to archive")
                conn.close()
                return

            row_count = len(df)
            self.progress.emit(40, f"Found {row_count:,} rows to archive")

            # Save to CSV
            self.progress.emit(60, f"Creating archive...")
            csv_path = self.archive_path.with_suffix('.csv')
            df.to_csv(csv_path, index=False)

            # Zip the CSV
            self.progress.emit(70, f"Compressing archive...")
            with zipfile.ZipFile(self.archive_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                zf.write(csv_path, csv_path.name)

            # Delete CSV (keep only zip)
            csv_path.unlink()

            # Delete from database
            self.progress.emit(80, f"Deleting data from database...")
            cursor = conn.cursor()
            cursor.execute(f"DELETE FROM {self.table} WHERE {self.condition}")
            conn.commit()

            deleted_count = cursor.rowcount

            # Vacuum to reclaim space
            self.progress.emit(90, f"Optimizing database...")
            cursor.execute("VACUUM")
            conn.commit()

            conn.close()

            # Get archive size
            archive_size = self.archive_path.stat().st_size / (1024 * 1024)

            self.progress.emit(100, f"Complete!")
            self.finished.emit(
                True,
                f"Archived {deleted_count:,} rows ({archive_size:.2f} MB)\n"
                f"Saved to: {self.archive_path}"
            )

        except Exception as e:
            self.finished.emit(False, f"Error: {str(e)}")


class DataManagementTab(QWidget):
    def __init__(self):
        super().__init__()
        self.db_path = DATABASE_PATH
        self.setup_ui()
        self.refresh_stats()

    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(15, 15, 15, 15)

        # Title
        title = QLabel("Data Management & Archiving")
        title.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        title.setStyleSheet("color: white; padding: 10px;")
        layout.addWidget(title)

        subtitle = QLabel("Archive old data to keep database light and fast")
        subtitle.setStyleSheet("color: #aaa; font-size: 12px; padding: 5px;")
        layout.addWidget(subtitle)


        # Database Stats
        self.create_stats_section(layout)

        # Options Flow Archiving
        self.create_flow_archive_section(layout)

        # Historical Data Archiving
        self.create_historical_archive_section(layout)

        # Progress
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
                background: #4CAF50;
            }
        """)
        layout.addWidget(self.progress_bar)

        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #ddd; padding: 5px;")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        layout.addStretch()
        self.setLayout(layout)

    def load_tickers(self):
        """Load available tickers with row counts"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Get all tables ending with _data
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name LIKE '%_data'
                ORDER BY name
            """)
            tables = cursor.fetchall()

            ticker_info = []
            for (table_name,) in tables:
                # Get row count
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                count = cursor.fetchone()[0]

                # Extract ticker (remove _data suffix)
                ticker = table_name.replace('_data', '').upper()

                ticker_info.append({
                    'ticker': ticker,
                    'table': table_name,
                    'count': count
                })

            conn.close()

            # Populate table
            self.ticker_table.setRowCount(len(ticker_info))

            for row_idx, info in enumerate(ticker_info):
                # Checkbox
                checkbox = QCheckBox()
                checkbox_widget = QWidget()
                checkbox_layout = QHBoxLayout(checkbox_widget)
                checkbox_layout.addWidget(checkbox)
                checkbox_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
                checkbox_layout.setContentsMargins(0, 0, 0, 0)
                self.ticker_table.setCellWidget(row_idx, 0, checkbox_widget)

                # Ticker name
                ticker_item = QTableWidgetItem(info['ticker'])
                ticker_item.setData(Qt.ItemDataRole.UserRole, info['table'])

                # Color essential tickers
                if info['ticker'] in ['SPX', 'VIX', 'SPY']:
                    ticker_item.setForeground(QColor("#4CAF50"))
                    ticker_item.setFont(QFont("Arial", 10, QFont.Weight.Bold))

                self.ticker_table.setItem(row_idx, 1, ticker_item)

                # Row count
                count_item = QTableWidgetItem(f"{info['count']:,}")
                count_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.ticker_table.setItem(row_idx, 2, count_item)

                # Individual delete button
                delete_btn = QPushButton("Delete")
                delete_btn.setStyleSheet("""
                    QPushButton {
                        background: #f44336;
                        color: white;
                        border: none;
                        padding: 5px 10px;
                        border-radius: 3px;
                    }
                    QPushButton:hover { background: #da190b; }
                """)
                delete_btn.clicked.connect(lambda checked, t=info['table']: self.delete_single_ticker(t))

                btn_widget = QWidget()
                btn_layout = QHBoxLayout(btn_widget)
                btn_layout.addWidget(delete_btn)
                btn_layout.setContentsMargins(5, 5, 5, 5)
                self.ticker_table.setCellWidget(row_idx, 3, btn_widget)

        except Exception as e:
            print(f"Error loading tickers: {e}")

    def keep_essential_tickers(self):
        """Keep only SPX, VIX, SPY - delete all others"""
        essential = ['SPX', 'VIX', 'SPY']

        # Count tickers to delete
        to_delete = []
        for row in range(self.ticker_table.rowCount()):
            ticker = self.ticker_table.item(row, 1).text()
            if ticker not in essential:
                table = self.ticker_table.item(row, 1).data(Qt.ItemDataRole.UserRole)
                to_delete.append((ticker, table))

        if not to_delete:
            QMessageBox.information(self, "No Action", "Only essential tickers remain!")
            return

        # Confirm
        ticker_list = ', '.join([t[0] for t in to_delete])
        reply = QMessageBox.question(
            self,
            "Confirm Bulk Delete",
            f"Delete {len(to_delete)} ticker(s)?\n\n"
            f"Tickers to delete: {ticker_list}\n\n"
            f"Data will be archived first.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply != QMessageBox.StandardButton.Yes:
            return

        # Archive and delete each
        archive_dir = Path(self.db_path).parent / "archives"
        archive_dir.mkdir(exist_ok=True)

        self.status_label.setText(f"Archiving {len(to_delete)} tickers...")

        for ticker, table in to_delete:
            archive_path = archive_dir / f"archive_{table}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"

            try:
                # Quick archive and delete
                conn = sqlite3.connect(self.db_path)
                df = pd.read_sql(f"SELECT * FROM {table}", conn)

                if not df.empty:
                    csv_path = archive_path.with_suffix('.csv')
                    df.to_csv(csv_path, index=False)

                    with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                        zf.write(csv_path, csv_path.name)
                    csv_path.unlink()

                # Drop table
                cursor = conn.cursor()
                cursor.execute(f"DROP TABLE IF EXISTS {table}")
                conn.commit()
                conn.close()

                self.status_label.setText(f"Deleted {ticker}...")

            except Exception as e:
                print(f"Error deleting {ticker}: {e}")

        # Vacuum
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("VACUUM")
            conn.close()
        except Exception:
            pass

        QMessageBox.information(
            self,
            "Cleanup Complete",
            f"Deleted {len(to_delete)} ticker(s)\n\n"
            f"Archives saved to db/archives/"
        )

        self.refresh_stats()

    def delete_selected_tickers(self):
        """Delete tickers selected via checkboxes"""
        to_delete = []

        for row in range(self.ticker_table.rowCount()):
            checkbox_widget = self.ticker_table.cellWidget(row, 0)
            checkbox = checkbox_widget.findChild(QCheckBox)

            if checkbox and checkbox.isChecked():
                ticker = self.ticker_table.item(row, 1).text()
                table = self.ticker_table.item(row, 1).data(Qt.ItemDataRole.UserRole)
                to_delete.append((ticker, table))

        if not to_delete:
            QMessageBox.warning(self, "No Selection", "Please select tickers to delete")
            return

        # Confirm
        ticker_list = ', '.join([t[0] for t in to_delete])
        reply = QMessageBox.question(
            self,
            "Confirm Delete",
            f"Delete {len(to_delete)} selected ticker(s)?\n\n"
            f"{ticker_list}\n\n"
            f"Data will be archived first.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply != QMessageBox.StandardButton.Yes:
            return

        # Same logic as keep_essential_tickers but for selected only
        archive_dir = Path(self.db_path).parent / "archives"
        archive_dir.mkdir(exist_ok=True)

        for ticker, table in to_delete:
            archive_path = archive_dir / f"archive_{table}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"

            try:
                conn = sqlite3.connect(self.db_path)
                df = pd.read_sql(f"SELECT * FROM {table}", conn)

                if not df.empty:
                    csv_path = archive_path.with_suffix('.csv')
                    df.to_csv(csv_path, index=False)

                    with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                        zf.write(csv_path, csv_path.name)
                    csv_path.unlink()

                cursor = conn.cursor()
                cursor.execute(f"DROP TABLE IF EXISTS {table}")
                conn.commit()
                conn.close()

            except Exception as e:
                print(f"Error deleting {ticker}: {e}")

        # Vacuum
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("VACUUM")
            conn.close()
        except Exception:
            pass

        QMessageBox.information(self, "Delete Complete", f"Deleted {len(to_delete)} ticker(s)")
        self.refresh_stats()

    def delete_single_ticker(self, table_name):
        """Delete a single ticker via the Delete button"""
        ticker = table_name.replace('_data', '').upper()

        reply = QMessageBox.question(
            self,
            "Confirm Delete",
            f"Delete all {ticker} data?\n\nData will be archived first.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply != QMessageBox.StandardButton.Yes:
            return

        archive_dir = Path(self.db_path).parent / "archives"
        archive_dir.mkdir(exist_ok=True)
        archive_path = archive_dir / f"archive_{table_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"

        try:
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql(f"SELECT * FROM {table_name}", conn)

            if not df.empty:
                csv_path = archive_path.with_suffix('.csv')
                df.to_csv(csv_path, index=False)

                with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                    zf.write(csv_path, csv_path.name)
                csv_path.unlink()

            cursor = conn.cursor()
            cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
            conn.commit()
            conn.close()

            # Vacuum
            conn = sqlite3.connect(self.db_path)
            conn.execute("VACUUM")
            conn.close()

            QMessageBox.information(self, "Delete Complete", f"Deleted {ticker}")
            self.refresh_stats()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to delete {ticker}: {e}")


    def create_stats_section(self, parent_layout):
        """Database statistics overview"""
        stats_group = QGroupBox("Database Overview")
        stats_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                color: white;
                border: 2px solid #4CAF50;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
        """)
        stats_layout = QVBoxLayout()

        # Stats display
        self.stats_label = QLabel("Loading...")
        self.stats_label.setStyleSheet("""
            color: #ddd;
            font-family: Consolas;
            font-size: 11px;
            padding: 15px;
            background: #2d2d2d;
            border-radius: 3px;
        """)
        self.stats_label.setTextFormat(Qt.TextFormat.RichText)
        stats_layout.addWidget(self.stats_label)

        # Refresh button
        refresh_btn = QPushButton("🔄 Refresh Stats")
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
        refresh_btn.clicked.connect(self.refresh_stats)
        stats_layout.addWidget(refresh_btn)

        stats_group.setLayout(stats_layout)
        parent_layout.addWidget(stats_group)

    def create_flow_archive_section(self, parent_layout):
        """Options flow archiving controls"""
        flow_group = QGroupBox("📊 Options Flow Data")
        flow_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                color: white;
                border: 2px solid #FF9800;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
        """)
        flow_layout = QVBoxLayout()

        # Info
        info = QLabel("Archive and remove old options flow data by expiry date")
        info.setStyleSheet("color: #aaa; font-size: 11px;")
        flow_layout.addWidget(info)

        # Controls
        controls_layout = QHBoxLayout()

        controls_layout.addWidget(QLabel("Delete trades older than:"))

        self.flow_days_spin = QSpinBox()
        self.flow_days_spin.setRange(1, 365)
        self.flow_days_spin.setValue(30)
        self.flow_days_spin.setSuffix(" days")
        self.flow_days_spin.setMaximumWidth(100)
        controls_layout.addWidget(self.flow_days_spin)

        controls_layout.addStretch()

        archive_flow_btn = QPushButton("💾 Archive & Delete Old Trades")
        archive_flow_btn.setStyleSheet("""
            QPushButton {
                background: #FF9800;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover { background: #F57C00; }
        """)
        archive_flow_btn.clicked.connect(self.archive_old_flow)
        controls_layout.addWidget(archive_flow_btn)

        flow_layout.addLayout(controls_layout)

        # Expiry-specific deletion
        expiry_layout = QHBoxLayout()
        expiry_layout.addWidget(QLabel("Or delete specific expiry:"))

        self.expiry_combo = QComboBox()
        self.expiry_combo.setMinimumWidth(200)
        expiry_layout.addWidget(self.expiry_combo)

        delete_expiry_btn = QPushButton("🗑️ Delete Expiry")
        delete_expiry_btn.setStyleSheet("""
            QPushButton {
                background: #f44336;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 3px;
                font-weight: bold;
            }
            QPushButton:hover { background: #da190b; }
        """)
        delete_expiry_btn.clicked.connect(self.delete_specific_expiry)
        expiry_layout.addWidget(delete_expiry_btn)

        expiry_layout.addStretch()
        flow_layout.addLayout(expiry_layout)

        flow_group.setLayout(flow_layout)
        parent_layout.addWidget(flow_group)

    def create_historical_archive_section(self, parent_layout):
        """Historical data archiving controls"""
        hist_group = QGroupBox("📈 Historical Market Data")
        hist_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                color: white;
                border: 2px solid #9C27B0;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
        """)
        hist_layout = QVBoxLayout()

        info = QLabel("Archive old historical price data by ticker")
        info.setStyleSheet("color: #aaa; font-size: 11px;")
        hist_layout.addWidget(info)

        # Ticker selection with table
        ticker_label = QLabel("Available Tickers:")
        ticker_label.setStyleSheet("color: white; font-weight: bold; padding: 5px;")
        hist_layout.addWidget(ticker_label)

        # Table for ticker selection
        self.ticker_table = QTableWidget()
        self.ticker_table.setColumnCount(4)
        self.ticker_table.setHorizontalHeaderLabels(["Select", "Ticker", "Rows", "Actions"])
        self.ticker_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.ticker_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.ticker_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.ticker_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.ticker_table.setMaximumHeight(200)
        self.ticker_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.ticker_table.setStyleSheet("""
            QTableWidget {
                background: #2d2d2d;
                color: white;
                gridline-color: #444;
            }
            QHeaderView::section {
                background: #3d3d3d;
                color: white;
                padding: 5px;
                border: 1px solid #444;
            }
        """)
        hist_layout.addWidget(self.ticker_table)

        # Bulk actions
        bulk_layout = QHBoxLayout()

        # Keep only essential tickers
        keep_essential_btn = QPushButton("⭐ Keep Only SPX, VIX, SPY")
        keep_essential_btn.setStyleSheet("""
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
        keep_essential_btn.clicked.connect(self.keep_essential_tickers)
        bulk_layout.addWidget(keep_essential_btn)

        # Delete selected
        delete_selected_btn = QPushButton("🗑️ Delete Selected")
        delete_selected_btn.setStyleSheet("""
            QPushButton {
                background: #f44336;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover { background: #da190b; }
        """)
        delete_selected_btn.clicked.connect(self.delete_selected_tickers)
        bulk_layout.addWidget(delete_selected_btn)

        bulk_layout.addStretch()
        hist_layout.addLayout(bulk_layout)

        # Archive old data section
        archive_layout = QHBoxLayout()
        archive_layout.addWidget(QLabel("Or archive data older than:"))

        self.hist_days_spin = QSpinBox()
        self.hist_days_spin.setRange(30, 3650)
        self.hist_days_spin.setValue(730)
        self.hist_days_spin.setSuffix(" days")
        self.hist_days_spin.setMaximumWidth(100)
        archive_layout.addWidget(self.hist_days_spin)

        archive_old_btn = QPushButton("💾 Archive Old Data")
        archive_old_btn.setStyleSheet("""
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
        archive_old_btn.clicked.connect(self.archive_old_historical)
        archive_layout.addWidget(archive_old_btn)

        archive_layout.addStretch()
        hist_layout.addLayout(archive_layout)

        hist_group.setLayout(hist_layout)
        parent_layout.addWidget(hist_group)

    def refresh_stats(self):
        """Refresh database statistics"""
        try:
            conn = sqlite3.connect(self.db_path)

            # Database size
            db_size = Path(self.db_path).stat().st_size / (1024 * 1024)

            # Table stats
            tables_info = []

            # Trades table
            trades_count = pd.read_sql("SELECT COUNT(*) as c FROM trades", conn)['c'][0]
            trades_expiries = pd.read_sql("""
                SELECT COUNT(DISTINCT expiry) as c FROM trades
            """, conn)['c'][0]
            tables_info.append(f"<b>trades:</b> {trades_count:,} rows, {trades_expiries} expiries")

            # OI snapshots
            oi_count = pd.read_sql("SELECT COUNT(*) as c FROM oi_snapshots", conn)['c'][0]
            tables_info.append(f"<b>oi_snapshots:</b> {oi_count:,} rows")

            # Historical data
            hist_tables = ['spx_data', 'vix_data', 'ndx_data', 'rut_data']
            hist_total = 0
            for table in hist_tables:
                try:
                    count = pd.read_sql(f"SELECT COUNT(*) as c FROM {table}", conn)['c'][0]
                    hist_total += count
                except:
                    pass

            tables_info.append(f"<b>Historical data:</b> {hist_total:,} rows across {len(hist_tables)} tickers")

            conn.close()

            # Update display
            stats_html = f"""
            <p style='font-size: 13px;'>
            <b>📊 Database Size:</b> {db_size:.2f} MB<br><br>
            <b>📋 Tables:</b><br>
            {'<br>'.join(f'  • {info}' for info in tables_info)}
            </p>
            """

            self.stats_label.setText(stats_html)

            # Update expiry combo
            self.load_expiries()

            self.load_tickers()  # Refresh ticker table

        except Exception as e:
            self.stats_label.setText(f"<p style='color: #f44336;'>Error loading stats: {e}</p>")

    def load_expiries(self):
        """Load available expiries into combo"""
        try:
            conn = sqlite3.connect(self.db_path)
            expiries = pd.read_sql("""
                SELECT DISTINCT expiry, COUNT(*) as count
                FROM trades
                GROUP BY expiry
                ORDER BY expiry DESC
            """, conn)
            conn.close()

            self.expiry_combo.clear()
            for _, row in expiries.iterrows():
                self.expiry_combo.addItem(
                    f"{row['expiry']} ({row['count']:,} trades)",
                    row['expiry']
                )
        except Exception as e:
            print(f"Error loading expiries: {e}")

    def archive_old_flow(self):
        """Archive old flow data"""
        days = self.flow_days_spin.value()
        cutoff_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        # Confirm
        reply = QMessageBox.question(
            self,
            "Confirm Archive",
            f"Archive all trades older than {days} days (before {cutoff_date})?\n\n"
            f"Data will be saved to archive_flow_{datetime.now().strftime('%Y%m%d')}.zip",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply != QMessageBox.StandardButton.Yes:
            return

        # Archive path
        archive_dir = Path(self.db_path).parent / "archives"
        archive_dir.mkdir(exist_ok=True)
        archive_path = archive_dir / f"archive_flow_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"

        # Start worker
        self.start_archive(
            table="trades",
            condition=f"ts < '{cutoff_date}'",
            archive_path=archive_path
        )

    def delete_specific_expiry(self):
        """Delete specific expiry"""
        if self.expiry_combo.count() == 0:
            QMessageBox.warning(self, "No Data", "No expiries available")
            return

        expiry = self.expiry_combo.currentData()

        reply = QMessageBox.question(
            self,
            "Confirm Delete",
            f"Delete ALL trades for expiry {expiry}?\n\n"
            f"Data will be archived first.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply != QMessageBox.StandardButton.Yes:
            return

        archive_dir = Path(self.db_path).parent / "archives"
        archive_dir.mkdir(exist_ok=True)
        archive_path = archive_dir / f"archive_expiry_{expiry}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"

        self.start_archive(
            table="trades",
            condition=f"expiry = '{expiry}'",
            archive_path=archive_path
        )

    def archive_old_historical(self):
        """Archive old historical data"""
        ticker = self.ticker_combo.currentText()
        days = self.hist_days_spin.value()
        cutoff_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        reply = QMessageBox.question(
            self,
            "Confirm Archive",
            f"Archive {ticker} data older than {days} days (before {cutoff_date})?\n\n"
            f"Data will be saved to archive.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply != QMessageBox.StandardButton.Yes:
            return

        # Determine tables
        if ticker == "All":
            tables = ['spx_data', 'vix_data', 'ndx_data', 'rut_data']
        else:
            tables = [f"{ticker.lower()}_data"]

        archive_dir = Path(self.db_path).parent / "archives"
        archive_dir.mkdir(exist_ok=True)

        for table in tables:
            archive_path = archive_dir / f"archive_{table}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
            self.start_archive(
                table=table,
                condition=f"date < '{cutoff_date}'",
                archive_path=archive_path
            )

    def start_archive(self, table, condition, archive_path):
        """Start archive worker thread"""
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Starting archive...")

        self.worker = ArchiveWorker(self.db_path, table, condition, archive_path)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.archive_finished)
        self.worker.start()

    def update_progress(self, value, message):
        """Update progress bar"""
        self.progress_bar.setValue(value)
        self.status_label.setText(message)

    def archive_finished(self, success, message):
        """Handle archive completion"""
        self.progress_bar.setVisible(False)

        if success:
            QMessageBox.information(self, "Archive Complete", message)
            self.refresh_stats()
        else:
            QMessageBox.critical(self, "Archive Failed", message)

        self.status_label.setText("")