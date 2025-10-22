"""
IB Data Manager - Unified interface for Interactive Brokers data
Combines functionality from tws_connector, contract_utils, database_initializer
"""
import os
import json
import sqlite3
import random
from datetime import datetime, timedelta
from typing import List, Tuple, Optional, Dict

from ib_insync import IB, Stock, Index, Future, Option, Contract


class IBDataManager:
    """
    Centralized manager for IB connections and data operations
    """

    def __init__(
            self,
            host: str = "127.0.0.1",
            port: int = 7497,
            db_path: str = "db/market_data.db",
            tickers_json: str = "core/ressources/tickers_verified.json"
    ):
        self.host = host
        self.port = port
        self.db_path = db_path
        self.tickers_json = tickers_json
        self.ib: Optional[IB] = None

        # Load tickers catalog
        if os.path.exists(tickers_json):
            with open(tickers_json, 'r') as f:
                self.tickers_catalog = json.load(f)
        else:
            self.tickers_catalog = {}
            print(f"Warning: {tickers_json} not found")

    def connect(self, client_id: Optional[int] = None) -> IB:
        """Connect to TWS/Gateway"""
        if self.ib is not None and self.ib.isConnected():
            return self.ib

        if client_id is None:
            client_id = random.randint(100, 999)

        self.ib = IB()
        try:
            self.ib.connect(self.host, self.port, clientId=client_id, timeout=5)
            print(f"Connected to IB @ {self.host}:{self.port} (clientId={client_id})")
            return self.ib
        except Exception as e:
            print(f"Connection failed: {e}")
            raise

    def disconnect(self):
        """Disconnect from TWS/Gateway"""
        if self.ib is not None and self.ib.isConnected():
            self.ib.disconnect()
            self.ib = None

    def is_connected(self) -> bool:
        """Check if currently connected"""
        return self.ib is not None and self.ib.isConnected()

    def test_connection(self) -> Tuple[bool, str]:
        """Test connection and return (success, message)"""
        try:
            self.connect()
            msg = f"Connected successfully to {self.host}:{self.port}"
            self.disconnect()
            return True, msg
        except Exception as e:
            return False, f"Connection failed: {str(e)}"

    def get_contract(self, symbol: str) -> Contract:
        """
        Get qualified contract from catalog
        """
        if symbol not in self.tickers_catalog:
            raise ValueError(f"Unknown ticker: {symbol}")

        info = self.tickers_catalog[symbol]
        sec_type = info["secType"]

        if sec_type == "STK":
            contract = Stock(symbol, info["exchange"], info["currency"])
        elif sec_type == "IND":
            contract = Index(symbol, info["exchange"], info["currency"])
        elif sec_type == "FUT":
            contract = Future(
                symbol=symbol,
                lastTradeDateOrContractMonth=info["lastTradeDateOrContractMonth"],
                exchange=info["exchange"],
                currency=info["currency"]
            )
        else:
            raise ValueError(f"Unsupported secType: {sec_type}")

        if not self.is_connected():
            self.connect()

        qualified = self.ib.qualifyContracts(contract)
        if not qualified:
            raise Exception(f"Failed to qualify contract: {symbol}")

        return qualified[0]

    def get_spot_price(self, symbol: str) -> float:
        """Get current spot price"""
        contract = self.get_contract(symbol)

        ticker = self.ib.reqMktData(contract, snapshot=True)
        self.ib.sleep(2)

        spot = ticker.last or ticker.close or ticker.marketPrice()
        self.ib.cancelMktData(contract)

        if spot is None or spot <= 0:
            raise ValueError(f"Could not get spot price for {symbol}")

        return float(spot)

    def fetch_historical_data(
            self,
            symbol: str,
            start_date: datetime,
            end_date: datetime,
            bar_size: str = "1 day"
    ) -> List:
        """
        Fetch historical data from IB in chunks (max 365 days per request)
        """
        contract = self.get_contract(symbol)
        all_bars = []

        # Split into chunks
        current = start_date
        while current < end_date:
            chunk_end = min(current + timedelta(days=365), end_date)

            try:
                bars = self.ib.reqHistoricalData(
                    contract,
                    endDateTime=chunk_end,
                    durationStr="1 Y",
                    barSizeSetting=bar_size,
                    whatToShow="TRADES",
                    useRTH=True,
                    formatDate=1
                )

                if bars:
                    all_bars.extend(bars)
                    print(f"  {symbol}: {current.date()} -> {chunk_end.date()}: {len(bars)} bars")

                self.ib.sleep(0.5)  # Rate limiting

            except Exception as e:
                print(f"  Error fetching {symbol} ({current.date()} -> {chunk_end.date()}): {e}")

            current = chunk_end

        return all_bars

    def update_database(
            self,
            symbols: List[str],
            force_full: bool = False,
            progress_callback=None
    ):
        """
        Update SQLite database with historical data
        If force_full=False, only fetch data since last known date
        progress_callback: function(message: str) to report progress
        """
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        con = sqlite3.connect(self.db_path)
        cursor = con.cursor()

        if not self.is_connected():
            self.connect()

        for i, symbol in enumerate(symbols):
            if progress_callback:
                progress_callback(f"Processing {symbol} ({i + 1}/{len(symbols)})...")

            table_name = f"{symbol.lower()}_data"

            # Determine start date
            if force_full:
                start_date = datetime(2015, 1, 1)
            else:
                cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                    (table_name,)
                )
                if cursor.fetchone():
                    cursor.execute(f"SELECT MAX(date) FROM {table_name}")
                    result = cursor.fetchone()
                    if result and result[0]:
                        start_date = datetime.strptime(result[0], "%Y-%m-%d") + timedelta(days=1)
                    else:
                        start_date = datetime(2015, 1, 1)
                else:
                    start_date = datetime(2015, 1, 1)

            end_date = datetime.now()

            if start_date >= end_date:
                if progress_callback:
                    progress_callback(f"  {symbol}: Already up to date")
                continue

            print(f"\nUpdating {symbol} from {start_date.date()} to {end_date.date()}...")

            try:
                bars = self.fetch_historical_data(symbol, start_date, end_date)

                if not bars:
                    if progress_callback:
                        progress_callback(f"  {symbol}: No data available")
                    continue

                # Create table if needed
                cursor.execute(f"""
                    CREATE TABLE IF NOT EXISTS {table_name} (
                        date TEXT PRIMARY KEY,
                        open REAL,
                        high REAL,
                        low REAL,
                        close REAL
                    )
                """)

                # Insert data
                for bar in bars:
                    cursor.execute(
                        f"""INSERT OR REPLACE INTO {table_name} 
                            (date, open, high, low, close) VALUES (?, ?, ?, ?, ?)""",
                        (
                            bar.date.strftime("%Y-%m-%d"),
                            bar.open,
                            bar.high,
                            bar.low,
                            bar.close
                        )
                    )

                con.commit()
                msg = f"  {symbol}: {len(bars)} bars inserted/updated"
                print(msg)
                if progress_callback:
                    progress_callback(msg)

            except Exception as e:
                msg = f"  {symbol}: Error - {str(e)}"
                print(msg)
                if progress_callback:
                    progress_callback(msg)

        con.close()

        if progress_callback:
            progress_callback("Database update complete!")

    def get_available_expiries(self, symbol: str) -> List[str]:
        """Get available option expiries for a symbol"""
        contract = self.get_contract(symbol)

        chains = self.ib.reqSecDefOptParams(
            contract.symbol, "", contract.secType, contract.conId
        )

        if not chains:
            return []

        expirations = sorted(set(chains[0].expirations))
        return expirations

    def get_available_tickers(self) -> List[str]:
        """Get list of all available tickers from catalog"""
        return sorted(self.tickers_catalog.keys())