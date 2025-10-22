"""
Flow Collector - Wrapper for new_collector_jeudi_claude.py
Adapted for GUI control with threading
"""
import asyncio
import logging
import os
import queue
import sqlite3
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import List, Optional, Tuple, Dict
from src.config import OPTIONFLOW_DB

import pandas as pd

# Setup asyncio before ib_insync import
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

try:
    import nest_asyncio

    nest_asyncio.apply()
except Exception:
    pass

from ib_insync import IB, Index, Option


@dataclass
class CollectorConfig:
    """Collector configuration"""
    db_path: str = OPTIONFLOW_DB
    host: str = "127.0.0.1"
    port: int = 7497
    client_id: int = 40
    symbols: str = "SPX"
    exchange_index: str = "CBOE"
    option_exchange: str = "CBOE"
    trading_class: Optional[str] = None  # None = auto
    expiry: str = "auto"  # "YYYYMMDD" or "auto"
    n_strikes: int = 30
    throttle_ms: int = 400
    spot_fallback: float = 6700.0
    probe_option_for_spot: bool = True
    oi_update_interval: int = 3600  # 1 hour
    force_0dte: bool = True

@dataclass
class ContractState:
    last_write_ms: int = 0
    last_sig: tuple = field(default_factory=tuple)
    prev_volume: Optional[int] = None
    prev_last: Optional[float] = None


class FlowCollector:
    """
    Options flow collector with GUI integration
    """

    def __init__(self, config: CollectorConfig, progress_callback=None):
        self.cfg = config
        self.ib = IB()
        self.progress_callback = progress_callback
        self.writer = None
        self.state: Dict[int, ContractState] = {}
        self.running = False
        self.thread = None

        # Stats
        self.stats = {
            'trades_collected': 0,
            'contracts_subscribed': 0,
            'last_update': None,
            'expiry': None,
            'spot': None
        }

    def _log(self, message: str, level: str = "INFO"):
        """Send log message to GUI"""
        if self.progress_callback:
            self.progress_callback(f"[{level}] {message}")
        print(f"[{level}] {message}")

    def start(self):
        """Start collector in background thread"""
        if self.running:
            self._log("Collector already running", "WARNING")
            return False

        self.running = True
        self.thread = threading.Thread(target=self._run_collector, daemon=True)
        self.thread.start()
        return True

    def stop(self):
        """Stop collector"""
        self.running = False
        if self.writer:
            self.writer.stop()

        try:
            if self.ib.isConnected():
                self.ib.disconnect()
        except Exception:
            pass

        if self.thread:
            self.thread.join(timeout=5.0)

    def get_stats(self) -> Dict:
        """Get current collector statistics"""
        return self.stats.copy()

    def _run_collector(self):
        """Main collector loop (runs in thread)"""
        try:
            self._log("Initializing collector...")
            self._ensure_db()
            self._log("Database initialized")

            # Initialize writer
            self.writer = BatchWriter(self.cfg.db_path, progress_callback=self._log)
            self.writer.start()

            # Connect to IB
            self._log(f"Connecting to IB @ {self.cfg.host}:{self.cfg.port}...")
            self.ib.connect(self.cfg.host, self.cfg.port, clientId=self.cfg.client_id)
            self._log("Connected to IB successfully")

            # Build universe
            self._log("Building option universe...")
            expiry, tclass, contracts = self._build_universe()
            self.stats['expiry'] = expiry
            self._log(f"Universe ready: {len(contracts)} contracts, expiry={expiry}, class={tclass}")

            # Initial OI snapshot
            self._log("Collecting initial open interest snapshot...")
            self._collect_oi_snapshot(contracts)

            # Subscribe to live data
            self._log("Subscribing to live market data...")
            subscribed = 0
            for c in contracts:
                try:
                    self.ib.reqMktData(c, "", snapshot=False, regulatorySnapshot=False)
                    subscribed += 1
                except Exception as e:
                    self._log(f"Subscribe failed for {c}: {e}", "DEBUG")

            self.stats['contracts_subscribed'] = subscribed
            self._log(f"Subscribed to {subscribed} contracts")
            self._log("Collector running - streaming data...")

            last_oi_update = time.time()

            # Main loop
            while self.running:
                self.ib.waitOnUpdate(timeout=1.0)

                # Update OI periodically
                current_time = time.time()
                if current_time - last_oi_update > self.cfg.oi_update_interval:
                    self._log(f"Updating open interest (interval: {self.cfg.oi_update_interval}s)...")
                    self._collect_oi_snapshot(contracts)
                    last_oi_update = current_time

                # Process ticks
                for t in list(self.ib.tickers()):
                    if not t or not t.contract:
                        continue
                    self._maybe_store_tick(t, expiry, tclass)

                self.stats['last_update'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        except Exception as e:
            self._log(f"Collector error: {str(e)}", "ERROR")

        finally:
            self._log("Collector stopped")
            self.running = False

    def _ensure_db(self):
        """Initialize database schema"""
        os.makedirs(os.path.dirname(self.cfg.db_path) or ".", exist_ok=True)
        conn = sqlite3.connect(self.cfg.db_path, timeout=30.0)

        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")

            # Create trades table
            conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                expiry TEXT NOT NULL,
                right TEXT NOT NULL,
                strike REAL NOT NULL,
                last REAL,
                bid REAL,
                ask REAL,
                qty INTEGER,
                volume INTEGER,
                spot REAL,
                estimation TEXT,
                confidence REAL,
                mid REAL,
                spread REAL,
                signed_qty INTEGER,
                symbol TEXT NOT NULL,
                trading_class TEXT,
                contract_local TEXT,
                iv REAL,
                delta REAL,
                gamma REAL,
                vega REAL,
                theta REAL,
                trade_qty INTEGER,
                open_interest INTEGER,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            """)

            # Create indexes
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_trades_ts ON trades(ts);",
                "CREATE INDEX IF NOT EXISTS idx_trades_key ON trades(expiry, right, strike);",
                "CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);",
                "CREATE INDEX IF NOT EXISTS idx_trades_oi ON trades(open_interest);",
            ]
            for idx in indexes:
                conn.execute(idx)

            # Add missing columns if needed
            existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(trades);").fetchall()}
            new_cols = [
                ("iv", "REAL"), ("delta", "REAL"), ("gamma", "REAL"),
                ("vega", "REAL"), ("theta", "REAL"), ("trade_qty", "INTEGER"),
                ("open_interest", "INTEGER")
            ]
            for col, typ in new_cols:
                if col not in existing_cols:
                    conn.execute(f"ALTER TABLE trades ADD COLUMN {col} {typ};")

            conn.commit()
        finally:
            conn.close()

    def _build_universe(self) -> Tuple[str, str, List[Option]]:
        """Build option universe"""
        symbol = self.cfg.symbols.split(",")[0].strip().upper() or "SPX"

        # Qualify index
        idx = None
        for ex in [self.cfg.exchange_index, "CBOE", "SMART"]:
            if not ex:
                continue
            try:
                cand = Index(symbol, exchange=ex)
                self.ib.qualifyContracts(cand)
                idx = cand
                self._log(f"Index qualified: {idx} on {ex}")
                break
            except Exception as e:
                self._log(f"Index qualify failed for {symbol}@{ex}: {e}", "DEBUG")

        if idx is None:
            raise RuntimeError(f"Could not qualify index {symbol}")

        # Get option parameters
        params = self.ib.reqSecDefOptParams(idx.symbol, "", idx.secType, idx.conId)
        if not params:
            raise RuntimeError("No option parameters returned")

        self._log(f"Found {len(params)} option parameter sets")

        # Choose SPXW if available
        node = None
        for p in params:
            if (p.tradingClass or "").upper().startswith("SPXW"):
                node = p
                break
        if node is None:
            node = params[0]

        # Determine expiry
        expiry_config = self.cfg.expiry

        if expiry_config == "0DTE":
            # Force 0DTE mode
            today_str = date.today().strftime("%Y%m%d")
            if today_str in node.expirations:
                expiry = today_str
                self._log(f"0DTE mode: Using today's expiry {expiry}", "INFO")
            else:
                self._log(f"0DTE requested but not available today, falling back to nearest", "WARNING")
                expiry = self._pick_nearest_expiry(node.expirations)

        elif expiry_config == "auto" or not expiry_config:
            if self.cfg.force_0dte:
                # Try 0DTE first
                today_str = date.today().strftime("%Y%m%d")
                if today_str in node.expirations:
                    expiry = today_str
                    self._log(f"Auto-selected 0DTE: {expiry}", "INFO")
                else:
                    self._log(f"0DTE not available, selecting nearest expiry", "INFO")
                    expiry = self._pick_nearest_expiry(node.expirations)
            else:
                expiry = self._pick_nearest_expiry(node.expirations)

        else:
            # Specific expiry provided (YYYYMMDD format)
            expiry = expiry_config
            if expiry not in node.expirations:
                self._log(f"Requested expiry {expiry} not in available expirations", "WARNING")
                self._log(f"Available expirations: {node.expirations[:10]}", "INFO")
                expiry = self._pick_nearest_expiry(node.expirations)
                self._log(f"Using nearest available expiry: {expiry}", "INFO")

        # Validate expiry is set
        if not expiry:
            raise RuntimeError("Could not determine expiry date")

        self._log(f"Final expiry selected: {expiry}", "INFO")


        tclass = (self.cfg.trading_class or self._suggest_trading_class(expiry)).upper()

        # Get spot price
        spot = self._get_spot_price(idx, node, symbol, expiry, tclass)
        self.stats['spot'] = spot
        self._log(f"Spot price: {spot}")

        # Select strikes
        strikes = sorted([float(s) for s in node.strikes if isinstance(s, (float, int))])
        chosen = self._select_strikes(strikes, spot)
        self._log(f"Selected {len(chosen)} strikes around {spot}")

        # Build contracts
        contracts = []
        for right in ("C", "P"):
            for k in chosen:
                contracts.append(Option(
                    symbol=symbol,
                    lastTradeDateOrContractMonth=expiry,
                    strike=float(k),
                    right=right,
                    exchange=self.cfg.option_exchange,
                    currency="USD",
                    multiplier="100",
                    tradingClass=tclass
                ))

        # Qualify contracts
        qualified = []
        for c in contracts:
            try:
                q = self.ib.qualifyContracts(c)
                if q:
                    qualified.append(q[0])
            except Exception:
                pass

        if not qualified:
            raise RuntimeError("No options qualified")

        return expiry, tclass, qualified

    def _pick_nearest_expiry(self, expirations: List[str]) -> str:
        """
        Pick the nearest expiry with improved 0DTE handling
        Priority: 0DTE (today) > Next available future expiry
        """
        today = date.today()
        parsed = []

        for e in expirations:
            try:
                exp_date = datetime.strptime(e, "%Y%m%d").date()
                days_to_expiry = (exp_date - today).days
                parsed.append((exp_date, days_to_expiry, e))
            except Exception:
                continue

        if not parsed:
            # Fallback: next Friday
            days_ahead = (4 - today.weekday()) % 7 or 7
            next_friday = today + timedelta(days=days_ahead)
            return next_friday.strftime("%Y%m%d")

        # Sort by date
        parsed.sort()

        # Check if 0DTE available (today)
        for exp_date, dte, exp_str in parsed:
            if dte == 0:
                self._log(f"0DTE expiry found: {exp_str}", "INFO")
                return exp_str

        # Otherwise, pick nearest future expiry (dte > 0)
        for exp_date, dte, exp_str in parsed:
            if dte > 0:
                self._log(f"Next expiry selected (DTE={dte}): {exp_str}", "INFO")
                return exp_str

        # Fallback if all expirations are in the past
        self._log("All expirations in past, using fallback", "WARNING")
        days_ahead = (4 - today.weekday()) % 7 or 7
        next_friday = today + timedelta(days=days_ahead)
        return next_friday.strftime("%Y%m%d")

    def _suggest_trading_class(self, expiry_yyyymmdd: str) -> str:
        """Suggest SPX or SPXW based on expiry"""
        try:
            d = datetime.strptime(expiry_yyyymmdd, "%Y%m%d").date()
            # Third Friday = monthly (SPX), else weekly (SPXW)
            is_third_friday = d.weekday() == 4 and 15 <= d.day <= 21
            return "SPX" if is_third_friday else "SPXW"
        except Exception:
            return "SPXW"

    def _get_spot_price(self, idx, node, symbol, expiry, tclass) -> float:
        """Get spot price with fallbacks"""
        spot = None

        # Try index market data
        try:
            t_index = self.ib.reqMktData(idx, "", snapshot=True)
            deadline = time.time() + 10
            while time.time() < deadline and (not spot or spot <= 0):
                self.ib.sleep(0.5)
                for px in [t_index.last, t_index.marketPrice(), t_index.close]:
                    if px and float(px) > 0:
                        spot = float(px)
                        break
                if spot:
                    break
        except Exception:
            pass

        # Try option undPrice
        if (not spot or spot <= 0) and self.cfg.probe_option_for_spot:
            strikes = [float(s) for s in node.strikes[:10]]
            for k in strikes:
                try:
                    probe = Option(symbol=symbol, lastTradeDateOrContractMonth=expiry,
                                   strike=float(k), right="C", exchange=self.cfg.option_exchange,
                                   currency="USD", multiplier="100", tradingClass=tclass)
                    self.ib.qualifyContracts(probe)
                    t = self.ib.reqMktData(probe, "", snapshot=True)
                    deadline = time.time() + 8
                    while time.time() < deadline:
                        self.ib.sleep(0.3)
                        mg = getattr(t, "modelGreeks", None)
                        if mg and getattr(mg, "undPrice", None):
                            spot = float(mg.undPrice)
                            break
                    if spot:
                        break
                except Exception:
                    continue

        # Fallback
        if not spot or spot <= 0:
            spot = self.cfg.spot_fallback
            self._log(f"Using fallback spot: {spot}", "WARNING")

        return spot

    def _select_strikes(self, strikes: List[float], spot: float) -> List[float]:
        """Select strikes around spot"""
        if not strikes:
            step = max(5.0, round(spot * 0.005, 1))
            base = int(spot // step) * step
            half = self.cfg.n_strikes // 2
            return [base + i * step for i in range(-half, half + 1)]

        nearest = min(strikes, key=lambda k: abs(k - spot))
        idx_near = strikes.index(nearest)
        half = self.cfg.n_strikes // 2
        lo = max(0, idx_near - half)
        hi = min(len(strikes), idx_near + half + 1)
        return strikes[lo:hi]

    def _collect_oi_snapshot(self, contracts: List[Option]):
        """Collect open interest snapshot"""
        try:
            tickers = self.ib.reqTickers(*contracts)
            self.ib.sleep(2.0)

            oi_count = sum(1 for t in tickers if t and getattr(t, 'openInterest', None))
            self._log(f"OI available for {oi_count}/{len(contracts)} contracts")
        except Exception as e:
            self._log(f"OI collection failed: {e}", "ERROR")

    def _maybe_store_tick(self, t, expiry: str, tclass: str):
        """Store tick if changed and throttle passed"""
        c = t.contract
        key = int(getattr(c, "conId", 0) or 0)
        if key == 0:
            return

        st = self.state.setdefault(key, ContractState())
        now_ms = int(time.time() * 1000)

        if (now_ms - st.last_write_ms) < self.cfg.throttle_ms:
            self._update_state_only(st, t)
            return

        # Extract data
        last = _as_float(getattr(t, "last", None))
        bid = _as_float(getattr(t, "bid", None))
        ask = _as_float(getattr(t, "ask", None))
        volume = _as_int(getattr(t, "volume", None))
        open_interest = _as_int(getattr(t, "openInterest", None))

        # Calculate trade size
        trade_qty = 0
        if volume is not None and st.prev_volume is not None:
            dv = volume - st.prev_volume
            if dv > 0:
                trade_qty = dv
        st.prev_volume = volume

        if trade_qty == 0 and last is not None:
            lastSize = _as_int(getattr(t, "lastSize", None))
            if lastSize and (st.prev_last is None or last != st.prev_last):
                trade_qty = lastSize
        st.prev_last = last

        # Classification
        mid = (bid + ask) / 2 if (bid and ask and bid > 0 and ask > 0) else None
        spread = (ask - bid) if (bid and ask and bid > 0 and ask > 0) else None
        est, conf, sign = _classify_trade(last, bid, ask)
        signed_qty = int((trade_qty or 0) * sign) if sign else 0

        # Greeks
        mg = getattr(t, "modelGreeks", None)
        iv = _as_float(getattr(mg, "impliedVol", None)) if mg else None
        delta = _as_float(getattr(mg, "delta", None)) if mg else None
        gamma = _as_float(getattr(mg, "gamma", None)) if mg else None
        vega = _as_float(getattr(mg, "vega", None)) if mg else None
        theta = _as_float(getattr(mg, "theta", None)) if mg else None
        spot_live = _as_float(getattr(mg, "undPrice", None)) if mg else None

        # Check signature
        sig = (last, bid, ask, volume, iv, gamma, delta, vega, theta, trade_qty, open_interest)
        if sig == st.last_sig:
            return

        # Build row
        row = (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            str(expiry), str(c.right), float(c.strike),
            last, bid, ask, int(trade_qty or 0), int(volume or 0), spot_live,
            est, float(conf), mid, spread, int(signed_qty),
            str(c.symbol), tclass, getattr(c, "localSymbol", None),
            iv, delta, gamma, vega, theta, int(trade_qty or 0), int(open_interest or 0)
        )

        self.writer.add(row)
        st.last_sig = sig
        st.last_write_ms = now_ms
        self.stats['trades_collected'] += 1

    def _update_state_only(self, st: ContractState, t):
        """Update state without writing"""
        volume = _as_int(getattr(t, "volume", None))
        last = _as_float(getattr(t, "last", None))
        if volume is not None:
            st.prev_volume = volume
        if last is not None:
            st.prev_last = last


# Helper classes and functions
class BatchWriter:
    """Batch write trades to database"""

    def __init__(self, db_path: str, batch_size: int = 200, flush_sec: float = 1.0, progress_callback=None):
        self.db_path = db_path
        self.batch_size = batch_size
        self.flush_sec = flush_sec
        self.q = queue.Queue()
        self._running = True
        self.progress_callback = progress_callback
        self.th = threading.Thread(target=self._worker, daemon=True)

    def start(self):
        self.th.start()

    def stop(self):
        self._running = False
        try:
            self.q.put_nowait(None)
        except Exception:
            pass
        self.th.join(timeout=3.0)

    def add(self, row: Tuple):
        try:
            self.q.put(row, timeout=0.2)
        except Exception:
            pass

    def _worker(self):
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")

            last_flush = time.time()
            buf = []

            while self._running:
                timeout = max(self.flush_sec - (time.time() - last_flush), 0.05)
                try:
                    item = self.q.get(timeout=timeout)
                except queue.Empty:
                    item = None

                if item is None:
                    if buf:
                        self._flush(conn, buf)
                        buf.clear()
                    last_flush = time.time()
                    if not self._running:
                        break
                    continue

                buf.append(item)
                if len(buf) >= self.batch_size:
                    self._flush(conn, buf)
                    buf.clear()
                    last_flush = time.time()

        except Exception as e:
            if self.progress_callback:
                self.progress_callback(f"[ERROR] Batch writer crashed: {e}")
        finally:
            if conn:
                try:
                    conn.close()
                except Exception:
                    pass

    def _flush(self, conn: sqlite3.Connection, rows: List[Tuple]):
        if not rows:
            return

        sql = """
            INSERT INTO trades (
                ts, expiry, right, strike, last, bid, ask, qty, volume, spot,
                estimation, confidence, mid, spread, signed_qty, symbol, trading_class, contract_local,
                iv, delta, gamma, vega, theta, trade_qty, open_interest
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """
        try:
            conn.executemany(sql, rows)
            conn.commit()
        except Exception as e:
            if self.progress_callback:
                self.progress_callback(f"[ERROR] DB flush failed: {e}")


def _classify_trade(last, bid, ask) -> Tuple[str, float, int]:
    """Classify trade as buy/sell"""
    try:
        if not all([last, bid, ask]) or bid <= 0 or ask <= 0 or last <= 0 or ask < bid:
            return "indetermine", 0.0, 0

        if last >= ask:
            return "achat", 0.95, 1
        if last <= bid:
            return "vente", 0.95, -1

        rng = max(ask - bid, 1e-6)
        dist = (last - bid) / rng

        if dist > 0.6:
            return "achat", float(min(0.9, 0.6 + 0.4 * dist)), 1
        if dist < 0.4:
            return "vente", float(min(0.9, 0.6 + 0.4 * (1 - dist))), -1

        return "indetermine", 0.2, 0
    except Exception:
        return "indetermine", 0.0, 0


def _as_float(x):
    try:
        if x is None or pd.isna(x):
            return None
        v = float(x)
        return v if pd.notna(v) else None
    except Exception:
        return None


def _as_int(x):
    try:
        if x is None or pd.isna(x):
            return None
        return int(float(x))
    except Exception:
        return None