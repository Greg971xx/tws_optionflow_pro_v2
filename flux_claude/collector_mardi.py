# =============================================
# File: collector_lundi.py
# Purpose: Fixed version - Collect SPX(/SPXW) options ticks + Greeks into SQLite avec volume *réaliste*
# Fixes:
#   - Better expiration date logic (avoid same-day expirations)
#   - Improved spot price detection
#   - Better error handling for contract qualification
#   - Added debugging for contract resolution
# =============================================

# --- S'assurer qu'une boucle asyncio existe avant d'importer ib_insync/eventkit ---
import asyncio

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())
try:
    import nest_asyncio;

    nest_asyncio.apply()
except Exception:
    pass

import argparse
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

import pandas as pd

# ---------------- ib_insync ----------------
try:
    from ib_insync import IB, Index, Option, util
except Exception as e:
    print("ib_insync import failed. Install it with: pip install ib_insync", file=sys.stderr)
    raise


# --------------- Logging -------------------
def setup_logging(level: str = "INFO"):
    os.makedirs("logs", exist_ok=True)
    logging.getLogger().handlers.clear()
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    fh = logging.FileHandler("logs/collector.log", encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)


# ------------- Config model ----------------
@dataclass
class Config:
    db_path: str = "db/optionflow.db"
    host: str = "127.0.0.1"
    port: int = 7497
    client_id: int = 20
    symbols: str = "SPX"  # comma-separated allowed
    exchange_index: str = "CBOE"  # index exchange preference
    option_exchange: str = "CBOE"  # options exchange
    trading_class: Optional[str] = None  # "SPX" (monthly) ou "SPXW" (weeklies). None -> auto selon le jour.
    expiry: str = "20250916" # "YYYYMMDD" ou "auto"
    n_strikes: int = 15  # total strikes autour de l'ATM
    throttle_ms: int = 400  # ms mini entre deux writes sur un même contrat
    log_level: str = "INFO"
    spot_fallback: float = 6615.0  # Updated default fallback
    probe_option_for_spot: bool = True


def load_config_from_yaml(path: Optional[str]) -> Config:
    cfg = Config()
    if not path:
        return cfg
    try:
        import yaml
    except Exception:
        logging.warning("pyyaml non installé; --config ignoré")
        return cfg
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        for k, v in data.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        logging.info(f"Config chargée depuis {path}")
    except Exception as e:
        logging.error(f"Echec de chargement config {path}: {e}")
    return cfg


# ------------- SQLite helpers --------------
DDL_TRADES = """
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    expiry TEXT NOT NULL,
    right TEXT NOT NULL,
    strike REAL NOT NULL,
    last REAL,
    bid REAL,
    ask REAL,
    qty INTEGER,                 -- = trade_qty (taille de trade réaliste par tick)
    volume INTEGER,              -- volume cumulatif IB
    spot REAL,
    estimation TEXT,
    confidence REAL,
    mid REAL,
    spread REAL,
    signed_qty INTEGER,          -- qty signée (direction) selon estimation
    symbol TEXT NOT NULL,
    trading_class TEXT,
    contract_local TEXT,
    -- Greeks
    iv REAL,
    delta REAL,
    gamma REAL,
    vega REAL,
    theta REAL,
    -- Trade size réaliste (stocké aussi dans qty pour compat)
    trade_qty INTEGER,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""

INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_trades_ts ON trades(ts);",
    "CREATE INDEX IF NOT EXISTS idx_trades_key ON trades(expiry, right, strike);",
    "CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);",
]

GREEKS_COLS = [
    ("iv", "REAL"),
    ("delta", "REAL"),
    ("gamma", "REAL"),
    ("vega", "REAL"),
    ("theta", "REAL"),
    ("trade_qty", "INTEGER"),
]


def ensure_db(db_path: str):
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=30.0)
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute(DDL_TRADES)
        for idx in INDEXES:
            conn.execute(idx)
        # migrations douces
        cur = conn.execute("PRAGMA table_info(trades);")
        existing = {row[1] for row in cur.fetchall()}
        for col, typ in GREEKS_COLS:
            if col not in existing:
                conn.execute(f"ALTER TABLE trades ADD COLUMN {col} {typ};")
        conn.commit()
    finally:
        conn.close()


# -------- Classification (heuristique mid) --------
def classify_trade(last: Optional[float], bid: Optional[float], ask: Optional[float]) -> Tuple[str, float, int]:
    """
    Retourne: (estimation, confidence, sign)
      estimation ∈ {"achat","vente","indetermine"}
      sign ∈ {+1, -1, 0}
    """
    try:
        if last is None or bid is None or ask is None or bid <= 0 or ask <= 0 or last <= 0 or ask < bid:
            return "indetermine", 0.0, 0
        mid = 0.5 * (bid + ask)
        if last >= (ask - 1e-9):
            return "achat", 0.95, +1
        if last <= (bid + 1e-9):
            return "vente", 0.95, -1
        rng = max(ask - bid, 1e-6)
        dist = (last - bid) / rng  # 0..1
        if dist > 0.6:
            return "achat", float(min(0.9, 0.6 + 0.4 * dist)), +1
        if dist < 0.4:
            return "vente", float(min(0.9, 0.6 + 0.4 * (1 - dist))), -1
        return "indetermine", 0.2, 0
    except Exception:
        return "indetermine", 0.0, 0


# --------------- Batch writer ----------------
class BatchWriter:
    def __init__(self, db_path: str, batch_size: int = 200, flush_sec: float = 1.0):
        self.db_path = db_path
        self.batch_size = batch_size
        self.flush_sec = flush_sec
        self.q: "queue.Queue[Tuple]" = queue.Queue()
        self._running = True
        self.th = threading.Thread(target=self._worker, daemon=True)
        self.log = logging.getLogger("BatchWriter")

    def start(self):
        self.th.start()

    def stop(self):
        self._running = False
        try:
            self.q.put_nowait(None)  # sentinel
        except Exception:
            pass
        self.th.join(timeout=3.0)

    def add(self, row: Tuple):
        try:
            self.q.put(row, timeout=0.2)
        except Exception as e:
            self.log.error(f"queue.put failed: {e}")

    def _worker(self):
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            last_flush = time.time()
            buf: List[Tuple] = []
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
            self.log.exception(f"Batch worker crashed: {e}")
        finally:
            if conn is not None:
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
                iv, delta, gamma, vega, theta, trade_qty
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """
        try:
            conn.executemany(sql, rows)
            conn.commit()
            self.log.debug(f"Flushed {len(rows)} rows")
        except Exception as e:
            self.log.exception(f"DB flush failed: {e}")


# -------------- IB utilities ------------------
def is_third_friday(d: date) -> bool:
    return d.weekday() == 4 and 15 <= d.day <= 21


def suggest_trading_class(expiry_yyyymmdd: str) -> str:
    try:
        d = datetime.strptime(expiry_yyyymmdd, "%Y%m%d").date()
        return "SPX" if is_third_friday(d) else "SPXW"
    except Exception:
        return "SPXW"


def pick_nearest_expiry(expirations: List[str]) -> str:
    """Pick the nearest expiry that's NOT today and preferably at least 1-2 days out"""
    today = date.today()
    parsed = []
    for e in expirations:
        try:
            exp_date = datetime.strptime(e, "%Y%m%d").date()
            # Skip expirations that are today or in the past
            if exp_date > today:
                parsed.append(exp_date)
        except Exception:
            continue

    if not parsed:
        # Fallback: try next Friday
        days_ahead = (4 - today.weekday()) % 7  # Next Friday
        if days_ahead == 0:  # Today is Friday
            days_ahead = 7
        next_friday = today + timedelta(days=days_ahead)
        return next_friday.strftime("%Y%m%d")

    parsed.sort()
    return parsed[0].strftime("%Y%m%d")


# -------------- Collector core -----------------
@dataclass
class ContractState:
    last_write_ms: int = 0
    last_sig: tuple = field(default_factory=tuple)
    prev_volume: Optional[int] = None
    prev_last: Optional[float] = None


class Collector:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.ib = IB()
        self.log = logging.getLogger("Collector")
        self.writer = BatchWriter(cfg.db_path, batch_size=200, flush_sec=1.0)
        self.state: Dict[int, ContractState] = {}  # conId -> state

    # ---- connect ----
    def connect(self):
        self.ib.connect(self.cfg.host, self.cfg.port, clientId=self.cfg.client_id)
        self.log.info(f"Connected to IB @ {self.cfg.host}:{self.cfg.port} (clientId={self.cfg.client_id})")

    # ---- setup universe ----
    def build_universe(self) -> Tuple[str, str, List[Option]]:
        symbol = self.cfg.symbols.split(",")[0].strip().upper() or "SPX"

        # Qualifier l'index sur plusieurs exchanges
        idx = None
        for ex in [self.cfg.exchange_index, "CBOE", "CBOEIND", "SMART"]:
            if not ex:
                continue
            try:
                cand = Index(symbol, exchange=ex)
                self.ib.qualifyContracts(cand)
                idx = cand
                self.log.info(f"Index qualified: {idx} on exchange {ex}")
                break
            except Exception as e:
                self.log.debug(f"Index qualify failed for {symbol}@{ex}: {e}")

        if idx is None:
            raise RuntimeError(f"Could not qualify index {symbol} on any exchange")

        # Paramètres options
        self.log.info("Requesting SecDefOptParams...")
        params = self.ib.reqSecDefOptParams(idx.symbol, "", idx.secType, idx.conId)
        if not params:
            raise RuntimeError("No SecDefOptParams returned")

        self.log.info(f"Found {len(params)} option parameter sets")
        for i, p in enumerate(params):
            self.log.info(
                f"  Param set {i}: tradingClass={p.tradingClass}, exchange={p.exchange}, multiplier={p.multiplier}")
            self.log.info(f"    Expirations: {p.expirations[:5]}{'...' if len(p.expirations) > 5 else ''}")

        # Choose parameter set (prefer SPXW)
        node = None
        for p in params:
            if (p.tradingClass or "").upper().startswith("SPXW"):
                node = p
                break
        if node is None and params:
            node = params[0]
        if node is None:
            raise RuntimeError("Could not select option parameter set")

        self.log.info(f"Selected parameter set: {node.tradingClass} on {node.exchange}")

        # Determine expiry
        expiry = self.cfg.expiry
        if expiry == "auto" or not expiry:
            expiry = pick_nearest_expiry(node.expirations)
            self.log.info(f"Auto-selected expiry: {expiry}")

        # Validate expiry exists
        if expiry not in node.expirations:
            self.log.warning(f"Requested expiry {expiry} not in available expirations")
            expiry = pick_nearest_expiry(node.expirations)
            self.log.info(f"Using nearest available expiry: {expiry}")

        tclass = (self.cfg.trading_class or suggest_trading_class(expiry)).upper()
        self.log.info(f"Using trading class: {tclass}")

        # Get spot price via multiple methods
        spot = self._get_spot_price(idx, node, symbol, expiry, tclass)
        self.log.info(f"Determined spot price: {spot}")

        # Strikes disponibles
        strikes = []
        try:
            strikes = sorted([float(s) for s in node.strikes if
                              isinstance(s, (float, int)) or (isinstance(s, str) and s.replace('.', '', 1).isdigit())])
            self.log.info(
                f"Available strikes: {len(strikes)} strikes from {strikes[0] if strikes else 'N/A'} to {strikes[-1] if strikes else 'N/A'}")
        except Exception as e:
            self.log.error(f"Error processing strikes: {e}")

        # Choisir n_strikes autour de l'ATM
        chosen = self._select_strikes(strikes, spot)
        self.log.info(f"Selected {len(chosen)} strikes around spot {spot}: {chosen}")

        # Build contracts
        contracts: List[Option] = []
        for right in ("C", "P"):
            for k in chosen:
                contracts.append(Option(
                    symbol=symbol,
                    lastTradeDateOrContractMonth=expiry,
                    strike=float(k),
                    right=right,
                    exchange=self.cfg.option_exchange or "CBOE",
                    currency="USD",
                    multiplier="100",
                    tradingClass=tclass
                ))

        # Qualifier les contrats avec debugging
        qual: List[Option] = []
        failed_count = 0
        for i, c in enumerate(contracts):
            try:
                q = self.ib.qualifyContracts(c)
                if q:
                    qual.append(q[0])
                    if i < 5:  # Log first few for debugging
                        self.log.info(f"Qualified contract {i}: {q[0]}")
                else:
                    failed_count += 1
                    if failed_count <= 5:  # Log first few failures
                        self.log.warning(f"No qualification result for: {c}")
            except Exception as e:
                failed_count += 1
                if failed_count <= 5:  # Log first few failures
                    self.log.warning(f"Qualify failed for {c}: {e}")

        if not qual:
            raise RuntimeError("No options qualified (check exchange/trading_class/permissions)")

        self.log.info(f"Successfully qualified {len(qual)} contracts ({failed_count} failed)")
        return expiry, tclass, qual

    def _get_spot_price(self, idx, node, symbol, expiry, tclass) -> float:
        """Try multiple methods to get spot price"""
        spot = None

        # Method 1: Direct index market data
        try:
            self.log.info("Attempting to get spot price from index...")
            t_index = self.ib.reqMktData(idx, "", snapshot=True, regulatorySnapshot=False)
            t_end = time.time() + 10
            while time.time() < t_end and (spot is None or spot <= 0):
                self.ib.sleep(0.5)
                for px in [t_index.last, t_index.marketPrice(), t_index.close]:
                    if px is not None and not pd.isna(px) and float(px) > 0:
                        spot = float(px)
                        self.log.info(f"Got spot from index: {spot}")
                        break
                if spot:
                    break
        except Exception as e:
            self.log.debug(f"reqMktData on index failed: {e}")

        # Method 2: Probe option for underlying price
        if (spot is None or spot <= 0) and self.cfg.probe_option_for_spot:
            self.log.info("Attempting to get spot price from option undPrice...")
            strikes = [float(s) for s in node.strikes[:10]]  # Try first 10 strikes
            for k in strikes:
                try:
                    probe = Option(symbol=symbol, lastTradeDateOrContractMonth=expiry,
                                   strike=float(k), right="C",
                                   exchange=self.cfg.option_exchange or "CBOE",
                                   currency="USD", multiplier="100", tradingClass=tclass)
                    self.ib.qualifyContracts(probe)
                    t = self.ib.reqMktData(probe, "", snapshot=True, regulatorySnapshot=False)
                    t_end = time.time() + 8
                    while time.time() < t_end:
                        self.ib.sleep(0.3)
                        mg = getattr(t, "modelGreeks", None)
                        if mg and getattr(mg, "undPrice", None):
                            spot = float(mg.undPrice)
                            self.log.info(f"Got spot from option undPrice: {spot}")
                            break
                    if spot:
                        break
                except Exception as e:
                    self.log.debug(f"probe option for spot failed at strike {k}: {e}")
                    continue

        # Method 3: Fallbacks
        if spot is None or spot <= 0:
            if self.cfg.spot_fallback and self.cfg.spot_fallback > 0:
                spot = float(self.cfg.spot_fallback)
                self.log.warning(f"Using configured spot_fallback={spot}")
            else:
                # Try to use a reasonable SPX value
                spot = 5950.0  # Conservative estimate for current SPX levels
                self.log.warning(f"Using hardcoded fallback spot={spot}")

        return spot

    def _select_strikes(self, strikes: List[float], spot: float) -> List[float]:
        """Select strikes around the spot price"""
        if not strikes:
            # Generate strikes if none available
            step = max(round(spot * 0.005, 1), 5.0)  # ~0.5% steps, min 5 points
            base = int(spot // step) * step
            half = max(self.cfg.n_strikes // 2, 1)
            return [base + i * step for i in range(-half, half + 1)]

        # Find nearest strike to spot
        nearest = min(strikes, key=lambda k: abs(k - spot))
        idx_near = strikes.index(nearest)
        half = max(self.cfg.n_strikes // 2, 1)
        lo = max(0, idx_near - half)
        hi = min(len(strikes), idx_near + half + 1)
        return strikes[lo:hi]

    # ---- subscribe & run ----
    def run(self):
        ensure_db(self.cfg.db_path)
        self.writer.start()
        self.connect()
        expiry, tclass, contracts = self.build_universe()

        # snapshot initial
        try:
            self.ib.reqTickers(*contracts)
            self.ib.sleep(0.5)
        except Exception:
            pass

        # flux live
        subscribed_count = 0
        for c in contracts:
            try:
                self.ib.reqMktData(c, "", snapshot=False, regulatorySnapshot=False)
                subscribed_count += 1
            except Exception as e:
                self.log.debug(f"reqMktData live failed for {c}: {e}")

        self.log.info(f"Streaming {subscribed_count} contracts... (Ctrl+C to stop)")
        try:
            while True:
                self.ib.waitOnUpdate(timeout=1.0)
                for t in list(self.ib.tickers()):
                    if not t or not t.contract:
                        continue
                    self._maybe_store_tick(t, expiry, tclass)
        except KeyboardInterrupt:
            self.log.info("Stopping...")
        finally:
            self.writer.stop()
            try:
                if self.ib.isConnected():
                    self.ib.disconnect()
            except Exception:
                pass

    def _maybe_store_tick(self, t, expiry: str, tclass: str):
        c = t.contract
        key = int(getattr(c, "conId", 0) or 0)
        if key == 0:
            return
        st = self.state.setdefault(key, ContractState())
        now_ms = int(time.time() * 1000)
        if (now_ms - st.last_write_ms) < self.cfg.throttle_ms:
            # on met quand même à jour l'état
            self._update_state_only(st, t)
            return

        last = _as_float(getattr(t, "last", None))
        bid = _as_float(getattr(t, "bid", None))
        ask = _as_float(getattr(t, "ask", None))
        lastSize = _as_int(getattr(t, "lastSize", None))
        volume = _as_int(getattr(t, "volume", None))

        # --- trade size réaliste ---
        trade_qty = 0
        if volume is not None:
            if st.prev_volume is not None:
                dv = volume - st.prev_volume
                if dv > 0:
                    trade_qty = dv
            st.prev_volume = volume

        # fallback sur lastSize quand un nouveau prix 'last' apparaît
        if trade_qty == 0 and last is not None and lastSize and lastSize > 0:
            if st.prev_last is None or last != st.prev_last:
                trade_qty = lastSize
        st.prev_last = last

        mid = (bid + ask) / 2.0 if (bid and ask and bid > 0 and ask > 0) else None
        spread = (ask - bid) if (bid and ask and bid > 0 and ask > 0) else None

        est, conf, sign = classify_trade(last, bid, ask)
        signed_qty = int((trade_qty or 0) * sign) if sign else 0

        # Greeks & spot
        iv = delta = gamma = vega = theta = None
        mg = getattr(t, "modelGreeks", None)
        if mg:
            iv = _as_float(getattr(mg, "impliedVol", None))
            delta = _as_float(getattr(mg, "delta", None))
            gamma = _as_float(getattr(mg, "gamma", None))
            vega = _as_float(getattr(mg, "vega", None))
            theta = _as_float(getattr(mg, "theta", None))
        spot_live = _as_float(getattr(mg, "undPrice", None)) if mg else None

        # signature pour éviter doublons
        sig = (last, bid, ask, volume, iv, gamma, delta, vega, theta, trade_qty)
        if sig == st.last_sig:
            return

        # INSERT: qty == trade_qty (compat app)
        row = (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            str(expiry),
            str(c.right),
            float(c.strike),
            last,
            bid,
            ask,
            int(trade_qty or 0),  # qty -> trade size réaliste
            int(volume or 0),
            spot_live,
            est,
            float(conf),
            mid,
            spread,
            int(signed_qty),
            str(c.symbol),
            tclass,
            getattr(c, "localSymbol", None),
            iv, delta, gamma, vega, theta,
            int(trade_qty or 0)  # trade_qty
        )
        self.writer.add(row)
        st.last_sig = sig
        st.last_write_ms = now_ms

    def _update_state_only(self, st: "ContractState", t):
        """Met à jour prev_volume/prev_last même si on ne write pas à cause du throttle."""
        last = _as_float(getattr(t, "last", None))
        volume = _as_int(getattr(t, "volume", None))
        if volume is not None:
            st.prev_volume = volume
        if last is not None:
            st.prev_last = last


# -------------- utils -----------------
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
        v = int(float(x))
        return v
    except Exception:
        return None


# -------------- main -----------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Collect SPX(/SPXW) options ticks with Greeks into SQLite (realistic volumes)")
    p.add_argument("--config", help="YAML config path")
    p.add_argument("--db", dest="db_path")
    p.add_argument("--host")
    p.add_argument("--port", type=int)
    p.add_argument("--client-id", type=int)
    p.add_argument("--symbols")
    p.add_argument("--exchange-index")
    p.add_argument("--option-exchange")
    p.add_argument("--trading-class")
    p.add_argument("--expiry")
    p.add_argument("--n-strikes", type=int)
    p.add_argument("--throttle-ms", type=int)
    p.add_argument("--log-level")
    p.add_argument("--spot-fallback", type=float, dest="spot_fallback")
    p.add_argument("--no-probe-option", action="store_true", help="Disable option probe for spot")
    return p.parse_args()


def merge_cli(cfg: Config, ns: argparse.Namespace) -> Config:
    if ns.db_path: cfg.db_path = ns.db_path
    if ns.host: cfg.host = ns.host
    if ns.port: cfg.port = ns.port
    if ns.client_id: cfg.client_id = ns.client_id
    if ns.symbols: cfg.symbols = ns.symbols
    if ns.exchange_index: cfg.exchange_index = ns.exchange_index
    if ns.option_exchange: cfg.option_exchange = ns.option_exchange
    if ns.trading_class: cfg.trading_class = ns.trading_class
    if ns.expiry: cfg.expiry = ns.expiry
    if ns.n_strikes: cfg.n_strikes = ns.n_strikes
    if ns.throttle_ms: cfg.throttle_ms = ns.throttle_ms
    if ns.log_level: cfg.log_level = ns.log_level
    if ns.spot_fallback: cfg.spot_fallback = ns.spot_fallback
    if ns.no_probe_option: cfg.probe_option_for_spot = False
    return cfg


if __name__ == "__main__":
    ns = parse_args()
    cfg = load_config_from_yaml(ns.config)
    cfg = merge_cli(cfg, ns)
    setup_logging(cfg.log_level)

    logging.info(f"Starting collector with cfg={cfg}")
    try:
        Collector(cfg).run()
    except Exception as e:
        logging.exception(f"Collector crashed: {e}")
        sys.exit(1)