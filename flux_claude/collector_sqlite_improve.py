# =============================================
# File: collector_sqlite_improve.py
# Purpose: Collect SPX(/SPXW) options ticks + Greeks into SQLite avec volume *réaliste*
# Notes:
#   - Calcule un vrai trade size par tick: Δ(volume) par contrat (fallback sur lastSize si nouveau trade).
#   - Stocke à la fois `trade_qty` et `qty = trade_qty` (pour que ton app continue de sommer `qty`).
#   - Migrations SQLite douces (ajoute colonnes greeks + trade_qty si manquantes).
#   - Spot robuste (index -> option undPrice -> fallback).
#   - Throttle + dédoublonnage léger.
# Requirements: ib_insync, pandas, nest_asyncio (facultatif), pyyaml (facultatif pour --config)
# =============================================

# --- S'assurer qu'une boucle asyncio existe avant d'importer ib_insync/eventkit ---
import asyncio
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())
try:
    import nest_asyncio; nest_asyncio.apply()
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
from datetime import datetime, date
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
    client_id: int = 10
    symbols: str = "SPX"           # comma-separated allowed
    exchange_index: str = "CBOE"   # index exchange preference
    option_exchange: str = "CBOE"  # options exchange
    trading_class: Optional[str] = None  # "SPX" (monthly) ou "SPXW" (weeklies). None -> auto selon le jour.
    expiry: str = "auto"           # "YYYYMMDD" ou "auto"
    n_strikes: int = 15            # total strikes autour de l'ATM
    throttle_ms: int = 400         # ms mini entre deux writes sur un même contrat
    log_level: str = "INFO"
    spot_fallback: float = 6615
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
    today = date.today()
    parsed = []
    for e in expirations:
        try:
            parsed.append(datetime.strptime(e, "%Y%m%d").date())
        except Exception:
            continue
    parsed.sort()
    for d in parsed:
        if d >= today:
            return d.strftime("%Y%m%d")
    return parsed[-1].strftime("%Y%m%d") if parsed else ""


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
                break
            except Exception as e:
                self.log.debug(f"Index qualify failed for {symbol}@{ex}: {e}")
        if idx is None:
            idx = Index(symbol, exchange=self.cfg.exchange_index or "CBOE")
            self.ib.qualifyContracts(idx)

        # Paramètres options
        params = self.ib.reqSecDefOptParams(idx.symbol, "", idx.secType, idx.conId)
        node = None
        for p in params:
            if (p.tradingClass or "").upper().startswith("SPXW"):
                node = p; break
        if node is None and params:
            node = params[0]
        if node is None:
            raise RuntimeError("Could not retrieve SecDefOptParams for index")

        expiry = 20250916
        if expiry == "auto" or not expiry:
            expiry = pick_nearest_expiry(node.expirations)
        tclass = (self.cfg.trading_class or suggest_trading_class(expiry)).upper()

        # Spot via index
        spot = None
        try:
            t_index = self.ib.reqMktData(idx, "", snapshot=True, regulatorySnapshot=False)
            t_end = time.time() + 10
            while time.time() < t_end and (spot is None or spot <= 0):
                self.ib.sleep(0.3)
                for px in [t_index.last, t_index.marketPrice()]:
                    if px is not None and not pd.isna(px) and float(px) > 0:
                        spot = float(px); break
        except Exception as e:
            self.log.debug(f"reqMktData on index failed: {e}")

        # Strikes disponibles
        strikes = []
        try:
            strikes = sorted([float(s) for s in node.strikes if isinstance(s, (float, int)) or (isinstance(s, str) and s.replace('.','',1).isdigit())])
        except Exception:
            pass

        # Spot via option (undPrice) si nécessaire
        if (spot is None or spot <= 0) and self.cfg.probe_option_for_spot and strikes:
            k_mid = strikes[len(strikes)//2]
            probe = Option(symbol=symbol, lastTradeDateOrContractMonth=expiry, strike=float(k_mid),
                           right="C", exchange=self.cfg.option_exchange or "CBOE",
                           currency="USD", multiplier="100", tradingClass=tclass)
            try:
                self.ib.qualifyContracts(probe)
                t = self.ib.reqMktData(probe, "", snapshot=True, regulatorySnapshot=False)
                t_end = time.time() + 8
                while time.time() < t_end and (spot is None or spot <= 0):
                    self.ib.sleep(0.3)
                    mg = getattr(t, "modelGreeks", None)
                    if mg and getattr(mg, "undPrice", None):
                        spot = float(mg.undPrice)
                        break
            except Exception as e:
                self.log.debug(f"probe option for spot failed: {e}")

        # Fallbacks finaux
        if (spot is None or spot <= 0):
            if self.cfg.spot_fallback and self.cfg.spot_fallback > 0:
                spot = float(self.cfg.spot_fallback)
                self.log.warning(f"Using configured spot_fallback={spot}")
            elif strikes:
                spot = float(strikes[len(strikes)//2])
                self.log.warning(f"Using median strike {spot} as spot proxy (no market data)")
            else:
                raise RuntimeError("Unable to fetch index spot to pick strikes")

        # Choisir n_strikes autour de l'ATM
        chosen: List[float] = []
        if strikes:
            nearest = min(strikes, key=lambda k: (abs(k-spot), k))
            idx_near = strikes.index(nearest)
            half = max(self.cfg.n_strikes // 2, 1)
            lo = max(0, idx_near - half)
            hi = min(len(strikes), idx_near + half + 1)
            chosen = strikes[lo:hi]
        else:
            step = max(round(spot * 0.005, 1), 1.0)
            base = int(spot // step) * step
            half = max(self.cfg.n_strikes // 2, 1)
            chosen = [base + i*step for i in range(-half, half+1)]

        contracts: List[Option] = []
        for right in ("C", "P"):
            for k in chosen:
                contracts.append(Option(symbol=symbol, lastTradeDateOrContractMonth=expiry,
                                        strike=float(k), right=right, exchange=self.cfg.option_exchange or "CBOE",
                                        currency="USD", multiplier="100", tradingClass=tclass))

        # Qualifier les contrats
        qual: List[Option] = []
        for c in contracts:
            try:
                q = self.ib.qualifyContracts(c)
                if q:
                    qual.append(q[0])
            except Exception as e:
                self.log.debug(f"Qualify failed for {c}: {e}")
        if not qual:
            raise RuntimeError("No options qualified (check exchange/trading_class/permissions)")

        self.log.info(f"Universe built: {symbol} {expiry} {tclass} x {len(qual)} contracts (spot≈{spot:.2f})")
        return expiry, tclass, qual

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
        for c in contracts:
            try:
                self.ib.reqMktData(c, "", snapshot=False, regulatorySnapshot=False)
            except Exception as e:
                self.log.debug(f"reqMktData live failed for {c}: {e}")

        self.log.info("Streaming... (Ctrl+C to stop)")
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
        bid  = _as_float(getattr(t, "bid", None))
        ask  = _as_float(getattr(t, "ask", None))
        lastSize = _as_int(getattr(t, "lastSize", None))
        volume   = _as_int(getattr(t, "volume", None))

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

        mid = (bid + ask)/2.0 if (bid and ask and bid>0 and ask>0) else None
        spread = (ask - bid) if (bid and ask and bid>0 and ask>0) else None

        est, conf, sign = classify_trade(last, bid, ask)
        signed_qty = int((trade_qty or 0) * sign) if sign else 0

        # Greeks & spot
        iv = delta = gamma = vega = theta = None
        mg = getattr(t, "modelGreeks", None)
        if mg:
            iv = _as_float(getattr(mg, "impliedVol", None))
            delta = _as_float(getattr(mg, "delta", None))
            gamma = _as_float(getattr(mg, "gamma", None))
            vega  = _as_float(getattr(mg, "vega", None))
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
            int(trade_qty or 0),   # qty -> trade size réaliste
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
            int(trade_qty or 0)    # trade_qty
        )
        self.writer.add(row)
        st.last_sig = sig
        st.last_write_ms = now_ms

    def _update_state_only(self, st: "ContractState", t):
        """Met à jour prev_volume/prev_last même si on ne write pas à cause du throttle."""
        last = _as_float(getattr(t, "last", None))
        volume   = _as_int(getattr(t, "volume", None))
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
    p = argparse.ArgumentParser(description="Collect SPX(/SPXW) options ticks with Greeks into SQLite (realistic volumes)")
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
