"""
Open Interest Fetcher
Adapted from oi_utils.py - Fetch OI + IV + Greeks from IBKR and store in SQLite
"""
from datetime import datetime
import time
import sqlite3
from typing import Optional, Dict, Tuple, List

import pandas as pd
import numpy as np

GENERIC_TICKS_OI = "101"  # OI: callOpenInterest / putOpenInterest
GENERIC_TICKS_FULL = "101,106"  # 101=OI, 106=Greeks


def ensure_oi_schema(db_path: str) -> None:
    """Create oi_snapshots table with full data columns"""
    con = sqlite3.connect(db_path, timeout=30.0)
    try:
        con.execute("""
        CREATE TABLE IF NOT EXISTS oi_snapshots (
          ts TEXT NOT NULL,
          symbol TEXT NOT NULL,
          trading_class TEXT,
          expiry TEXT NOT NULL,
          right TEXT NOT NULL,
          strike REAL NOT NULL,
          open_interest INTEGER,
          iv REAL,
          delta REAL,
          gamma REAL,
          vega REAL,
          theta REAL,
          bid REAL,
          ask REAL,
          last REAL,
          volume INTEGER,
          PRIMARY KEY (ts, symbol, trading_class, expiry, right, strike)
        );
        """)
        con.execute("CREATE INDEX IF NOT EXISTS idx_oi_last ON oi_snapshots(expiry, right, strike, ts);")
        con.commit()
    finally:
        con.close()


def _norm_expiry(expiry: str) -> str:
    """Force YYYYMMDD (digits only)"""
    return "".join(ch for ch in str(expiry) if ch.isdigit())


def _oi_field_for_right(right: str) -> str:
    """Field to read from ticker based on CALL/PUT"""
    return "callOpenInterest" if str(right).upper().startswith("C") else "putOpenInterest"


def _normalize_universe(
    strikes_rights: pd.DataFrame,
    symbol_hint: Optional[str],
    tclass_hint: Optional[str],
) -> Tuple[pd.DataFrame, str, str, Dict[str, set]]:
    """
    Clean/normalize universe (symbol/right/strike)
    Returns: (df_normalized, symbol, trading_class, wanted_dict)
    """
    sr = strikes_rights.copy()

    # Add hints if columns missing
    if "symbol" not in sr.columns and symbol_hint:
        sr["symbol"] = symbol_hint
    if "trading_class" not in sr.columns and tclass_hint:
        sr["trading_class"] = tclass_hint

    # Uppercase text columns
    for col in ["symbol", "trading_class"]:
        if col in sr.columns:
            sr[col] = sr[col].astype(str).str.upper()

    # Right/type mapping
    side_source = "right" if "right" in sr.columns else ("type" if "type" in sr.columns else None)
    if side_source:
        sr["right"] = sr[side_source].astype(str).str.upper().map(
            lambda v: "C" if v in ("C", "CALL") else ("P" if v in ("P", "PUT") else np.nan)
        )
    else:
        sr["right"] = np.nan

    # Numeric strike
    sr["strike"] = pd.to_numeric(sr.get("strike", np.nan), errors="coerce")

    # Filtering
    sr = sr.dropna(subset=["right", "strike"])
    sr = sr[sr["strike"] > 0]
    sr = sr.drop_duplicates(subset=["right", "strike"])

    # Defaults
    symbol = (sr["symbol"].dropna().iloc[0] if "symbol" in sr.columns and not sr["symbol"].dropna().empty
              else (symbol_hint or "SPX"))
    tclass = (
        sr["trading_class"].dropna().iloc[0] if "trading_class" in sr.columns and not sr["trading_class"].dropna().empty
        else (tclass_hint or "SPXW"))

    wanted = {
        "C": set(sr.loc[sr["right"] == "C", "strike"].tolist()),
        "P": set(sr.loc[sr["right"] == "P", "strike"].tolist()),
    }
    return sr, symbol, tclass, wanted


def _try_fetch_oi_full(ib, contract, timeout_s: float = 8.0) -> Optional[Dict]:
    """
    Fetch OI + IV + Greeks for a contract
    Returns dict with all data or None if failed
    """
    field_oi = _oi_field_for_right(contract.right)

    for mkt_type in (1, 2, 3, 4):
        try:
            ib.reqMarketDataType(mkt_type)
        except Exception:
            pass

        try:
            # Request market data with OI + Greeks
            t = ib.reqMktData(
                contract,
                genericTickList=GENERIC_TICKS_FULL,
                snapshot=False,
                regulatorySnapshot=False
            )
            deadline = time.time() + timeout_s

            while time.time() < deadline:
                ib.sleep(0.25)

                # Check if we have OI (minimum required)
                oi = getattr(t, field_oi, None)
                if oi is not None:
                    # Extract all available data
                    data = {
                        'open_interest': int(oi or 0),
                        'iv': getattr(t, 'impliedVolatility', None),
                        'bid': getattr(t, 'bid', None),
                        'ask': getattr(t, 'ask', None),
                        'last': getattr(t, 'last', None),
                        'volume': getattr(t, 'volume', None),
                    }

                    # Extract Greeks if available
                    if hasattr(t, 'modelGreeks') and t.modelGreeks:
                        data['delta'] = getattr(t.modelGreeks, 'delta', None)
                        data['gamma'] = getattr(t.modelGreeks, 'gamma', None)
                        data['vega'] = getattr(t.modelGreeks, 'vega', None)
                        data['theta'] = getattr(t.modelGreeks, 'theta', None)
                    else:
                        data['delta'] = None
                        data['gamma'] = None
                        data['vega'] = None
                        data['theta'] = None

                    ib.cancelMktData(t)
                    return data

            ib.cancelMktData(t)
        except Exception:
            pass

    return None


def _qualify_contracts_for_exchange(
    ib,
    symbol: str,
    exp: str,
    tclass: str,
    exchange: str,
    wanted: Dict[str, set],
    debug: bool = False
) -> List:
    """Qualify contracts via reqContractDetails"""
    from ib_insync import Option

    qualified = []
    for r in ("C", "P"):
        if not wanted[r]:
            continue

        req = Option(
            symbol=symbol,
            lastTradeDateOrContractMonth=exp,
            right=r,
            exchange=exchange,
            currency="USD",
            multiplier="100",
            tradingClass=tclass
        )

        cds = ib.reqContractDetails(req)
        if debug:
            print(f"[OI] reqContractDetails {r} @ {exchange}: {len(cds) if cds else 0} results")

        if not cds:
            continue

        for cd in cds:
            c = cd.contract
            try:
                stk = float(c.strike)
            except Exception:
                continue

            if c.tradingClass == tclass and stk in wanted[r]:
                qualified.append(c)

    return qualified


def fetch_oi_snapshot_for_expiry(
    db_path: str,
    expiry: str,
    host: str = "127.0.0.1",
    port: int = 7497,
    client_id: int = 21,
    strikes_rights: Optional[pd.DataFrame] = None,
    symbol_hint: Optional[str] = "SPX",
    tclass_hint: Optional[str] = "SPXW",
    exchange: str = "CBOE",
    pause: float = 0.35,
    timeout_s: float = 8.0,
    try_smart_on_empty: bool = True,
    debug: bool = False,
    progress_callback=None
) -> Tuple[bool, int]:
    """
    Fetch OI snapshot for an expiry
    Returns: (success: bool, inserted_count: int)
    """
    ensure_oi_schema(db_path)
    exp = _norm_expiry(expiry)

    # 1) Get universe from trades if not provided
    if strikes_rights is None:
        con = sqlite3.connect(db_path)
        try:
            q = """
                SELECT DISTINCT symbol, trading_class, right, strike
                FROM trades
                WHERE REPLACE(REPLACE(REPLACE(COALESCE(expiry,''),'-',''),'/',''),'.','') = ?
            """
            strikes_rights = pd.read_sql(q, con, params=[exp])
        finally:
            con.close()

    if strikes_rights is None or strikes_rights.empty:
        if progress_callback:
            progress_callback("Empty universe (no trades)")
        return False, 0

    # 2) Normalize
    sr, symbol, tclass, wanted = _normalize_universe(strikes_rights, symbol_hint, tclass_hint)

    total_wanted = len(wanted["C"]) + len(wanted["P"])
    if total_wanted == 0:
        if progress_callback:
            progress_callback("No valid strikes after normalization")
        return False, 0

    if progress_callback:
        progress_callback(f"Universe: {len(wanted['C'])} calls, {len(wanted['P'])} puts")

    # 3) Connect IB
    try:
        from ib_insync import IB
    except Exception as e:
        if progress_callback:
            progress_callback(f"ib_insync unavailable: {e}")
        return False, 0

    ib = IB()
    try:
        ib.connect(host, port, clientId=client_id)
        if progress_callback:
            progress_callback(f"Connected to IB @ {host}:{port}")
    except Exception as e:
        if progress_callback:
            progress_callback(f"Connection failed: {e}")
        return False, 0

    inserted = 0

    try:
        # 4) Qualify contracts
        if progress_callback:
            progress_callback(f"Qualifying contracts @ {exchange}...")

        qualified = _qualify_contracts_for_exchange(ib, symbol, exp, tclass, exchange, wanted, debug=debug)

        if progress_callback:
            progress_callback(f"Qualified: {len(qualified)} contracts")

        if qualified:
            # 5) Fetch OI + IV + Greeks sequentially
            ins_sql = """INSERT OR REPLACE INTO oi_snapshots
                         (ts, symbol, trading_class, expiry, right, strike, open_interest,
                          iv, delta, gamma, vega, theta, bid, ask, last, volume)
                         VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"""
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            con = sqlite3.connect(db_path, timeout=30.0)
            try:
                for i, c in enumerate(qualified):
                    if progress_callback and i % 10 == 0:
                        progress_callback(f"Fetching OI+IV: {i + 1}/{len(qualified)}")

                    data = _try_fetch_oi_full(ib, c, timeout_s=timeout_s)
                    if data is None:
                        continue

                    con.execute(ins_sql, (
                        now, c.symbol, c.tradingClass, exp, c.right, float(c.strike),
                        data['open_interest'],
                        data.get('iv'),
                        data.get('delta'),
                        data.get('gamma'),
                        data.get('vega'),
                        data.get('theta'),
                        data.get('bid'),
                        data.get('ask'),
                        data.get('last'),
                        data.get('volume')
                    ))
                    inserted += 1
                    ib.sleep(pause)

                con.commit()
            finally:
                con.close()

        # 6) Fallback to SMART if CBOE empty
        if inserted == 0 and try_smart_on_empty and exchange.upper() == "CBOE":
            if progress_callback:
                progress_callback("No data from CBOE, trying SMART...")

            qualified = _qualify_contracts_for_exchange(ib, symbol, exp, tclass, "SMART", wanted, debug=debug)

            if qualified:
                ins_sql = """INSERT OR REPLACE INTO oi_snapshots
                             (ts, symbol, trading_class, expiry, right, strike, open_interest,
                              iv, delta, gamma, vega, theta, bid, ask, last, volume)
                             VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"""
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                con = sqlite3.connect(db_path, timeout=30.0)
                try:
                    for i, c in enumerate(qualified):
                        if progress_callback and i % 10 == 0:
                            progress_callback(f"Fetching OI+IV (SMART): {i + 1}/{len(qualified)}")

                        # ✅ FIXED: Use _try_fetch_oi_full instead of _try_fetch_oi
                        data = _try_fetch_oi_full(ib, c, timeout_s=timeout_s)
                        if data is None:
                            continue

                        con.execute(ins_sql, (
                            now, c.symbol, c.tradingClass, exp, c.right, float(c.strike),
                            data['open_interest'],
                            data.get('iv'),
                            data.get('delta'),
                            data.get('gamma'),
                            data.get('vega'),
                            data.get('theta'),
                            data.get('bid'),
                            data.get('ask'),
                            data.get('last'),
                            data.get('volume')
                        ))
                        inserted += 1
                        ib.sleep(pause)

                    con.commit()
                finally:
                    con.close()

        if progress_callback:
            progress_callback(f"Completed: {inserted} OI values inserted")

        return inserted > 0, inserted

    finally:
        try:
            ib.disconnect()
        except Exception:
            pass


def latest_oi_frame(db_path: str, expiry: str) -> pd.DataFrame:
    """Get latest OI per (right, strike) for an expiry"""
    ensure_oi_schema(db_path)
    exp = _norm_expiry(expiry)

    con = sqlite3.connect(db_path)
    try:
        q = """
        WITH latest AS (
          SELECT right, strike, MAX(ts) AS ts
          FROM oi_snapshots
          WHERE expiry = ?
          GROUP BY right, strike
        )
        SELECT o.right, o.strike, o.open_interest
        FROM oi_snapshots o
        JOIN latest l
          ON o.right = l.right AND o.strike = l.strike AND o.ts = l.ts
        WHERE o.expiry = ?
        """
        df = pd.read_sql(q, con, params=[exp, exp])

        if df.empty:
            return df

        df["option_type"] = np.where(df["right"].astype(str).str.upper() == "C", "CALL", "PUT")
        return df[["option_type", "strike", "open_interest"]]

    finally:
        con.close()


def get_oi_snapshot_meta(db_path: str, expiry: str) -> Tuple[Optional[str], int, int, int]:
    """Get (last_ts, n_total, n_calls, n_puts) for an expiry"""
    exp = _norm_expiry(expiry)
    con = sqlite3.connect(db_path)

    try:
        row = pd.read_sql("""
            SELECT 
              MAX(ts) AS last_ts,
              COUNT(*) AS n_total,
              SUM(CASE WHEN UPPER(right)='C' THEN 1 ELSE 0 END) AS n_calls,
              SUM(CASE WHEN UPPER(right)='P' THEN 1 ELSE 0 END) AS n_puts
            FROM oi_snapshots
            WHERE expiry = ?
        """, con, params=[exp])

        if row.empty:
            return None, 0, 0, 0

        r0 = row.iloc[0]
        return (
            r0.get("last_ts"),
            int(r0.get("n_total") or 0),
            int(r0.get("n_calls") or 0),
            int(r0.get("n_puts") or 0)
        )
    finally:
        con.close()


def get_oi_delta_between_snapshots(
    db_path: str,
    expiry: str,
    snapshot1_ts: Optional[str] = None,
    snapshot2_ts: Optional[str] = None
) -> pd.DataFrame:
    """
    Calculate OI changes between two snapshots
    If timestamps not provided, uses the two most recent snapshots
    """
    exp = _norm_expiry(expiry)
    conn = sqlite3.connect(db_path)

    try:
        # Get timestamps if not provided
        if snapshot1_ts is None or snapshot2_ts is None:
            ts_query = """
                SELECT DISTINCT ts 
                FROM oi_snapshots 
                WHERE expiry = ?
                ORDER BY ts DESC
                LIMIT 2
            """
            timestamps = pd.read_sql(ts_query, conn, params=[exp])

            if len(timestamps) < 2:
                return pd.DataFrame()

            snapshot2_ts = timestamps.iloc[0]['ts']
            snapshot1_ts = timestamps.iloc[1]['ts']

        # Get both snapshots
        query = """
            SELECT 
                ts, right, strike, open_interest
            FROM oi_snapshots
            WHERE expiry = ? AND ts IN (?, ?)
            ORDER BY right, strike, ts
        """

        df = pd.read_sql(query, conn, params=[exp, snapshot1_ts, snapshot2_ts])

        if df.empty:
            return df

        # Pivot to compare
        df_old = df[df['ts'] == snapshot1_ts][['right', 'strike', 'open_interest']].rename(
            columns={'open_interest': 'oi_old'}
        )
        df_new = df[df['ts'] == snapshot2_ts][['right', 'strike', 'open_interest']].rename(
            columns={'open_interest': 'oi_new'}
        )

        # Merge
        delta = pd.merge(df_old, df_new, on=['right', 'strike'], how='outer').fillna(0)

        # Calculate delta
        delta['delta_oi'] = delta['oi_new'] - delta['oi_old']
        delta['delta_oi_pct'] = (
            (delta['delta_oi'] / delta['oi_old'] * 100)
            .replace([float('inf'), float('-inf')], 0)
            .fillna(0)
        )

        # Add option_type
        delta['option_type'] = delta['right'].map({'C': 'CALL', 'P': 'PUT'})

        # Add timestamps
        delta['snapshot_old'] = snapshot1_ts
        delta['snapshot_new'] = snapshot2_ts

        # Sort by absolute delta
        delta['abs_delta'] = delta['delta_oi'].abs()
        delta = delta.sort_values('abs_delta', ascending=False)

        return delta[[
            'snapshot_old', 'snapshot_new', 'option_type', 'right', 'strike',
            'oi_old', 'oi_new', 'delta_oi', 'delta_oi_pct'
        ]]

    finally:
        conn.close()


def get_available_oi_snapshots(db_path: str, expiry: str) -> List[str]:
    """Get list of available snapshot timestamps for an expiry"""
    exp = _norm_expiry(expiry)
    conn = sqlite3.connect(db_path)

    try:
        query = """
            SELECT DISTINCT ts
            FROM oi_snapshots
            WHERE expiry = ?
            ORDER BY ts DESC
        """
        result = pd.read_sql(query, conn, params=[exp])
        return result['ts'].tolist()
    finally:
        conn.close()