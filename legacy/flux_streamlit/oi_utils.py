# oi_utils.py
# --------------------------------------------
# Utilitaires pour relever l'Open Interest (OI) via IBKR
# et l'intégrer à la base SQLite (table oi_snapshots),
# puis fusionner l'OI récent dans un DataFrame par strike.

from __future__ import annotations

from datetime import datetime
import os
import time
import sqlite3
from typing import Optional, Dict, Tuple, List

import pandas as pd
import numpy as np


# ==================== Constantes ====================

GENERIC_TICKS_OI = "101"  # OI: callOpenInterest / putOpenInterest côté ib_insync


# ==================== Helpers génériques ====================

def _canon(p: str) -> str:
    """Chemin absolu/normalisé (évite les confusions de cwd)."""
    return os.path.abspath(os.path.expanduser(str(p)))


def _norm_expiry(expiry: str) -> str:
    """Force YYYYMMDD (digits only)."""
    return "".join(ch for ch in str(expiry) if ch.isdigit())


def _oi_field_for_right(right: str) -> str:
    """Champ OI à lire dans le ticker ib_insync selon CALL/PUT."""
    return "callOpenInterest" if str(right).upper().startswith("C") else "putOpenInterest"


def ensure_oi_schema(db_path: str) -> None:
    """Crée la table oi_snapshots (si absente) et les index."""
    con = sqlite3.connect(_canon(db_path), timeout=30.0)
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
          PRIMARY KEY (ts, symbol, trading_class, expiry, right, strike)
        );
        """)
        con.execute("CREATE INDEX IF NOT EXISTS idx_oi_last ON oi_snapshots(expiry, right, strike, ts);")
        con.commit()
    finally:
        con.close()


# ==================== Normalisation de l'univers ====================

def _normalize_universe(
    strikes_rights: pd.DataFrame,
    symbol_hint: Optional[str],
    tclass_hint: Optional[str],
) -> Tuple[pd.DataFrame, str, str, Dict[str, set]]:
    """
    Nettoie/normalise l'univers (symbol/right/strike) et renvoie :
    - DataFrame normalisé
    - symbol (SPX par défaut si manquant)
    - trading_class (SPXW par défaut si manquant)
    - wanted: dict {"C": set(strikes), "P": set(strikes)}
    """
    sr = strikes_rights.copy()

    # Ajoute les hints si colonnes absentes
    if "symbol" not in sr.columns and symbol_hint:
        sr["symbol"] = symbol_hint
    if "trading_class" not in sr.columns and tclass_hint:
        sr["trading_class"] = tclass_hint

    # Colonnes textuelles en MAJ
    for col in ["symbol", "trading_class"]:
        if col in sr.columns:
            sr[col] = sr[col].astype(str).str.upper()

    # Source du "côté" : right > type
    side_source = "right" if "right" in sr.columns else ("type" if "type" in sr.columns else None)
    if side_source:
        sr["right"] = sr[side_source].astype(str).str.upper().map(
            lambda v: "C" if v in ("C", "CALL") else ("P" if v in ("P", "PUT") else np.nan)
        )
    else:
        sr["right"] = np.nan

    # Strike numérique
    sr["strike"] = pd.to_numeric(sr.get("strike", np.nan), errors="coerce")

    # Filtrage robuste
    sr = sr.dropna(subset=["right", "strike"])
    sr = sr[sr["strike"] > 0]
    sr = sr.drop_duplicates(subset=["right", "strike"])

    # Valeurs par défaut si manquants
    symbol = (sr["symbol"].dropna().iloc[0] if "symbol" in sr.columns and not sr["symbol"].dropna().empty
              else (symbol_hint or "SPX"))
    tclass = (sr["trading_class"].dropna().iloc[0] if "trading_class" in sr.columns and not sr["trading_class"].dropna().empty
              else (tclass_hint or "SPXW"))

    wanted = {
        "C": set(sr.loc[sr["right"] == "C", "strike"].tolist()),
        "P": set(sr.loc[sr["right"] == "P", "strike"].tolist()),
    }
    return sr, symbol, tclass, wanted


# ==================== Snapshot OI via IBKR ====================

def _snapshot_oi_in_batches(
    ib,
    contracts: List,
    db_path: str,
    expiry: str,
    batch_size: int = 50,
    pause: float = 0.35,
    timeout_s: float = 8.0,
    retries: int = 1,
) -> int:
    """
    Snapshot OI par paquets (format INSERT OR REPLACE) – en réalité en STREAMING,
    car les generic ticks (101) ne supportent PAS snapshot=True.
    """
    ins_sql = """INSERT OR REPLACE INTO oi_snapshots
                 (ts, symbol, trading_class, expiry, right, strike, open_interest)
                 VALUES (?,?,?,?,?,?,?)"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    field_for = lambda r: "callOpenInterest" if str(r).upper().startswith("C") else "putOpenInterest"

    def one_pass(batch) -> List[Tuple]:
        # streaming ONLY for generic ticks
        reqs = []
        for c in batch:
            t = ib.reqMktData(
                c,
                genericTickList=GENERIC_TICKS_OI,  # "101"
                snapshot=False,
                regulatorySnapshot=False
            )
            reqs.append((c, t))
            ib.sleep(0.02)

        deadline = time.time() + timeout_s
        results = []

        try:
            while time.time() < deadline and reqs:
                done = []
                for c, t in reqs:
                    fld = field_for(c.right)
                    val = getattr(t, fld, None)
                    # ignore None/NaN, keep waiting
                    if val is not None:
                        try:
                            oi = int(val)
                        except Exception:
                            oi = None
                        if oi is not None:
                            results.append((c, oi))
                            done.append((c, t))
                for d in done:
                    reqs.remove(d)
                ib.sleep(0.25)
        finally:
            # cancel all subscriptions cleanly
            for _, t in reqs:
                try:
                    ib.cancelMktData(t)
                except Exception:
                    pass

        return results

    inserted = 0
    con = sqlite3.connect(_canon(db_path), timeout=30.0)
    try:
        # on peut tenter plusieurs marketDataType si besoin
        for mkt_type in (1, 2, 3, 4):
            try:
                ib.reqMarketDataType(mkt_type)
            except Exception:
                pass

            for i in range(0, len(contracts), batch_size):
                batch = contracts[i:i + batch_size]

                res: List[Tuple] = []
                for k in range(retries + 1):
                    try:
                        res = one_pass(batch)
                        break
                    except Exception:
                        ib.sleep(1.0 + 0.5 * k)

                if not res:
                    ib.sleep(pause)
                    continue

                for c, oi in res:
                    con.execute(ins_sql, (
                        now, c.symbol, c.tradingClass, expiry, c.right, float(c.strike), oi
                    ))
                con.commit()
                inserted += len(res)
                ib.sleep(pause)
    finally:
        con.close()
    return inserted



def _try_fetch_oi(ib, contract, timeout_s: float = 8.0, generic: str = GENERIC_TICKS_OI) -> Optional[int]:
    """
    Essaie de récupérer l'OI pour un contrat donné en variant:
    - marketDataType: 1 (live), 2 (frozen), 3 (delayed), 4 (delayed-frozen)
    - snapshot: True puis False (streaming court)
    Retourne un int (oi) ou None si échec.
    """
    field = _oi_field_for_right(contract.right)

    for mkt_type in (1, 2, 3, 4):
        try:
            ib.reqMarketDataType(mkt_type)
        except Exception:
            pass

        # snapshot=True
        try:
            t = ib.reqMktData(contract, "", snapshot=True, regulatorySnapshot=False, genericTickList=generic)
            deadline = time.time() + timeout_s
            while time.time() < deadline:
                ib.sleep(0.25)
                oi = getattr(t, field, None)
                if oi is not None:
                    ib.cancelMktData(t)
                    return int(oi or 0)
            ib.cancelMktData(t)
        except Exception:
            pass

        # snapshot=False (petit streaming)
        try:
            t = ib.reqMktData(contract, "", snapshot=False, regulatorySnapshot=False, genericTickList=generic)
            deadline = time.time() + min(2.0, timeout_s)
            while time.time() < deadline:
                ib.sleep(0.25)
                oi = getattr(t, field, None)
                if oi is not None:
                    ib.cancelMktData(t)
                    return int(oi or 0)
            ib.cancelMktData(t)
        except Exception:
            pass

    return None


def _qualify_contracts_for_exchange(ib, symbol: str, exp: str, tclass: str, exchange: str,
                                    wanted: Dict[str, set], debug: bool = False) -> List:
    """reqContractDetails par côté, filtre tradingClass & strike, renvoie la liste de contrats qualifiés."""
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
            print(f"[OI] reqCD {r} @ {exchange}: {len(cds) if cds else 0} résultats.")
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
    symbol_hint: Optional[str] = None,
    tclass_hint: Optional[str] = None,
    exchange: str = "CBOE",
    batch_size: int = 50,
    pause: float = 0.35,
    timeout_s: float = 8.0,
    retries: int = 1,
    try_smart_on_empty: bool = True,
    debug: bool = False,
) -> bool:
    """
    Snapshot OI pour une échéance :
      1) Univers depuis `trades` (ou DataFrame fourni)
      2) Résolution via reqContractDetails (évite "Ambiguous contract")
      3) Snapshot MktData '101' par lots + persist DB
      4) Fallback séquentiel + (optionnel) re-test en SMART si CBOE vide

    Renvoie True si >= 1 insertion en base, sinon False.
    """
    ensure_oi_schema(db_path)
    exp = _norm_expiry(expiry)

    # 1) Univers à partir de trades si non fourni
    if strikes_rights is None:
        con = sqlite3.connect(_canon(db_path))
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
        if debug: print("[OI] Univers vide (trades) → stop.")
        return False

    # 2) Normalisation stricte
    sr, symbol, tclass, wanted = _normalize_universe(strikes_rights, symbol_hint, tclass_hint)
    if (len(wanted["C"]) + len(wanted["P"])) == 0:
        if debug: print("[OI] Aucune (right, strike) exploitable après nettoyage → stop.")
        return False

    # 3) IB: connexion
    try:
        from ib_insync import IB
    except Exception as e:
        if debug: print("[OI] ib_insync indisponible:", e)
        return False

    ib = IB()
    try:
        ib.connect(host, port, clientId=client_id)
    except Exception as e:
        if debug: print("[OI] Connexion IB échouée:", e)
        return False

    inserted = 0
    try:
        # 3a) Qualification @ exchange demandé
        qualified = _qualify_contracts_for_exchange(ib, symbol, exp, tclass, exchange, wanted, debug=debug)
        if debug:
            print(f"[OI] Qualified contrats: {len(qualified)} | "
                  f"DB={_canon(db_path)} | exp={exp} | symbol={symbol} | tclass={tclass} | exch={exchange}")

        if qualified:
            # 4) Snapshot OI par lots + écriture DB
            inserted = _snapshot_oi_in_batches(
                ib=ib, contracts=qualified, db_path=db_path, expiry=exp,
                batch_size=batch_size, pause=pause, timeout_s=timeout_s, retries=retries
            )

            # 5) Fallback séquentiel si batching a tout raté
            if inserted == 0:
                ins_sql = """INSERT OR REPLACE INTO oi_snapshots
                             (ts, symbol, trading_class, expiry, right, strike, open_interest)
                             VALUES (?,?,?,?,?,?,?)"""
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                con = sqlite3.connect(_canon(db_path), timeout=30.0)
                try:
                    for c in qualified:
                        oi = _try_fetch_oi(ib, c, timeout_s=timeout_s, generic=GENERIC_TICKS_OI)
                        if oi is None:
                            continue
                        con.execute(ins_sql, (now, c.symbol, c.tradingClass, exp, c.right, float(c.strike), int(oi)))
                        inserted += 1
                        ib.sleep(pause)
                    con.commit()
                finally:
                    con.close()

        # 6) Si toujours rien ET on était en CBOE → retente SMART automatiquement
        if inserted == 0 and try_smart_on_empty and exchange.upper() == "CBOE":
            if debug: print("[OI] Aucun insert @ CBOE → retry @ SMART")
            qualified = _qualify_contracts_for_exchange(ib, symbol, exp, tclass, "SMART", wanted, debug=debug)
            if qualified:
                inserted = _snapshot_oi_in_batches(
                    ib=ib, contracts=qualified, db_path=db_path, expiry=exp,
                    batch_size=batch_size, pause=pause, timeout_s=timeout_s, retries=retries
                )
                if inserted == 0:
                    # fallback séquentiel @ SMART
                    ins_sql = """INSERT OR REPLACE INTO oi_snapshots
                                 (ts, symbol, trading_class, expiry, right, strike, open_interest)
                                 VALUES (?,?,?,?,?,?,?)"""
                    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    con = sqlite3.connect(_canon(db_path), timeout=30.0)
                    try:
                        for c in qualified:
                            oi = _try_fetch_oi(ib, c, timeout_s=timeout_s, generic=GENERIC_TICKS_OI)
                            if oi is None:
                                continue
                            con.execute(ins_sql, (now, c.symbol, c.tradingClass, exp, c.right, float(c.strike), int(oi)))
                            inserted += 1
                            ib.sleep(pause)
                        con.commit()
                    finally:
                        con.close()

        if debug: print(f"[OI] Inserted rows: {inserted}")
        return inserted > 0

    finally:
        try:
            ib.disconnect()
        except Exception:
            pass


# ==================== Lecture & fusion OI ====================

def latest_oi_frame(db_path: str, expiry: str) -> pd.DataFrame:
    """Renvoie l’OI le plus récent par (right, strike) pour une échéance."""
    ensure_oi_schema(db_path)
    exp = _norm_expiry(expiry)
    con = sqlite3.connect(_canon(db_path))
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


def merge_latest_oi_into_strike_metrics(
    strike_metrics: pd.DataFrame,
    db_path: str,
    expiry: str,
    option_type_col: str = "option_type",
    strike_col: str = "strike"
) -> pd.DataFrame:
    """
    Fusionne l’OI (le plus récent) dans un DataFrame par strike (CALL/PUT).
    strike_metrics doit avoir colonnes: [option_type_col, strike_col].
    """
    if strike_metrics is None or strike_metrics.empty:
        return strike_metrics
    oi = latest_oi_frame(db_path, expiry)
    if oi.empty:
        return strike_metrics
    merged = strike_metrics.merge(
        oi.rename(columns={"option_type": option_type_col, "strike": strike_col}),
        on=[option_type_col, strike_col],
        how="left"
    )
    return merged  # contient 'open_interest'


def get_oi_snapshot_meta(db_path: str, expiry: str) -> Tuple[Optional[str], int, int, int]:
    """Retourne (last_ts, n_total, n_calls, n_puts) pour une échéance."""
    exp = _norm_expiry(expiry)
    con = sqlite3.connect(_canon(db_path))
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
        return (r0.get("last_ts"), int(r0.get("n_total") or 0),
                int(r0.get("n_calls") or 0), int(r0.get("n_puts") or 0))
    finally:
        con.close()
