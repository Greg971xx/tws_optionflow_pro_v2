
# tools/backfill_volatility_stats.py
import os
import sqlite3
from typing import List

import numpy as np
import pandas as pd


DB_PATH = os.path.join("db", "market_data.db")
TICKER = "SPX"  # tu peux changer pour un autre sous-jacent
ALIASES = ["SPX", "^GSPC", "SPX Index", "INX"]  # alias possibles en base


def ensure_table(conn: sqlite3.Connection):
    conn.execute("""
    CREATE TABLE IF NOT EXISTS volatility_stats(
      date TEXT NOT NULL,
      ticker TEXT NOT NULL,
      ma20_c2c REAL, ma252_c2c REAL,
      ma20_o2c REAL, ma252_o2c REAL,
      hv5 REAL, hv20 REAL, hv252 REAL,
      rank252_ma20_c2c REAL,
      PRIMARY KEY(date, ticker)
    )
    """)
    conn.commit()


def list_tickers(conn: sqlite3.Connection):
    cur = conn.cursor()
    rows = cur.execute(
        "SELECT ticker, COUNT(*) FROM volatility_stats GROUP BY ticker ORDER BY 2 DESC"
    ).fetchall()
    if not rows:
        print("ℹ️  Table 'volatility_stats' vide (ou inexistante).")
    else:
        print("📦 Tickers présents dans volatility_stats (count):")
        for t, n in rows:
            print(f"   - {t}: {n}")


def choose_existing_alias(conn: sqlite3.Connection, aliases: List[str]) -> str | None:
    qmarks = ",".join("?" * len(aliases))
    rows = conn.execute(
        f"SELECT DISTINCT ticker FROM volatility_stats WHERE ticker IN ({qmarks})",
        aliases,
    ).fetchall()
    return rows[0][0] if rows else None


def compute_frame_from_prices(ticker: str, conn: sqlite3.Connection) -> pd.DataFrame:
    table_prices = f"{ticker.lower()}_data"
    dfp = pd.read_sql(f"SELECT date, open, close FROM {table_prices} ORDER BY date",
                      conn, parse_dates=["date"])
    if dfp.empty:
        raise RuntimeError(f"Aucune donnée dans la table {table_prices}")

    dfp = dfp.sort_values("date")
    c2c = dfp["close"].pct_change()
    o2c = (dfp["close"] - dfp["open"]) / dfp["open"]

    def ma(s, n): return s.rolling(n).mean()
    def hv(s, n): return s.rolling(n).std() * np.sqrt(252) * 100
    def rank_pct(s, w=252):
        return s.rolling(w).apply(
            lambda x: (x[-1] - x.min()) / (x.max() - x.min()) if (x.max() - x.min()) else np.nan,
            raw=True
        )

    out = pd.DataFrame({
        "date": dfp["date"].dt.strftime("%Y-%m-%d"),
        "ticker": ticker,
        "ma20_c2c": ma(c2c, 20),
        "ma252_c2c": ma(c2c, 252),
        "ma20_o2c": ma(o2c, 20),
        "ma252_o2c": ma(o2c, 252),
        "hv5": hv(c2c, 5),
        "hv20": hv(c2c, 20),
        "hv252": hv(c2c, 252),
    })
    out["rank252_ma20_c2c"] = rank_pct(out["ma20_c2c"], 252)
    return out


def upsert_vol_stats(df: pd.DataFrame, conn: sqlite3.Connection):
    rows = [tuple(x) for x in df.to_numpy()]
    conn.executemany("""
    INSERT INTO volatility_stats
    (date,ticker,ma20_c2c,ma252_c2c,ma20_o2c,ma252_o2c,hv5,hv20,hv252,rank252_ma20_c2c)
    VALUES (?,?,?,?,?,?,?,?,?,?)
    ON CONFLICT(date,ticker) DO UPDATE SET
      ma20_c2c=excluded.ma20_c2c,
      ma252_c2c=excluded.ma252_c2c,
      ma20_o2c=excluded.ma20_o2c,
      ma252_o2c=excluded.ma252_o2c,
      hv5=excluded.hv5,
      hv20=excluded.hv20,
      hv252=excluded.hv252,
      rank252_ma20_c2c=excluded.rank252_ma20_c2c
    """, rows)
    conn.commit()
    print(f"✅ Upsert terminé : {len(rows)} lignes écrites/actualisées pour {df['ticker'].iloc[0]}.")


def main():
    if not os.path.exists(DB_PATH):
        raise SystemExit(f"❌ DB introuvable: {DB_PATH} (lance depuis la racine du projet)")

    conn = sqlite3.connect(DB_PATH)
    try:
        ensure_table(conn)
        list_tickers(conn)

        # 1) Si un alias de SPX existe déjà en base, on l’utilise tel quel
        existing = choose_existing_alias(conn, ALIASES)
        if existing:
            print(f"\nℹ️  Données déjà présentes pour un alias : '{existing}'. Rien à faire.")
            return

        # 2) Sinon on backfill à partir de la table de prix {spx_data}
        print(f"\n⏳ Backfill de volatility_stats pour '{TICKER}' depuis {TICKER.lower()}_data ...")
        df = compute_frame_from_prices(TICKER, conn)

        if df.dropna(how="all", subset=[
            "ma20_c2c","ma252_c2c","ma20_o2c","ma252_o2c","hv5","hv20","hv252","rank252_ma20_c2c"
        ]).empty:
            raise RuntimeError("Les calculs retournent tout NaN — vérifie les données de prix.")

        upsert_vol_stats(df, conn)

        # Re-liste pour confirmer
        print("\n📦 Après backfill :")
        list_tickers(conn)

    finally:
        conn.close()


if __name__ == "__main__":
    main()
