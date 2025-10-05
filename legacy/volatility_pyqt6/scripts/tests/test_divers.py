# tools/backfill_volatility_stats.py
import os, sqlite3
import numpy as np
import pandas as pd

DB_PATH = os.path.join("db", "market_data.db")
TICKER = "SPX"  # change si besoin

def list_tickers(conn):
    print("📦 Tickers présents dans volatility_stats (count):")
    for t, n in conn.execute("SELECT ticker, COUNT(*) FROM volatility_stats GROUP BY ticker ORDER BY 2 DESC"):
        print(f"   - {t}: {n}")

def ensure_table(conn):
    conn.execute("""
    CREATE TABLE IF NOT EXISTS volatility_stats(
      date TEXT NOT NULL,
      ticker TEXT NOT NULL,
      ma20_c2c REAL, ma252_c2c REAL,
      ma20_o2c REAL, ma252_o2c REAL,
      hv5 REAL, hv20 REAL, hv252 REAL,
      rank252_ma20_c2c REAL
    )
    """)
    conn.commit()

def dedupe_and_index(conn):
    # 1) Déduplique (garde le plus ancien rowid par (date,ticker))
    conn.execute("""
    DELETE FROM volatility_stats
    WHERE rowid NOT IN (
        SELECT MIN(rowid) FROM volatility_stats GROUP BY date, ticker
    )
    """)
    conn.commit()
    # 2) Index unique pour permettre ON CONFLICT(date,ticker)
    conn.execute("""
    CREATE UNIQUE INDEX IF NOT EXISTS volatility_stats_uq
    ON volatility_stats(date, ticker)
    """)
    conn.commit()

def compute_frame_from_prices(ticker: str, conn: sqlite3.Connection) -> pd.DataFrame:
    table_prices = f"{ticker.lower()}_data"
    dfp = pd.read_sql(f"SELECT date, open, close FROM {table_prices} ORDER BY date",
                      conn, parse_dates=["date"])
    if dfp.empty:
        raise RuntimeError(f"Aucune donnée dans la table {table_prices}")

    dfp = dfp.sort_values("date")
    c2c = dfp["close"].pct_change()
    o2c = (dfp["close"] - dfp["open"]) / dfp["open"]

    ma = lambda s, n: s.rolling(n).mean()
    hv = lambda s, n: s.rolling(n).std() * np.sqrt(252) * 100
    rank_pct = lambda s, w=252: s.rolling(w).apply(
        lambda x: (x[-1]-x.min())/(x.max()-x.min()) if (x.max()-x.min()) else np.nan, raw=True
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

def upsert(df: pd.DataFrame, conn: sqlite3.Connection):
    rows = [tuple(x) for x in df.to_numpy()]
    conn.executemany("""
    INSERT INTO volatility_stats
    (date,ticker,ma20_c2c,ma252_c2c,ma20_o2c,ma252_o2c,hv5,hv20,hv252,rank252_ma20_c2c)
    VALUES (?,?,?,?,?,?,?,?,?,?)
    ON CONFLICT(date, ticker) DO UPDATE SET
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

def main():
    if not os.path.exists(DB_PATH):
        raise SystemExit(f"❌ DB introuvable: {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)
    try:
        # Assure table + index unique
        ensure_table(conn)
        dedupe_and_index(conn)

        # Info avant
        list_tickers(conn)

        print(f"\n⏳ Calcul & upsert de {TICKER} depuis {TICKER.lower()}_data ...")
        df = compute_frame_from_prices(TICKER, conn)

        # filtre les lignes totalement vides
        df = df.dropna(how="all", subset=[
            "ma20_c2c","ma252_c2c","ma20_o2c","ma252_o2c","hv5","hv20","hv252","rank252_ma20_c2c"
        ])
        upsert(df, conn)
        print(f"✅ Upsert terminé pour {TICKER}: {len(df)} lignes.")

        print("\n📦 Après upsert :")
        list_tickers(conn)
    finally:
        conn.close()

if __name__ == "__main__":
    main()
