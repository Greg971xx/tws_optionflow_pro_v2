import sqlite3
import pandas as pd
import numpy as np
import os

def compute_rank_percentile(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    rank = (series - series.min()) / (series.max() - series.min())
    percentile = series.expanding().apply(lambda x: (x < x[-1]).mean(), raw=True)
    return rank, percentile

def get_all_tickers(db_path: str) -> list:
    """Retourne tous les tickers présents en base ayant une table *_data"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    conn.close()
    return [t.replace("_data", "").upper() for t in tables if t.endswith("_data")]

def update_volatility_stats(ticker: str, db_path: str):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        df = pd.read_sql(f"SELECT * FROM {ticker.lower()}_data", conn, parse_dates=["date"])
    except Exception as e:
        print(f"❌ {ticker} : erreur de lecture - {e}")
        conn.close()
        return

    df = df.sort_values("date")

    # ➕ Volatilité journalière
    df["vr_c2c_abs"] = df["close"].pct_change().abs()
    df["vr_o2c_abs"] = ((df["close"] - df["open"]) / df["open"]).abs()

    df["vr_c2c"] = df["close"].pct_change()
    df["vr_o2c"] = ((df["close"] - df["open"]) / df["open"])

    df["vr_high_low"] = (df["high"] - df["low"]) / df["open"]

    hv_periods = [5, 20, 60, 120, 252]
    ma_periods = [20, 60, 120, 252]

    for p in hv_periods:
        df[f"hv{p}"] = df["vr_c2c"].rolling(p).std() * np.sqrt(252)
        df[f"rank_hv{p}"], df[f"percentile_hv{p}"] = compute_rank_percentile(df[f"hv{p}"])

    for p in ma_periods:
        df[f"ma{p}_c2c"] = df["vr_c2c"].rolling(p).mean()
        df[f"ma{p}_o2c"] = df["vr_o2c"].rolling(p).mean()
        df[f"rank_ma{p}_c2c"], df[f"percentile_ma{p}_c2c"] = compute_rank_percentile(df[f"ma{p}_c2c"])
        df[f"rank_ma{p}_o2c"], df[f"percentile_ma{p}_o2c"] = compute_rank_percentile(df[f"ma{p}_o2c"])

    df["rank_vr_c2c"], df["percentile_vr_c2c"] = compute_rank_percentile(df["vr_c2c"])
    df["rank_vr_o2c"], df["percentile_vr_o2c"] = compute_rank_percentile(df["vr_o2c"])
    df["rank_vr_high_low"], df["percentile_vr_high_low"] = compute_rank_percentile(df["vr_high_low"])

    # Création de la table avec toutes les colonnes possibles
    column_defs = [
        "date TEXT", "ticker TEXT",
        "vr_c2c_abs REAL", "vr_o2c_abs REAL","vr_c2c REAL", "vr_o2c REAL", "vr_high_low REAL",
        "rank_vr_c2c REAL", "percentile_vr_c2c REAL",
        "rank_vr_o2c REAL", "percentile_vr_o2c REAL",
        "rank_vr_high_low REAL", "percentile_vr_high_low REAL"
    ]

    for p in hv_periods:
        column_defs += [f"hv{p} REAL", f"rank_hv{p} REAL", f"percentile_hv{p} REAL"]

    for p in ma_periods:
        for typ in ["c2c", "o2c"]:
            column_defs += [f"ma{p}_{typ} REAL", f"rank_ma{p}_{typ} REAL", f"percentile_ma{p}_{typ} REAL"]

    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS volatility_stats (
            {', '.join(column_defs)},
            PRIMARY KEY (date, ticker)
        )
    """)

    cursor.execute("DELETE FROM volatility_stats WHERE ticker = ?", (ticker,))
    conn.commit()

    df["ticker"] = ticker
    df["date"] = df["date"].astype(str)

    keep_cols = ["date", "ticker",
                 "vr_c2c_abs", "vr_o2c_abs","vr_c2c", "vr_o2c", "vr_high_low",
                 "rank_vr_c2c", "percentile_vr_c2c",
                 "rank_vr_o2c", "percentile_vr_o2c",
                 "rank_vr_high_low", "percentile_vr_high_low"] + \
                [col for col in df.columns if col.startswith(("hv", "ma", "rank_hv", "percentile_hv", "rank_ma", "percentile_ma"))]

    df_clean = df[keep_cols].dropna()

    placeholders = ", ".join(["?"] * len(df_clean.columns))
    insert_sql = f"INSERT OR REPLACE INTO volatility_stats ({', '.join(df_clean.columns)}) VALUES ({placeholders})"
    for _, row in df_clean.iterrows():
        cursor.execute(insert_sql, tuple(row))

    conn.commit()
    conn.close()
    print(f"✅ {ticker} : {len(df_clean)} lignes insérées.")

if __name__ == "__main__":
    db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "db", "market_data.db"))
    tickers = get_all_tickers(db_path)
    print(f"📊 Mise à jour des tickers : {tickers}")
    for ticker in tickers:
        update_volatility_stats(ticker, db_path)
