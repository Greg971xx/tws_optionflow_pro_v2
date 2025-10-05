import sqlite3
import pandas as pd
import numpy as np
import os

def compute_rank_percentile(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    rank = (series - series.min()) / (series.max() - series.min())
    percentile = series.expanding().apply(lambda x: (x < x[-1]).mean(), raw=True)
    return rank, percentile

def drop_volatility_stats_table(db_path: str = None):
    if db_path is None:
        db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "db", "market_data.db"))
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS volatility_stats")
    conn.commit()
    conn.close()
    print("✅ Table 'volatility_stats' supprimée proprement.")

def update_volatility_stats(ticker: str, db_path: str = None):
    if db_path is None:
        db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "db", "market_data.db"))

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        df = pd.read_sql(f"SELECT * FROM {ticker.lower()}_data", conn, parse_dates=["date"])
    except Exception as e:
        print(f"❌ Erreur : impossible de charger les données pour {ticker} : {e}")
        conn.close()
        return

    df = df.sort_values("date")

    # 🔢 Volatilité journalière brute (non annualisée)
    df["vr_c2c"] = df["close"].pct_change().abs()
    df["vr_o2c"] = ((df["close"] - df["open"]) / df["open"]).abs()

    # ➕ Volatilité journalière signée en pourcentage
    df["vr_c2c_pct"] = df["close"].pct_change() * 100
    df["vr_o2c_pct"] = ((df["close"] - df["open"]) / df["open"]) * 100

    # Périodes pour la volatilité historique (HV) et les moyennes mobiles (MA)
    hv_periods = [5, 20, 60, 120, 252]
    ma_periods = [20, 60, 120, 252]

    for p in hv_periods:
        df[f"hv{p}"] = df["vr_c2c"].rolling(p).std() * np.sqrt(252)
        rank_col, percentile_col = compute_rank_percentile(df[f"hv{p}"])
        df[f"rank_hv{p}"] = rank_col
        df[f"percentile_hv{p}"] = percentile_col

    for p in ma_periods:
        df[f"ma{p}_c2c"] = df["vr_c2c"].rolling(p).mean()
        df[f"ma{p}_o2c"] = df["vr_o2c"].rolling(p).mean()

        df[f"rank_ma{p}_c2c"], df[f"percentile_ma{p}_c2c"] = compute_rank_percentile(df[f"ma{p}_c2c"])
        df[f"rank_ma{p}_o2c"], df[f"percentile_ma{p}_o2c"] = compute_rank_percentile(df[f"ma{p}_o2c"])

    # Création de la table si elle n'existe pas
    column_defs = [
        "date TEXT", "ticker TEXT",
        "vr_c2c_pct REAL", "vr_o2c_pct REAL"  # ➕ nouvelles colonnes
    ]

    for p in hv_periods:
        column_defs += [
            f"hv{p} REAL",
            f"rank_hv{p} REAL",
            f"percentile_hv{p} REAL"
        ]

    for p in ma_periods:
        for typ in ["c2c", "o2c"]:
            column_defs += [
                f"ma{p}_{typ} REAL",
                f"rank_ma{p}_{typ} REAL",
                f"percentile_ma{p}_{typ} REAL"
            ]

    create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS volatility_stats (
            {', '.join(column_defs)},
            PRIMARY KEY (date, ticker)
        )
    """
    cursor.execute(create_table_sql)

    # Supprimer les anciennes lignes du ticker
    cursor.execute("DELETE FROM volatility_stats WHERE ticker = ?", (ticker,))
    conn.commit()

    df["ticker"] = ticker
    df["date"] = df["date"].astype(str)

    all_columns = ["date", "ticker", "vr_c2c_pct", "vr_o2c_pct"] + \
                  [col for col in df.columns if col.startswith(("hv", "ma", "rank", "percentile"))]

    df_insert = df[all_columns].dropna()

    placeholders = ", ".join(["?"] * len(df_insert.columns))
    insert_sql = f"""
        INSERT OR REPLACE INTO volatility_stats ({', '.join(df_insert.columns)})
        VALUES ({placeholders})
    """
    for _, row in df_insert.iterrows():
        cursor.execute(insert_sql, tuple(row))

    conn.commit()
    conn.close()
    print(f"✅ {len(df_insert)} lignes insérées dans volatility_stats pour {ticker}")
