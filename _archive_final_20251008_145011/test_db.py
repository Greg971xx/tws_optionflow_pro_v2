import sqlite3
import pandas as pd

DB_PATH = "db/market_data.db"  # adapte si nécessaire
TICKER_TEST = "SPX"  # peu importe, juste pour voir

with sqlite3.connect(DB_PATH) as conn:
    # 1️⃣ Colonnes présentes
    print("📌 Colonnes dans volatility_stats :")
    cols_df = pd.read_sql("PRAGMA table_info(volatility_stats)", conn)
    print(cols_df[["name", "type"]])

    # 2️⃣ Nombre de lignes par ticker
    print("\n📊 Nombre de lignes par ticker :")
    counts_df = pd.read_sql(
        "SELECT ticker, COUNT(*) as nb_lignes FROM volatility_stats GROUP BY ticker ORDER BY nb_lignes DESC",
        conn
    )
    print(counts_df)

    # 3️⃣ Dernières lignes pour le ticker test
    print(f"\n📌 Dernières lignes pour {TICKER_TEST}:")
    try:
        df = pd.read_sql(
            "SELECT * FROM volatility_stats WHERE ticker = ? ORDER BY date DESC LIMIT 5",
            conn,
            params=(TICKER_TEST,)
        )
        print(df)
    except Exception as e:
        print(f"❌ Erreur lecture : {e}")
