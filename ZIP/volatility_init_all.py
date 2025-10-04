import os
import sqlite3
import pandas as pd
from datetime import datetime
from core.volatility_stats_builder import update_volatility_stats


def is_volatility_data_up_to_date(ticker: str, db_path: str) -> bool:
    """Vérifie si les données de volatilité sont à jour pour le ticker."""
    conn = sqlite3.connect(db_path)
    try:
        df_price = pd.read_sql(
            f"SELECT date FROM {ticker.lower()}_data ORDER BY date DESC LIMIT 1", conn
        )
        if df_price.empty:
            print(f"⚠️ Aucun prix disponible pour {ticker}.")
            return False
        latest_price_date = pd.to_datetime(df_price["date"].iloc[0]).date()

        df_stats = pd.read_sql(
            "SELECT date FROM volatility_stats WHERE ticker = ? ORDER BY date DESC LIMIT 1",
            conn,
            params=(ticker.upper(),),  # car on stocke les tickers en MAJ dans volatility_stats
        )
        if df_stats.empty:
            print(f"ℹ️ Aucune stat de volatilité encore présente pour {ticker}.")
            return False
        latest_stats_date = pd.to_datetime(df_stats["date"].iloc[0]).date()

        return latest_stats_date >= latest_price_date
    except Exception as e:
        print(f"⚠️ Erreur pendant la vérification pour {ticker}: {e}")
        return False
    finally:
        conn.close()


def update_all_volatility_stats(log_func=print):
    """Met à jour les données de volatilité pour tous les tickers si nécessaire."""
    db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "db", "market_data.db"))
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [name[0] for name in cursor.fetchall()]
    conn.close()

    # On détecte tous les tickers dont la table finit par '_data'
    tickers = [t.replace("_data", "").upper() for t in tables if t.endswith("_data")]

    print(f"\n🔍 Tables trouvées : {tables}")
    print(f"📈 Tickers détectés : {tickers}\n")

    log_func(f"🔍 Tables trouvées : {tables}")

    for ticker in tickers:
        print(f"➡️ Vérification des données de {ticker}...")
        log_func(f"➡️ Vérification des données de {ticker}...")
        if not is_volatility_data_up_to_date(ticker, db_path):
            print(f"🛠️ Mise à jour des stats de volatilité pour {ticker}")
            log_func(f"✅ Données de volatilité déjà à jour pour {ticker}")
            update_volatility_stats(ticker, db_path)
        else:
            print(f"✅ Données de volatilité déjà à jour pour {ticker}")

    print("\n✅ Tous les tickers ont été traités.")


if __name__ == "__main__":
    update_all_volatility_stats()
