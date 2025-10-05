import os
import sqlite3
from datetime import datetime, timedelta

from ib_insync import util, Stock, Index, Future
from core.tws_connector import get_ib_connection
from core.tickers_catalog import TICKERS_CATALOG

DB_PATH = "db/market_data.db"
START_DATE = datetime(2015, 1, 1)
TODAY = datetime.now()

def daterange_chunks(start, end, delta_days=365):
    """Génère des paires (début, fin) pour des plages de dates ≤ 365 jours."""
    current = start
    while current < end:
        chunk_end = min(current + timedelta(days=delta_days), end)
        yield current, chunk_end
        current = chunk_end

def initialize_database(tickers_to_import):
    os.makedirs("db", exist_ok=True)
    ib = get_ib_connection()
    con = sqlite3.connect(DB_PATH)
    cursor = con.cursor()

    for symbol in tickers_to_import:
        contract = TICKERS_CATALOG.get(symbol)
        if not contract:
            print(f"❌ {symbol} : contrat introuvable dans le catalogue.")
            continue

        if isinstance(contract, dict):
            secType = contract.get("secType", "IND")
            if secType == "STK":
                contract = Stock(contract["symbol"], contract["exchange"], contract["currency"])
            elif secType == "IND":
                contract = Index(contract["symbol"], contract["exchange"], contract["currency"])
            elif secType == "FUT":
                contract = Future(contract["symbol"], contract["lastTradeDateOrContractMonth"],
                                  contract["exchange"], contract["currency"])
            else:
                print(f"❌ {symbol} : type de contrat non supporté : {secType}")
                continue

        table_name = f"{symbol.lower()}_data"
        total_inserted = 0

        # Vérifie la dernière date connue en base
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
        if cursor.fetchone():
            cursor.execute(f"SELECT MAX(date) FROM {table_name}")
            result = cursor.fetchone()
            latest_date_str = result[0] if result else None
            start_date = datetime.strptime(latest_date_str, "%Y-%m-%d") + timedelta(days=1) if latest_date_str else START_DATE
        else:
            start_date = START_DATE

        for start, end in daterange_chunks(start_date, TODAY):
            try:
                bars = ib.reqHistoricalData(
                    contract,
                    endDateTime=end,
                    durationStr="1 Y",
                    barSizeSetting="1 day",
                    whatToShow="TRADES",
                    useRTH=True,
                    formatDate=1
                )

                if not bars:
                    print(f"⚠️ {symbol} : aucune donnée entre {start.date()} et {end.date()}")
                    continue

                # ⚠️ Création de la table seulement si des données sont trouvées
                if total_inserted == 0:
                    cursor.execute(f"""
                        CREATE TABLE IF NOT EXISTS {table_name} (
                            date TEXT PRIMARY KEY,
                            open REAL,
                            high REAL,
                            low REAL,
                            close REAL
                        )
                    """)

                for bar in bars:
                    cursor.execute(
                        f"""
                        INSERT OR REPLACE INTO {table_name} (date, open, high, low, close)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            bar.date.strftime("%Y-%m-%d"),
                            bar.open,
                            bar.high,
                            bar.low,
                            bar.close
                        )
                    )
                total_inserted += len(bars)
                print(f"🗕️ {symbol} : {start.date()} → {end.date()} : {len(bars)} lignes insérées.")

            except Exception as e:
                print(f"❌ {symbol} : erreur {e} entre {start.date()} et {end.date()}")

        if total_inserted == 0:
            print(f"❌ {symbol} : aucune donnée historique récupérée — table non créée.")
        else:
            print(f"✅ {symbol} : {total_inserted} lignes ajoutées à la table '{table_name}'.")

    con.commit()
    con.close()
    ib.disconnect()
    print("✅ Base de données initialisée.")
