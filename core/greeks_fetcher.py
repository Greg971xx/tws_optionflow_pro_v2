import sqlite3
import math
import random
from datetime import datetime
from typing import Optional
from ib_insync import IB, Contract, Option
from core.contract_utils import resolve_underlying_contract

DB_PATH = "db/market_data.db"

def create_greeks_table():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS greeks_observations (
            date TEXT,
            ticker TEXT,
            type TEXT,
            strike REAL,
            delta REAL,
            gamma REAL,
            vega REAL,
            theta REAL,
            iv REAL,
            undPrice REAL,
            PRIMARY KEY (date, ticker, type, strike)
        )
    """)
    conn.commit()
    conn.close()

def update_iv_from_greeks(ticker: str, force_update=False):
    create_greeks_table()
    today = datetime.now().strftime("%Y-%m-%d")

    if not force_update:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM greeks_observations WHERE date = ? AND ticker = ?", (today, ticker))
        exists = cursor.fetchone() is not None
        conn.close()
        if exists:
            print(f"⏩ {ticker} : données déjà présentes pour aujourd'hui")
            return None, None

    ib = IB()
    client_id = random.randint(100, 999)
    ib.connect('127.0.0.1', 7497, clientId=client_id)

    index_contract = resolve_underlying_contract(ib, ticker)
    print(f"✅ Contrat futur actif sélectionné : {index_contract}")

    # 🔍 Récupérer les expirations disponibles
    fut_fop_exchange = "CME" if index_contract.secType == "FUT" else ""
    opt_params_list = ib.reqSecDefOptParams(
        underlyingSymbol=index_contract.symbol,
        futFopExchange=fut_fop_exchange,
        underlyingSecType=index_contract.secType,
        underlyingConId=index_contract.conId
    )

    expirations = []
    for opt_param in opt_params_list:
        expirations = sorted(datetime.strptime(e, "%Y%m%d") for e in opt_param.expirations)
        break

    print(f"📅 Échéances disponibles pour {ticker} : {[e.strftime('%Y-%m-%d') for e in expirations]}")
    if not expirations:
        ib.disconnect()
        raise Exception(f"Aucune expiration trouvée pour {ticker}")

    expiry = datetime.strptime("20250918", "%Y%m%d")

    # 🔍 Récupérer le prix spot
    ticker_data = ib.reqMktData(index_contract, snapshot=False, regulatorySnapshot=False)
    ib.sleep(4)
    spot_price = next((val for val in [ticker_data.last, ticker_data.close, ticker_data.bid, ticker_data.ask] if val and not math.isnan(val)), None)

    if spot_price is None:
        ib.disconnect()
        raise Exception(f"Impossible d'obtenir un prix spot pour {ticker}")

    # ➕ Construction des strikes
    central_strike = round(spot_price / 5) * 5
    strikes = [central_strike + i * 5 for i in range(-2, 3)]

    rows = []

    for option_type in ["CALL", "PUT"]:
        for strike in strikes:
            option_type_letter = option_type[0]

            # 1. Recherche du contrat qualifié
            lookup_contract = Contract(
                secType="FOP" if index_contract.secType == "FUT" else "OPT",
                symbol=index_contract.symbol,
                lastTradeDateOrContractMonth=expiry.strftime("%Y%m%d"),
                strike=strike,
                right=option_type_letter,
                exchange="CME" if index_contract.secType == "FUT" else index_contract.exchange,
                currency=index_contract.currency
            )

            contracts = ib.reqContractDetails(lookup_contract)
            if not contracts:
                print(f"❌ Aucun détail pour {lookup_contract}")
                continue

            qualified = contracts[0].contract

            # 2. Contrat complet à utiliser
            option_contract = Contract(
                secType=qualified.secType,
                symbol=qualified.symbol,
                lastTradeDateOrContractMonth=qualified.lastTradeDateOrContractMonth,
                strike=qualified.strike,
                right=qualified.right,
                exchange=qualified.exchange,
                currency=qualified.currency,
                multiplier=qualified.multiplier,
                tradingClass=qualified.tradingClass
            )

            ib.qualifyContracts(option_contract)
            print(f"✅ Contrat qualifié : {option_contract}")

            data = ib.reqMktData(option_contract, snapshot=False, regulatorySnapshot=False)
            ib.sleep(3)

            if data.modelGreeks and data.modelGreeks.impliedVol is not None:
                rows.append((
                    today, ticker, option_type, strike,
                    data.modelGreeks.delta,
                    data.modelGreeks.gamma,
                    data.modelGreeks.vega,
                    data.modelGreeks.theta,
                    data.modelGreeks.impliedVol,
                    spot_price
                ))

    ib.disconnect()

    if not rows:
        raise Exception("Aucune donnée de Greeks disponible")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.executemany("""
        INSERT OR REPLACE INTO greeks_observations
        (date, ticker, type, strike, delta, gamma, vega, theta, iv, undPrice)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, rows)
    conn.commit()
    conn.close()

    iv_atm = sorted(rows, key=lambda x: abs(x[4] - 0.5))
    return iv_atm[0][8], iv_atm[0][9]  # IV, spot
