from ib_insync import IB, Contract, Future, Index, Stock
import json
import time

# Liste des tickers à tester avec leurs types potentiels
TICKERS_TO_TEST = {
    "SPX": ["IND"],
    "SPY": ["STK"],
    "NDX": ["IND"],
    "QQQ": ["STK"],
    "RUT": ["IND"],
    "IWM": ["STK"],
    "DIA": ["STK"],
    "DJX": ["IND"],
    "VIX": ["IND"],
    "VXN": ["IND"],
    "RVX": ["IND"],
    "OVX": ["IND"],
    "VXD": ["IND"],

    "AAPL": ["STK"], "AMZN": ["STK"], "META": ["STK"], "GOOGL": ["STK"],
    "MSFT": ["STK"], "NVDA": ["STK"], "TSLA": ["STK"],
    "ES": ["FUT"], "NQ": ["FUT"], "RTY": ["FUT"], "YM": ["IND"],
    "GC": ["FUT"], "CL": ["FUT"], "SI": ["FUT"],
    "GLD": ["STK"], "USO": ["STK"], "SLV": ["STK"],
    "UUP": ["STK"], "FXE": ["STK"], "FXY": ["STK"],
    "6E": ["FUT"], "6J": ["FUT"],
    "M6E": ["FUT"], "M6J": ["FUT"],
    "ESTX50": ["IND"], "DAX": ["IND"], "CAC40": ["IND"], "V2TX": ["IND"]
}

# Remplace EXCHANGES_BY_TYPE par :
EXCHANGE_MAPPING = {
    "SPX": ("CBOE", "USD"),
    "NDX": ("NASDAQ", "USD"),
    "RUT": ("RUSSELL", "USD"),
    "DJX": ("CBOE", "USD"),
    "VIX": ("CBOE", "USD"),
    "VXN": ("CBOE", "USD"),
    "RVX": ("CBOE", "USD"),
    "ESTX50": ("EUREX", "EUR"),
    "DAX": ("EUREX", "EUR"),
    "CAC40": ("MONEP", "EUR"),
    "V2TX": ("EUREX", "EUR"),
    "ES": ("CME", "USD"),
    "NQ": ("CME", "USD"),
    "RTY": ("CME", "USD"),
    "YM": ("CBOT", "USD"),
    "GC": ("COMEX", "USD"),
    "CL": ("NYMEX", "USD"),
    "SI": ("COMEX", "USD"),
    "6E": ("CME", "USD"),
    "M6E":("CME", "USD"),
    "6J": ("CME", "USD"),
    "M6J": ("CME", "USD"),
    "OVX": ("CBOE", "USD"),
    "VXD": ("CBOE", "USD")
    # tous les autres -> fallback sur SMART/USD
}



OUTPUT_FILE = "tickers_verified.json"

def create_contract(symbol, secType):
    exchange, currency = EXCHANGE_MAPPING.get(symbol, ("SMART", "USD"))

    if secType == "FUT":
        return Future(symbol=symbol, exchange=exchange, currency=currency)
    elif secType == "IND":
        return Index(symbol=symbol, exchange=exchange, currency=currency)
    elif secType == "STK":
        return Stock(symbol=symbol, exchange=exchange, currency=currency)
    else:
        return None


def main():
    ib = IB()
    ib.connect("127.0.0.1", 7497, clientId=12)

    verified = {}

    for symbol, secTypes in TICKERS_TO_TEST.items():
        for secType in secTypes:
            contract = create_contract(symbol, secType)
            if contract is None:
                continue

            try:
                details = ib.reqContractDetails(contract)
                if details:
                    cd = details[0].contract
                    verified[symbol] = {
                        "secType": cd.secType,
                        "exchange": cd.exchange,
                        "currency": cd.currency,
                        "conId": cd.conId
                    }
                    if cd.secType == "FUT":
                        verified[symbol]["lastTradeDateOrContractMonth"] = cd.lastTradeDateOrContractMonth
                    print(f"✅ {symbol} ({secType}) OK")
                    break  # on garde le premier format valide
                else:
                    print(f"❌ {symbol} ({secType}) not found")
            except Exception as e:
                print(f"❌ {symbol} ({secType}) error: {e}")
            time.sleep(0.3)

    # Save results
    with open(OUTPUT_FILE, "w") as f:
        json.dump(verified, f, indent=2)

    print(f"\n✔️ Sauvegardé dans {OUTPUT_FILE} ({len(verified)} tickers validés)")
    ib.disconnect()

if __name__ == "__main__":
    main()
