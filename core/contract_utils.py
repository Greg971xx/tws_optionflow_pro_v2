from ib_insync import Stock, Index, Future, Contract, IB
from typing import Optional
import time
from datetime import datetime

# Mapping d'exchange et devise par ticker (peut être enrichi)
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
    "NQ": ("GLOBEX", "USD"),
    "RTY": ("GLOBEX", "USD"),
    "YM": ("CBOT", "USD"),
    "GC": ("COMEX", "USD"),
    "CL": ("NYMEX", "USD"),
    "SI": ("COMEX", "USD"),
    "6E": ("CME", "USD"),
    "M6E": ("CME", "USD"),
    "6J": ("CME", "USD"),
    "M6J": ("CME", "USD"),
    "OVX": ("CBOE", "USD"),
    "VXD": ("CBOE", "USD")
}

# Mapping du type probable par ticker
TICKER_TYPES = {
    "IND": ["SPX", "NDX", "RUT", "VIX", "DAX", "CAC40", "ESTX50", "VXN", "RVX", "DJX", "V2TX", "VXD", "OVX"],
    "FUT": ["ES", "NQ", "RTY", "GC", "CL", "SI", "6E", "6J", "M6E", "M6J", "YM"],
    "STK": []  # fallback pour tout le reste (AAPL, TSLA, SPY, etc.)
}


def resolve_underlying_contract(ib: IB, symbol: str, expiry: Optional[str] = None):
    """
    Résout un contrat sous-jacent pour STOCK, INDEX ou FUTURE.
    Pour les Futures (ES, NQ, etc.), renvoie le contrat actif si aucune expiry n’est donnée.
    """
    exchange, currency = EXCHANGE_MAPPING.get(symbol, ("SMART", "USD"))

    # 🔍 Futures (ES, NQ, GC, etc.)
    if symbol in TICKER_TYPES["FUT"]:
        if expiry:
            future_contract = Future(symbol=symbol, exchange=exchange, currency=currency,
                                     lastTradeDateOrContractMonth=expiry)
        else:
            future_contract = Future(symbol=symbol, exchange=exchange, currency=currency)

        contracts = ib.reqContractDetails(future_contract)
        if not contracts:
            raise Exception(f"❌ Aucun contrat futur trouvé pour {symbol} sur {exchange}")

        print(f"✅ Contrat futur actif sélectionné : {contracts[0].contract}")
        return contracts[0].contract

    # 🔍 Indices, Stocks
    if symbol in TICKER_TYPES["IND"]:
        contract = Index(symbol=symbol, exchange=exchange, currency=currency)
    elif symbol in TICKER_TYPES["STK"]:
        contract = Stock(symbol=symbol, exchange=exchange, currency=currency)
    else:
        # fallback : essayer STK puis IND
        for cls in [Stock, Index]:
            contract = cls(symbol=symbol, exchange=exchange, currency=currency)
            try:
                details = ib.reqContractDetails(contract)
                if details:
                    return details[0].contract
            except Exception:
                continue

        raise Exception(f"❌ Impossible de qualifier {symbol} (ni STK, ni IND)")

    # 🔍 Qualification finale
    details = ib.reqContractDetails(contract)
    if not details:
        raise Exception(f"❌ Aucun détail trouvé pour {symbol} ({contract.secType})")

    return details[0].contract
