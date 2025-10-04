# core/ib_gex_utils.py

from ib_insync import IB, Option
from typing import List, Tuple
import pandas as pd
from core.contract_utils import resolve_underlying_contract
from datetime import datetime

def fetch_expirations(symbol: str) -> List[str]:
    try:
        ib = IB()
        import random
        client_id = random.randint(100, 999)
        ib.connect("127.0.0.1", 7497, clientId=client_id)

        contract = resolve_underlying_contract(ib, symbol)
        qualified = ib.qualifyContracts(contract)
        if not qualified:
            raise Exception(f"❌ Contract non qualifié : {contract}")
        chains = ib.reqSecDefOptParams(contract.symbol, "", contract.secType, contract.conId)
        expirations = sorted(set(chains[0].expirations))
        return expirations
    finally:
        ib.disconnect()

def get_spot_and_strikes(symbol: str, ib: IB, strike_count: int = 20) -> Tuple[float, List[float]]:
    contract = resolve_underlying_contract(ib, symbol)
    qualified = ib.qualifyContracts(contract)
    if not qualified:
        raise Exception(f"❌ Contract non qualifié : {contract}")

    ticker = ib.reqMktData(contract, snapshot=True, regulatorySnapshot=True)
    ib.sleep(1.5)
    spot = ticker.last if ticker.last else ticker.close
    if spot is None:
        raise ValueError("❌ Impossible d'obtenir le prix spot.")

    base_strike = round(spot / 5) * 5
    strikes = [base_strike + i * 5 for i in range(-strike_count, strike_count + 1)]
    ib.cancelMktData(contract)
    return spot, strikes

def fetch_gex_data(symbol: str, expiry: str) -> pd.DataFrame:
    ib = IB()
    ib.connect("127.0.0.1", 7497, clientId=12)

    contract = resolve_underlying_contract(ib, symbol)
    qualified = ib.qualifyContracts(contract)
    if not qualified:
        ib.disconnect()
        raise Exception(f"❌ Contract non qualifié : {contract}")

    spot, strikes = get_spot_and_strikes(symbol, ib)

    rows = []
    for strike in strikes:
        for opt_type in ['C', 'P']:
            option = Option(symbol, expiry, strike, opt_type, exchange="SMART")
            ib.qualifyContracts(option)
            ticker = ib.reqMktData(option, snapshot=True, regulatorySnapshot=True)
            ib.sleep(0.5)

            greek = ticker.modelGreeks
            if greek and greek.gamma is not None:
                row = {
                    "strike": strike,
                    "type": opt_type,
                    "gamma": greek.gamma,
                    "openInterest": ticker.openInterest or 0,
                    "contractMultiplier": 100,
                    "undPrice": spot
                }
                rows.append(row)

            ib.cancelMktData(option)

    ib.disconnect()
    return pd.DataFrame(rows)
