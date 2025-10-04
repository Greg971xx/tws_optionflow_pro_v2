from ib_insync import IB, Future, Contract
import math
from datetime import datetime

SYMBOL = "ES"
EXCHANGE = "CME"
CURRENCY = "USD"
RIGHT = "C"

ib = IB()
ib.connect("127.0.0.1", 7497, clientId=22)

# === 1. Contrat futur actif ===
fut = Future(symbol=SYMBOL, exchange=EXCHANGE, currency=CURRENCY)
contracts = ib.reqContractDetails(fut)
qualified_fut = sorted(contracts, key=lambda x: x.contract.lastTradeDateOrContractMonth)[0].contract

ticker_fut = ib.reqMktData(qualified_fut, "", snapshot=True)
ib.sleep(2)

# === 2. Spot price fiable ===
# Liste des champs potentiels
candidates = [ticker_fut.last, ticker_fut.close, ticker_fut.bid, ticker_fut.ask]
spot_price = next((x for x in candidates if x is not None and not math.isnan(x)), None)

if spot_price is None:
    raise Exception("❌ Spot price introuvable pour le contrat futur.")

print(f"📉 Ticker futur brut : {ticker_fut}")

if not spot_price or math.isnan(spot_price):
    raise Exception("❌ Spot price introuvable pour le contrat futur.")

print(f"\n✅ Futur actif : {qualified_fut.localSymbol} | Spot = {spot_price:.2f}")
print(f"toto {qualified_fut.localSymbol}")

# === 3. Paramètres des options ===
params = ib.reqSecDefOptParams(
    underlyingSymbol=qualified_fut.symbol,
    futFopExchange=qualified_fut.exchange,
    underlyingSecType="FUT",
    underlyingConId=qualified_fut.conId
)

if not params:
    raise Exception("❌ Aucun paramètre d'option trouvé.")

opt_param = params[0]

# === 4. Récupérer les échéances futures disponibles ===
def get_available_expirations(ticker: str):
    """ Récupérer les échéances futures disponibles pour un ticker et les trier """
    expirations = []
    for param in params:
        for expiry in param.expirations:
            expirations.append(expiry)

    # Trier les échéances par date
    expirations_sorted = sorted(expirations, key=lambda x: datetime.strptime(x, '%Y%m%d'))
    return expirations_sorted

expirations = get_available_expirations(SYMBOL)
print(f"\n📅 Échéances futures disponibles pour {SYMBOL} : {expirations}")

expiry = "20250808"  # Peut être ajusté en fonction des échéances récupérées

# === 5. Récupération des strikes ===
strikes = sorted(opt_param.strikes)
if not strikes:
    raise Exception("❌ Aucun strike disponible pour l'option.")

# === 6. Strike ATM autour du spot ===
strike = min(strikes, key=lambda x: abs(x - spot_price))

print(f"\n🧩 Construction d’un contrat option :")
print(f" - Expiry       : {expiry}")
print(f" - Strike ATM   : {strike}")

# === 7. Construction contrat brut ===
option_base = Contract(
    symbol=SYMBOL,
    secType="FOP",
    lastTradeDateOrContractMonth=expiry,
    strike=strike,
    right=RIGHT,
    exchange=EXCHANGE,
    currency=CURRENCY
)

details = ib.reqContractDetails(option_base)
fop_details = [d for d in details if d.contract.secType == "FOP"]

if not fop_details:
    raise Exception("❌ Aucun contrat FOP qualifié.")

c = fop_details[0].contract
print(f"\n🎯 Option qualifiée : {c.localSymbol} | conId={c.conId} | tradingClass={c.tradingClass}")

# === 8. Récupération des données marché + Greeks ===
ticker = ib.reqMktData(c, "", snapshot=False, regulatorySnapshot=False)
ib.sleep(2)

print(f"\n📈 Prix Option :")
print(f"Last = {getattr(ticker, 'last', 'n/a')}")
print(f"Bid  = {getattr(ticker, 'bid', 'n/a')}")
print(f"Ask  = {getattr(ticker, 'ask', 'n/a')}")
print(f"Close = {getattr(ticker, 'close', 'n/a')}")

if ticker.modelGreeks:
    g = ticker.modelGreeks
    print("\n📊 Greeks (modelGreeks) :")
    print(f"Delta = {g.delta}")
    print(f"Gamma = {g.gamma}")
    print(f"Vega  = {g.vega}")
    print(f"Theta = {g.theta}")
    print(f"IV    = {g.impliedVol}")
else:
    print("❌ Aucune donnée modelGreeks disponible.")
    print(ticker)

ib.disconnect()
