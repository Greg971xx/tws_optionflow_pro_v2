import time
import csv
from datetime import datetime
from ib_insync import *

# === CONFIGURATION ===
symbol = 'AAPL'
expiry = '20250808'      # ⚠️ Adapter si besoin
right = 'C'              # 'C' = Call, 'P' = Put
output_file = 'greeks_check_adaptive.csv'
interval = 15 * 60       # 15 minutes

# === CONNEXION TWS ===
ib = IB()
try:
    ib.connect('127.0.0.1', 7497, clientId=12)
except Exception as e:
    print(f"❌ Erreur connexion TWS : {e}")
    exit()

# === SPOT OU CLOSE POUR SPX ===
try:
    underlying = Stock(symbol, 'SMART', 'USD')
    ib.qualifyContracts(underlying)
    spot_data = ib.reqMktData(underlying, '', False, False)

    print("⏳ Tentative récupération spot temps réel...", end='', flush=True)
    for _ in range(10):
        ib.sleep(1)
        if spot_data.last is not None and spot_data.last > 0:
            spot_price = spot_data.last
            print(" ✅")
            break
    else:
        raise Exception("⚠️ Pas de spot, fallback historique...")

except Exception:
    print("⏳ Récupération du dernier close (historique)...", end='', flush=True)
    bars = ib.reqHistoricalData(
        Index(symbol, 'SMART', 'USD'),
        endDateTime='',
        durationStr='1 D',
        barSizeSetting='1 day',
        whatToShow='TRADES',
        useRTH=True,
        formatDate=1
    )
    if not bars:
        print(" ❌ Impossible d'obtenir un prix")
        exit()
    spot_price = bars[-1].close
    print(" ✅")

# === STRIKE ATM ===
strike_step = 5 if symbol in ['AAPL', 'NDX'] else 1
strike = round(spot_price / strike_step) * strike_step
print(f"📈 Spot estimé : {spot_price} → 🎯 Strike ATM = {strike}")


# === CONTRAT OPTION ===
contract = Option(symbol=symbol,
                  lastTradeDateOrContractMonth=expiry,
                  strike=strike,
                  right=right,
                  exchange='SMART',
                  currency='USD')

print("🛠️ Contrat brut créé :")
print(f"  Symbole       : {symbol}")
print(f"  Expiry        : {expiry}")
print(f"  Strike        : {strike}")
print(f"  Type          : {right}")
print(f"  Exchange      : SMART")

ib.qualifyContracts(contract)

print("✅ Contrat qualifié avec IB. Détails :")
for key, value in contract.__dict__.items():
    print(f"  {key}: {value}")

# === FICHIER CSV ===
with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Timestamp', 'Bid', 'Ask', 'IV', 'Delta', 'Gamma', 'Vega', 'Theta'])

    print("🚀 Début du monitoring toutes les 15 minutes...")

    while True:
        ticker = ib.reqMktData(contract, '', False, False)

        print("⏳ Attente des Greeks...", end='', flush=True)
        start = time.time()

        # Attente adaptative max 30 secondes
        for i in range(30):
            ib.sleep(1)
            greeks = ticker.modelGreeks
            if greeks and greeks.impliedVol is not None:
                print(" ✅")
                break
            print('.', end='', flush=True)
        else:
            print(" ❌ Timeout")

        end = time.time()

        # Récupération complète des données
        greeks = ticker.modelGreeks
        if greeks:
            iv     = greeks.impliedVol
            delta  = greeks.delta
            gamma  = greeks.gamma
            vega   = greeks.vega
            theta  = greeks.theta
        else:
            iv = delta = gamma = vega = theta = None

        bid = ticker.bid
        ask = ticker.ask
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        if iv is not None:
            print(f"✅ {timestamp} — IV: {iv:.4f}, Δ: {delta:.4f}, Θ: {theta:.4f} | Bid: {bid}, Ask: {ask}")
        else:
            print(f"❌ {timestamp} — Pas de Greeks (Bid: {bid}, Ask: {ask}) après {end - start:.1f} sec")

        writer.writerow([timestamp, bid, ask, iv, delta, gamma, vega, theta])
        file.flush()

        time.sleep(interval)
