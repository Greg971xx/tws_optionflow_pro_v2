# recherche.py (patch)
from ib_insync import *
import time
import math

HOST, PORT, CID = "127.0.0.1", 7497, 21

# Essaie SPXW puis SPX, et CBOE puis SMART
SYMBOLS = [("SPX", "SPXW"), ("SPX", "SPX")]
EXCHANGES = ["CBOE", "SMART"]

# Choisis une échéance proche mais > aujourd'hui
EXP = "20250930"       # YYYYMMDD
RIGHT = "C"            # "C" ou "P"
STRIKE = 6660

GEN_TICKS = "101"      # OI uniquement
TIMEOUT_S = 8.0        # attente max par essai

def oi_field(r):
    return "callOpenInterest" if r.upper().startswith("C") else "putOpenInterest"

ib = IB()
ib.errorEvent += print
ib.connect(HOST, PORT, clientId=CID)

import math

def safe_oi_int(val):
    """Retourne un int si possible, sinon None (gère NaN/inf/etc.)."""
    try:
        if val is None:
            return None
        if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
            return None
        return int(val)
    except Exception:
        return None

def try_one(contract) -> int | None:
    """
    Requête OI en streaming (snapshot=False) et ticklist "101".
    On essaie marketDataType = 1,2,3,4.
    Retourne un int (>=0) si dispo, sinon None.
    """
    fld = oi_field(contract.right)

    for mkt in (1, 2, 3, 4):
        try:
            ib.reqMarketDataType(mkt)
        except Exception:
            pass

        # IMPORTANT: streaming pour generic ticks (snapshot=False)
        t = ib.reqMktData(
            contract,
            genericTickList=GEN_TICKS,   # "101"
            snapshot=False,
            regulatorySnapshot=False
        )
        deadline = time.time() + TIMEOUT_S
        got = None
        try:
            while time.time() < deadline:
                ib.sleep(0.25)
                raw = getattr(t, fld, None)
                oi_val = safe_oi_int(raw)
                if oi_val is not None:      # <- on ignore NaN/None
                    got = oi_val
                    break
        finally:
            try:
                ib.cancelMktData(t)
            except Exception:
                pass

        if got is not None:
            return got
    return None


oi_val = None
tried = []

for sym, tclass in SYMBOLS:
    for exch in EXCHANGES:
        # Qualifier d'abord le contrat (évite "Ambiguous contract")
        opt = Option(symbol=sym,
                     lastTradeDateOrContractMonth=EXP,
                     strike=float(STRIKE),
                     right=RIGHT,
                     exchange=exch,
                     currency="USD",
                     multiplier="100",
                     tradingClass=tclass)
        try:
            qual = ib.qualifyContracts(opt)
            if not qual:
                tried.append((sym, tclass, exch, "qualify=0"))
                continue
            c = qual[0]
        except Exception as e:
            tried.append((sym, tclass, exch, f"qualify_err={e}"))
            continue

        r = try_one(c)
        tried.append((sym, tclass, exch, f"oi={r}"))
        if r is not None:
            oi_val = r
            print(f"\n✅ OI trouvé: {r} | {sym} {tclass} {exch} {EXP} {RIGHT} {STRIKE}")
            break
    if oi_val is not None:
        break

ib.disconnect()

print("\nTentatives détaillées:")
for row in tried:
    print("  ", row)

if oi_val is None:
    print("\n❌ Aucun OI reçu. Causes probables:")
    print("   1) Abonnement OI manquant sur options US (tick générique 101 non autorisé).")
    print("   2) Échange / tradingClass non éligible pour ton entitlement.")
    print("   3) Marché fermé + flux retardé ne renvoyant pas 101.")
