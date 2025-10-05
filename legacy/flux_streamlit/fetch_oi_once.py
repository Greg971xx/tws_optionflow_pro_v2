# fetch_oi_once.py
from oi_utils import ensure_oi_schema, fetch_oi_snapshot_for_expiry
import sqlite3
import pandas as pd

DB = r"C:\Users\decle\PycharmProjects\flux_claude\db\optionflow.db"
EXP = "20250930"          # YYYYMMDD (échéance à relever)

def main():
    ensure_oi_schema(DB)
    ok = fetch_oi_snapshot_for_expiry(
        db_path=DB,
        expiry=EXP,
        host="127.0.0.1",
        port=7497,
        client_id=21,
        exchange="CBOE",          # on commence par CBOE
        batch_size=50,
        pause=0.35,
        timeout_s=8.0,
        retries=1,
        try_smart_on_empty=True,  # fallback SMART si CBOE renvoie vide
        debug=True,
    )
    print("Inserted? ->", ok)

    # Vérification rapide
    con = sqlite3.connect(DB)
    n = pd.read_sql("SELECT COUNT(*) AS n FROM oi_snapshots WHERE expiry=?;", con, params=[EXP]).iloc[0,0]
    print(f"oi_snapshots rows for {EXP}: {n}")
    if n > 0:
        preview = pd.read_sql("""
            SELECT ts, symbol, trading_class, expiry, right, strike, open_interest
            FROM oi_snapshots
            WHERE expiry=?
            ORDER BY ts DESC, strike
            LIMIT 10;
        """, con, params=[EXP])
        print(preview.to_string(index=False))
    con.close()

if __name__ == "__main__":
    main()
