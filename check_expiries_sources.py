"""
Check where expiries come from and compare sources
"""
import sqlite3
import pandas as pd
from src.config import DATABASE_PATH

conn = sqlite3.connect(DATABASE_PATH)

print("=" * 70)
print("EXPIRIES COMPARISON - trades vs oi_snapshots")
print("=" * 70)

# 1. Expiries from trades table
print("\n📊 Expiries in TRADES table:")
trades_exp = pd.read_sql("""
    SELECT 
        expiry,
        COUNT(*) as trade_count,
        MAX(ts) as last_trade
    FROM trades
    WHERE expiry IS NOT NULL
    GROUP BY expiry
    ORDER BY expiry DESC
    LIMIT 10
""", conn)

if trades_exp.empty:
    print("   ⚠️  No expiries in trades table")
else:
    for _, row in trades_exp.iterrows():
        print(f"   • {row['expiry']}: {row['trade_count']:,} trades (last: {row['last_trade']})")

# 2. Expiries from oi_snapshots table
print("\n📊 Expiries in OI_SNAPSHOTS table:")
oi_exp = pd.read_sql("""
    SELECT 
        expiry,
        COUNT(*) as snapshot_count,
        MAX(ts) as last_snapshot
    FROM oi_snapshots
    WHERE expiry IS NOT NULL
    GROUP BY expiry
    ORDER BY expiry DESC
    LIMIT 10
""", conn)

if oi_exp.empty:
    print("   ⚠️  No expiries in oi_snapshots table")
else:
    for _, row in oi_exp.iterrows():
        print(f"   • {row['expiry']}: {row['snapshot_count']:,} snapshots (last: {row['last_snapshot']})")

# 3. Check if collector is running
print("\n🔍 Recent collector activity:")
recent_trades = pd.read_sql("""
    SELECT 
        expiry,
        COUNT(*) as count,
        MAX(ts) as last_ts
    FROM trades
    WHERE datetime(ts) > datetime('now', '-1 hour')
    GROUP BY expiry
    ORDER BY last_ts DESC
""", conn)

if recent_trades.empty:
    print("   ⚠️  No trades in last hour (collector not running?)")
else:
    print(f"   ✓ Collector active! Recent trades:")
    for _, row in recent_trades.iterrows():
        print(f"     • {row['expiry']}: {row['count']} trades (last: {row['last_ts']})")

conn.close()

print("\n" + "=" * 70)
print("💡 ANALYSIS")
print("=" * 70)
print("\nIf expiries exist in TRADES but not in OI_SNAPSHOTS:")
print("  → OI Manager needs to pull from trades table")
print("\nIf recent trades exist but OI Manager doesn't show them:")
print("  → Bug in get_available_expiries() function")