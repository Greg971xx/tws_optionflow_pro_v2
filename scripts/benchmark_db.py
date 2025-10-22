"""
Database Query Benchmark
Measure performance of common queries
"""
import time
import sqlite3
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import DATABASE_PATH


def benchmark_query(name, query, params=None):
    """Benchmark a single query"""
    conn = sqlite3.connect(DATABASE_PATH)

    start = time.time()
    if params:
        df = pd.read_sql(query, conn, params=params)
    else:
        df = pd.read_sql(query, conn)
    end = time.time()

    conn.close()

    duration = end - start
    rows = len(df)
    mb = df.memory_usage(deep=True).sum() / (1024 * 1024)

    return {
        'name': name,
        'duration': duration,
        'rows': rows,
        'memory_mb': mb,
        'rows_per_sec': rows / duration if duration > 0 else 0
    }


def run_benchmarks():
    """Run all database benchmarks"""

    print("=" * 80)
    print("DATABASE QUERY BENCHMARK")
    print("=" * 80)

    benchmarks = []

    # 1. Count trades
    print("\n📊 Counting trades...")
    benchmarks.append(benchmark_query(
        "Count all trades",
        "SELECT COUNT(*) as c FROM trades"
    ))

    # 2. Load ALL trades (worst case)
    print("⚠️  Loading ALL trades (this will be slow)...")
    benchmarks.append(benchmark_query(
        "Load ALL trades",
        "SELECT * FROM trades"
    ))

    # 3. Load trades for specific expiry
    print("📅 Loading trades for one expiry...")
    # Get most recent expiry
    conn = sqlite3.connect(DATABASE_PATH)
    expiry = pd.read_sql("SELECT expiry FROM trades ORDER BY expiry DESC LIMIT 1", conn)['expiry'][0]
    conn.close()

    benchmarks.append(benchmark_query(
        f"Load trades (expiry={expiry})",
        "SELECT * FROM trades WHERE expiry = ?",
        params=[expiry]
    ))

    # 4. Load trades with limit
    print("🎯 Loading trades with LIMIT 10000...")
    benchmarks.append(benchmark_query(
        "Load trades (LIMIT 10000)",
        "SELECT * FROM trades LIMIT 10000"
    ))

    # 5. Aggregate by strike
    print("📊 Aggregating by strike...")
    benchmarks.append(benchmark_query(
        "Aggregate by strike",
        """
        SELECT strike, right, 
               COUNT(*) as trade_count,
               SUM(qty) as total_qty
        FROM trades
        GROUP BY strike, right
        """
    ))

    # 6. Load historical data
    print("📈 Loading SPX historical data...")
    benchmarks.append(benchmark_query(
        "Load SPX data",
        "SELECT * FROM spx_data"
    ))

    # 7. Load OI snapshots
    print("💾 Loading OI snapshots...")
    benchmarks.append(benchmark_query(
        "Load OI snapshots",
        "SELECT * FROM oi_snapshots"
    ))

    # Results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    df_results = pd.DataFrame(benchmarks)

    # Sort by duration
    df_results = df_results.sort_values('duration', ascending=False)

    print(f"\n{'Query':<40} {'Duration':<12} {'Rows':<12} {'MB':<10} {'Rows/s':<12}")
    print("-" * 86)

    for _, row in df_results.iterrows():
        # Color code by duration
        if row['duration'] > 2.0:
            status = "🔴"  # Critical
        elif row['duration'] > 0.5:
            status = "🟡"  # Warning
        else:
            status = "🟢"  # Good

        print(f"{status} {row['name']:<38} "
              f"{row['duration']:>10.2f}s "
              f"{row['rows']:>10,} "
              f"{row['memory_mb']:>8.1f} "
              f"{row['rows_per_sec']:>10,.0f}")

    # Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    slow_queries = df_results[df_results['duration'] > 1.0]

    if not slow_queries.empty:
        print("\n⚠️  Slow Queries Detected:")
        for _, row in slow_queries.iterrows():
            print(f"\n   • {row['name']}")
            print(f"     Duration: {row['duration']:.2f}s")
            print(f"     Rows: {row['rows']:,}")

            # Specific recommendations
            if 'ALL trades' in row['name']:
                print(f"     💡 Solution: NEVER load all trades. Use LIMIT or pagination.")
            elif row['rows'] > 100000:
                print(f"     💡 Solution: Add LIMIT clause or use pagination.")
            elif 'expiry' in row['name'] and row['duration'] > 1.0:
                print(f"     💡 Solution: Add index on expiry column or reduce columns selected.")
    else:
        print("\n✅ All queries are fast (<1s)")

    return df_results


if __name__ == "__main__":
    results = run_benchmarks()