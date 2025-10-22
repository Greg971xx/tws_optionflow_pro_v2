"""
Verify configuration is correct after migration
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def verify():
    """Check all imports work correctly"""

    print("🔍 Verifying configuration...\n")

    try:
        from src.config import (
            DATABASE_PATH,
            MARKET_DATA_DB,
            OPTIONFLOW_DB,
            PROJECT_ROOT,
            DB_DIR
        )

        print("✅ Configuration imports successful")

        print(f"\n📁 Paths:")
        print(f"   Project Root: {PROJECT_ROOT}")
        print(f"   DB Directory: {DB_DIR}")
        print(f"   Database:     {DATABASE_PATH}")

        # Check they're all the same
        if MARKET_DATA_DB == OPTIONFLOW_DB == DATABASE_PATH:
            print(f"\n✅ All paths point to same database (correct)")
        else:
            print(f"\n❌ Paths are different (error)")
            return False

        # Check database exists
        db_path = Path(DATABASE_PATH)
        if db_path.exists():
            size = db_path.stat().st_size / (1024 * 1024)
            print(f"\n✅ Database exists: {size:.2f} MB")

            # Check tables
            import sqlite3
            import pandas as pd

            conn = sqlite3.connect(DATABASE_PATH)
            tables = pd.read_sql(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'",
                conn
            )

            print(f"\n📊 Tables ({len(tables)}):")

            critical = ['trades', 'oi_snapshots', 'spx_data', 'vix_data']
            for table_name in critical:
                if table_name in tables['name'].values:
                    count = pd.read_sql(f"SELECT COUNT(*) as c FROM {table_name}", conn)
                    print(f"   ✓ {table_name}: {count['c'][0]:,} rows")
                else:
                    print(f"   ✗ {table_name}: NOT FOUND")

            conn.close()

            return True
        else:
            print(f"\n❌ Database not found: {DATABASE_PATH}")
            return False

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = verify()
    sys.exit(0 if success else 1)