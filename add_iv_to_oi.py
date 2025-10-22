"""
Add IV and Greeks columns to oi_snapshots table
"""
import sqlite3
from src.config import DATABASE_PATH


def upgrade_oi_table():
    """Add new columns to oi_snapshots"""

    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    print("📊 Upgrading oi_snapshots table...")

    # List of columns to add
    new_columns = [
        ('iv', 'REAL'),
        ('delta', 'REAL'),
        ('gamma', 'REAL'),
        ('vega', 'REAL'),
        ('theta', 'REAL'),
        ('bid', 'REAL'),
        ('ask', 'REAL'),
        ('last', 'REAL'),
        ('volume', 'INTEGER')
    ]

    for col_name, col_type in new_columns:
        try:
            cursor.execute(f"ALTER TABLE oi_snapshots ADD COLUMN {col_name} {col_type}")
            print(f"   ✓ Added column: {col_name}")
        except sqlite3.OperationalError as e:
            if "duplicate column name" in str(e):
                print(f"   ⊘ Column already exists: {col_name}")
            else:
                print(f"   ✗ Error adding {col_name}: {e}")

    conn.commit()
    conn.close()

    print("\n✅ Table upgrade complete!")

    # Verify
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(oi_snapshots)")
    columns = cursor.fetchall()

    print("\n📋 Updated schema:")
    for col in columns:
        print(f"   • {col[1]} ({col[2]})")

    conn.close()


if __name__ == "__main__":
    upgrade_oi_table()