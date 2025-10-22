"""
Check what data is stored in oi_snapshots table
"""
import sqlite3
import pandas as pd
from src.config import DATABASE_PATH

conn = sqlite3.connect(DATABASE_PATH)

print("=" * 70)
print("OI_SNAPSHOTS TABLE STRUCTURE")
print("=" * 70)

# Get table schema
cursor = conn.cursor()
cursor.execute("PRAGMA table_info(oi_snapshots)")
columns = cursor.fetchall()

print("\n📋 Columns in oi_snapshots:")
for col in columns:
    print(f"   • {col[1]} ({col[2]})")

# Get sample data
print("\n📊 Sample Data (5 rows):")
df = pd.read_sql("SELECT * FROM oi_snapshots LIMIT 5", conn)
print(df)

print("\n🔍 Column Details:")
for col in df.columns:
    non_null = df[col].notna().sum()
    print(f"   • {col}: {non_null}/5 non-null values")

conn.close()