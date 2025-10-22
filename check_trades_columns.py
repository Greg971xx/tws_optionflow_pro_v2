"""
Check what columns exist in trades table
"""
import sqlite3
import pandas as pd
from src.config import DATABASE_PATH

conn = sqlite3.connect(DATABASE_PATH)

# Get sample data
df = pd.read_sql("SELECT * FROM trades LIMIT 5", conn)

print("=" * 70)
print("TRADES TABLE COLUMNS")
print("=" * 70)
print("\nColumns found:")
for col in df.columns:
    print(f"  • {col}")

print("\nSample data:")
print(df.head())

print("\nColumn types:")
print(df.dtypes)

conn.close()