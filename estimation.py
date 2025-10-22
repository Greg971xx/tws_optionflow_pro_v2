"""Check estimation values"""
import sqlite3
from src.config import DATABASE_PATH

conn = sqlite3.connect(DATABASE_PATH)
cursor = conn.cursor()

cursor.execute("""
    SELECT DISTINCT estimation, COUNT(*) as count
    FROM trades
    GROUP BY estimation
    ORDER BY count DESC
""")

print("Estimation values in database:")
for row in cursor.fetchall():
    print(f"  • '{row[0]}': {row[1]:,} rows")

conn.close()