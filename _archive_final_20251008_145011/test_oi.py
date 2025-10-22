import sqlite3
import pandas as pd

conn = sqlite3.connect(r"C:\Users\decle\PycharmProjects\flux_claude\db\optionflow.db")

# Voir si des OI ont été collectés
oi = pd.read_sql("SELECT * FROM oi_snapshots WHERE expiry='20251007' LIMIT 100", conn)
print(oi)

conn.close()