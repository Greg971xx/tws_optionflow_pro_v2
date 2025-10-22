# Test rapide
import sqlite3
import pandas as pd

db_path = r"C:\Users\decle\PycharmProjects\tws_optionflow_pro_v2\flux_claude\db\optionflow.db"
conn = sqlite3.connect(db_path)

# Liste toutes les tables
tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)
print("Tables dans optionflow.db:")
print(tables)

conn.close()