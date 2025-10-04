import sqlite3
import pandas as pd
import os

def update_rank_percentile_252(db_path='db/market_data.db'):
    print("🔄 Mise à jour des colonnes rank252_ et percentile252_...")
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql("SELECT * FROM volatility_stats", conn)
        df['date'] = pd.to_datetime(df['date'])
        tickers = df['ticker'].unique()

        columns_to_process = [
            'ma20_c2c', 'ma20_o2c', 'ma60_c2c', 'ma60_o2c',
            'ma120_c2c', 'ma120_o2c', 'ma252_c2c', 'ma252_o2c',
            'hv5', 'hv20', 'hv60', 'hv120', 'hv252'
        ]

        result = []

        for ticker in tickers:
            df_ticker = df[df['ticker'] == ticker].copy()
            df_ticker.sort_values('date', inplace=True)
            for col in columns_to_process:
                rank_col = f'rank252_{col}'
                perc_col = f'percentile252_{col}'
                df_ticker[rank_col] = df_ticker[col].rolling(252).apply(
                    lambda x: (x[-1] - x.min()) / (x.max() - x.min()) if (x.max() - x.min()) != 0 else 0,
                    raw=True
                )
                df_ticker[perc_col] = df_ticker[col].rolling(252).apply(
                    lambda x: (x < x[-1]).mean(),
                    raw=True
                )
            result.append(df_ticker)

        df_final = pd.concat(result)
        df_final.drop(columns=[col for col in df_final.columns if col not in ['date', 'ticker'] + columns_to_process + [f'rank252_{c}' for c in columns_to_process] + [f'percentile252_{c}' for c in columns_to_process]], inplace=True)

        # On met à jour en base
        df_existing = pd.read_sql("SELECT * FROM volatility_stats", conn)
        df_existing.set_index(['date', 'ticker'], inplace=True)
        df_final.set_index(['date', 'ticker'], inplace=True)

        for col in df_final.columns:
            df_existing[col] = df_final[col]

        df_existing.reset_index(inplace=True)
        df_existing.to_sql("volatility_stats", conn, if_exists='replace', index=False)
        conn.close()
        print("✅ Mise à jour terminée.")
    except Exception as e:
        print(f"❌ Erreur durant la mise à jour : {e}")
