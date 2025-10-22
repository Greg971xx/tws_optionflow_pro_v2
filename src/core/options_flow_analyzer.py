"""
Options Flow Analyzer - Comprehensive flow metrics and strike analysis
STABLE VERSION - Optimized queries with French label support
"""
import os
import sqlite3
from datetime import datetime
from typing import Dict, List

import pandas as pd
import numpy as np


def load_comprehensive_data(
        db_path: str,
        selected_expiry: str = "ALL",
        sample_size: int = 1000000,
        min_volume_filter: int = 0,
        confidence_threshold: float = 0.6
) -> pd.DataFrame:
    """
    Load and prepare comprehensive options flow data
    OPTIMIZED: Uses LIMIT to avoid loading millions of rows
    """

    conn = sqlite3.connect(db_path)

    try:
        # Optimized queries with LIMIT and French label filter
        if selected_expiry == "ALL":
            query = """
                SELECT * FROM trades 
                WHERE qty > 0
                AND right IN ('C', 'P')
                AND estimation IN ('achat', 'vente', 'indetermine')
                ORDER BY ts DESC 
                LIMIT ?
            """
            df = pd.read_sql(query, conn, params=[sample_size])
        else:
            query = """
                SELECT * FROM trades 
                WHERE expiry = ? 
                AND qty > 0
                AND right IN ('C', 'P')
                AND estimation IN ('achat', 'vente', 'indetermine')
                ORDER BY ts DESC
                LIMIT ?
            """
            df = pd.read_sql(query, conn, params=[selected_expiry, sample_size])

        if df.empty:
            return pd.DataFrame()

        # Preprocess
        df = _preprocess_data(df)

        # Advanced features
        df = _add_advanced_features(df)

        return df

    finally:
        conn.close()


def _preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess raw trade data
    Handles French labels: 'achat', 'vente', 'indetermine'
    """
    df = df.copy()

    # Convert timestamp
    if 'ts' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['ts']):
        df['ts'] = pd.to_datetime(df['ts'])

    # Ensure numeric types
    for col in ['strike', 'qty', 'spot', 'iv', 'delta', 'gamma', 'vega', 'theta', 'signed_qty']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Direction flags (French labels)
    if 'estimation' in df.columns:
        df['estimation'] = df['estimation'].fillna('').astype(str).str.lower()
        df['is_buy'] = (df['estimation'] == 'achat').astype(int)
        df['is_sell'] = (df['estimation'] == 'vente').astype(int)

        # Handle 'indetermine' using signed_qty
        if 'signed_qty' in df.columns:
            indeterminate = df['estimation'] == 'indetermine'
            df.loc[indeterminate & (df['signed_qty'] > 0), 'is_buy'] = 1
            df.loc[indeterminate & (df['signed_qty'] < 0), 'is_sell'] = 1
    else:
        df['is_buy'] = 1
        df['is_sell'] = 0

    # Call/Put flags
    if 'right' in df.columns:
        df['right'] = df['right'].fillna('').astype(str).str.upper()
        df['is_call'] = (df['right'] == 'C').astype(int)
        df['is_put'] = (df['right'] == 'P').astype(int)
    else:
        df['is_call'] = 0
        df['is_put'] = 0

    return df


def _add_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add advanced features for analysis
    All vectorized operations
    """

    # Market session
    if 'ts' in df.columns and pd.api.types.is_datetime64_any_dtype(df['ts']):
        hours = df['ts'].dt.hour
        df['market_session'] = 'afternoon'
        df.loc[hours < 12, 'market_session'] = 'morning'
        df.loc[hours >= 15, 'market_session'] = 'close'
    else:
        df['market_session'] = 'afternoon'

    # Moneyness
    if 'spot' in df.columns and 'strike' in df.columns:
        df['moneyness'] = (df['strike'] / df['spot']) - 1.0
        df['money_class'] = 'ATM'
        df.loc[df['moneyness'] > 0.02, 'money_class'] = 'OTM'
        df.loc[df['moneyness'] < -0.02, 'money_class'] = 'ITM'

    # Volume categories
    if 'qty' in df.columns:
        df['volume_category'] = 'small'
        df.loc[df['qty'] >= 10, 'volume_category'] = 'medium'
        df.loc[df['qty'] >= 50, 'volume_category'] = 'large'
        df.loc[df['qty'] >= 100, 'volume_category'] = 'block'

    return df


def compute_comprehensive_metrics(df: pd.DataFrame) -> Dict:
    """
    Compute comprehensive flow metrics
    Vectorized aggregations
    """

    if df.empty:
        return {}

    call_mask = df['is_call'] == 1
    put_mask = df['is_put'] == 1
    buy_mask = df['is_buy'] == 1
    sell_mask = df['is_sell'] == 1

    metrics = {
        'call_buys': int((call_mask & buy_mask).sum()),
        'call_sells': int((call_mask & sell_mask).sum()),
        'put_buys': int((put_mask & buy_mask).sum()),
        'put_sells': int((put_mask & sell_mask).sum()),
        'call_net_flow': int((call_mask & buy_mask).sum() - (call_mask & sell_mask).sum()),
        'put_net_flow': int((put_mask & buy_mask).sum() - (put_mask & sell_mask).sum()),
    }

    if 'qty' in df.columns:
        metrics['agg_call_buys'] = int(df.loc[call_mask & buy_mask, 'qty'].sum())
        metrics['agg_call_sells'] = int(df.loc[call_mask & sell_mask, 'qty'].sum())
        metrics['agg_put_buys'] = int(df.loc[put_mask & buy_mask, 'qty'].sum())
        metrics['agg_put_sells'] = int(df.loc[put_mask & sell_mask, 'qty'].sum())
        metrics['agg_call_net'] = metrics['agg_call_buys'] - metrics['agg_call_sells']
        metrics['agg_put_net'] = metrics['agg_put_buys'] - metrics['agg_put_sells']

    return metrics


def build_enhanced_strike_metrics(
        df: pd.DataFrame,
        metrics: Dict = None
) -> pd.DataFrame:
    """
    Build enhanced per-strike metrics
    Compatible with flow_tab.py expectations
    """

    if df.empty or 'strike' not in df.columns:
        return pd.DataFrame()

    # Group by strike and right
    result = df.groupby(['strike', 'right']).agg({
        'qty': 'sum',
        'is_buy': 'sum',
        'is_sell': 'sum',
        'iv': 'mean',
        'delta': 'mean',
        'gamma': 'mean',
        'spot': 'first'
    }).reset_index()

    # Rename columns
    result.columns = ['strike', 'right', 'total_qty', 'buys', 'sells', 'avg_iv', 'avg_delta', 'avg_gamma', 'spot']

    # Add derived columns expected by flow_tab
    result['option_type'] = result['right']  # Alias for compatibility
    result['total_aggressive'] = result['total_qty']  # Proxy for aggressive flow
    result['net_flow'] = result['buys'] - result['sells']

    # Fill NaN
    result = result.fillna(0)

    return result


def get_available_expiries(db_path: str) -> List[Dict]:
    """
    Get available expiries from database with metadata
    """
    if not os.path.exists(db_path):
        return []

    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query(
            """
            SELECT DISTINCT expiry, COUNT(*) as count 
            FROM trades 
            WHERE expiry IS NOT NULL 
            AND right IN ('C', 'P')
            GROUP BY expiry 
            ORDER BY expiry DESC
            """,
            conn
        )
        conn.close()

        expiries = []
        today = datetime.now().date()

        for _, row in df.iterrows():
            exp_str = row['expiry']
            count = row['count']

            try:
                exp_date = datetime.strptime(exp_str, '%Y%m%d').date()
                dte = (exp_date - today).days

                if dte == 0:
                    dte_label = "0DTE"
                elif dte < 0:
                    dte_label = f"Expired"
                else:
                    dte_label = f"{dte}DTE"

                label = f"{exp_date} ({dte_label}) - {count:,} trades"

                expiries.append({
                    'value': exp_str,
                    'label': label,
                    'dte': dte
                })
            except:
                expiries.append({
                    'value': exp_str,
                    'label': f"{exp_str} - {count:,} trades",
                    'dte': None
                })

        return expiries

    except Exception as e:
        print(f"Error loading expiries: {e}")
        return []


def analyze_flow_sentiment(df: pd.DataFrame) -> Dict:
    """
    Analyze overall flow sentiment
    """
    if df.empty:
        return {
            'sentiment': 'NEUTRAL',
            'strength': 0,
            'confidence': 0
        }

    metrics = compute_comprehensive_metrics(df)

    call_net = metrics.get('agg_call_net', 0)
    put_net = metrics.get('agg_put_net', 0)
    net_flow = call_net - put_net
    total_flow = abs(call_net) + abs(put_net)

    if total_flow == 0:
        sentiment = 'NEUTRAL'
        strength = 0
    else:
        strength = abs(net_flow) / total_flow

        if net_flow > 100:
            sentiment = 'BULLISH'
        elif net_flow < -100:
            sentiment = 'BEARISH'
        else:
            sentiment = 'NEUTRAL'

    total_trades = sum([
        metrics.get('call_buys', 0),
        metrics.get('call_sells', 0),
        metrics.get('put_buys', 0),
        metrics.get('put_sells', 0)
    ])

    confidence = min(1.0, total_trades / 1000.0)

    return {
        'sentiment': sentiment,
        'strength': strength,
        'confidence': confidence,
        'call_net': call_net,
        'put_net': put_net,
        'net_flow': net_flow
    }


def get_top_strikes(df: pd.DataFrame, top_n: int = 10) -> Dict:
    """
    Get top strikes by volume
    """
    if df.empty or 'strike' not in df.columns:
        return {'calls': [], 'puts': []}

    calls = df[df['is_call'] == 1]
    puts = df[df['is_put'] == 1]

    top_calls = calls.groupby('strike')['qty'].sum().nlargest(top_n).reset_index()
    top_puts = puts.groupby('strike')['qty'].sum().nlargest(top_n).reset_index()

    return {
        'calls': top_calls.to_dict('records'),
        'puts': top_puts.to_dict('records')
    }