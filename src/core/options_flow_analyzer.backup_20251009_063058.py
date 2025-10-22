"""
Options Flow Analyzer
Pure logic for options flow analysis
"""
import os
import sqlite3
from typing import Optional, Dict, List
from datetime import datetime

import numpy as np
import pandas as pd


def load_comprehensive_data(
        db_path: str,
        selected_expiry: str = "ALL",
        sample_size: int = 50000,
        min_volume_filter: int = 0,
        confidence_threshold: float = 0.6
) -> pd.DataFrame:
    """
    Load and prepare comprehensive options flow data
    OPTIMIZED: Uses LIMIT to avoid loading millions of rows
    """

    conn = sqlite3.connect(db_path)

    try:
        # ✅ Optimized queries with LIMIT + filter valid options
        if selected_expiry == "ALL":
            query = """
                SELECT * FROM trades 
                WHERE qty > 0 
                AND right IN ('C', 'P')
                AND estimation IN ('BUY', 'SELL')
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
                AND estimation IN ('BUY', 'SELL')
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
    OPTIMIZED: Vectorized operations
    FIXED: Uses 'estimation' and 'signed_qty' instead of 'direction'
    """
    df = df.copy()

    # Convert timestamp
    if 'ts' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['ts']):
        df['ts'] = pd.to_datetime(df['ts'])

    # Ensure numeric types
    for col in ['strike', 'qty', 'spot', 'iv', 'delta', 'gamma', 'vega', 'theta', 'signed_qty']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # ✅ FIX: Infer direction from 'estimation' or 'signed_qty'
    if 'estimation' in df.columns:
        # Use estimation column (BUY/SELL)
        df['is_buy'] = (df['estimation'].str.upper() == 'BUY').astype(int)
        df['is_sell'] = (df['estimation'].str.upper() == 'SELL').astype(int)
    elif 'signed_qty' in df.columns:
        # Fallback: use signed_qty (positive = buy, negative = sell)
        df['is_buy'] = (df['signed_qty'] > 0).astype(int)
        df['is_sell'] = (df['signed_qty'] < 0).astype(int)
    else:
        # Last resort: assume all buys
        print("⚠️  WARNING: No direction indicator found, assuming all BUYs")
        df['is_buy'] = 1
        df['is_sell'] = 0

    # ✅ Call/Put flags (vectorized)
    if 'right' in df.columns:
        # Clean up right column (handle empty strings)
        df['right'] = df['right'].fillna('').astype(str)
        df['is_call'] = (df['right'].str.upper() == 'C').astype(int)
        df['is_put'] = (df['right'].str.upper() == 'P').astype(int)
    else:
        print("⚠️  WARNING: 'right' column not found")
        df['is_call'] = 0
        df['is_put'] = 0

    return df


def _get_market_session(hour: int) -> str:
    """Determine market session from hour"""
    if pd.isna(hour):
        return "unknown"
    elif hour < 9 or (hour == 9 and hour < 30):
        return "pre_market"
    elif hour >= 16:
        return "after_market"
    else:
        return "regular"


def _add_advanced_features(df: pd.DataFrame, confidence_threshold: float) -> pd.DataFrame:
    """Add advanced features for flow analysis"""
    is_call = df["option_type"] == "CALL"
    is_put = df["option_type"] == "PUT"
    is_buy = df["estimation"] == "achat"
    is_sell = df["estimation"] == "vente"

    df.loc[:, "is_buy"] = is_buy.astype(int)
    df.loc[:, "is_sell"] = is_sell.astype(int)
    df.loc[:, "side_sign"] = np.where(df["estimation"] == "achat", 1,
                                      np.where(df["estimation"] == "vente", -1, 0))

    # Greeks
    df.loc[:, "client_delta"] = df["delta"].astype(float) * df["side_sign"] * df["qty"] * 100.0
    df.loc[:, "client_gamma"] = df["gamma"].astype(float) * df["side_sign"] * df["qty"] * 100.0
    df.loc[:, "mm_gamma"] = -df["client_gamma"]

    # Aggressiveness
    df.loc[:, "is_aggressive"] = (df["confidence"] > 0.8).astype(int)
    df.loc[:, "is_aggressive_buy"] = (is_buy & (df["confidence"] > 0.8)).astype(int)
    df.loc[:, "is_aggressive_sell"] = (is_sell & (df["confidence"] > 0.8)).astype(int)

    # Volumes
    df.loc[:, "buy_volume"] = df["qty"] * df["is_buy"]
    df.loc[:, "sell_volume"] = df["qty"] * df["is_sell"]
    df.loc[:, "aggressive_buy_volume"] = df["qty"] * df["is_aggressive_buy"]
    df.loc[:, "aggressive_sell_volume"] = df["qty"] * df["is_aggressive_sell"]
    df.loc[:, "net_volume"] = df["buy_volume"] - df["sell_volume"]
    df.loc[:, "aggressive_net_volume"] = df["aggressive_buy_volume"] - df["aggressive_sell_volume"]

    # Directional
    df.loc[:, "bullish_component"] = (
            np.where(is_call & is_buy, df["qty"], 0) +
            np.where(is_put & is_sell, df["qty"], 0)
    )
    df.loc[:, "bearish_component"] = (
            np.where(is_put & is_buy, df["qty"], 0) +
            np.where(is_call & is_sell, df["qty"], 0)
    )
    df.loc[:, "directional_signed"] = df["bullish_component"] - df["bearish_component"]

    # Notionals
    df.loc[:, "notional"] = df["last"] * df["qty"] * 100.0
    df.loc[:, "aggressive_buy_notional"] = df["notional"] * df["is_aggressive_buy"]
    df.loc[:, "aggressive_sell_notional"] = df["notional"] * df["is_aggressive_sell"]

    df.loc[:, "spread"] = df["ask"] - df["bid"]
    df.loc[:, "spread_pct"] = df["spread"] / df["last"].clip(lower=0.01)
    df.loc[:, "mid"] = (df["bid"] + df["ask"]) / 2.0

    return df


def compute_comprehensive_metrics(df: pd.DataFrame) -> Dict[str, any]:
    """
    Compute cumulative flow metrics
    These metrics should NEVER decrease for a given expiry
    """
    if df.empty:
        return {
            'call_buys': 0, 'call_sells': 0,
            'put_buys': 0, 'put_sells': 0,
            'call_net_flow': 0, 'put_net_flow': 0,
            'agg_call_buys': 0, 'agg_call_sells': 0,
            'agg_put_buys': 0, 'agg_put_sells': 0,
            'agg_call_net': 0, 'agg_put_net': 0,
        }

    # Cumulative counts (by number of trades)
    call_buys = int(len(df[(df["option_type"] == "CALL") & (df["estimation"] == "achat")]))
    call_sells = int(len(df[(df["option_type"] == "CALL") & (df["estimation"] == "vente")]))
    put_buys = int(len(df[(df["option_type"] == "PUT") & (df["estimation"] == "achat")]))
    put_sells = int(len(df[(df["option_type"] == "PUT") & (df["estimation"] == "vente")]))

    # Cumulative volumes (by quantity)
    agg_call_buys = int(df["aggressive_buy_volume"][df["option_type"] == "CALL"].sum())
    agg_call_sells = int(df["aggressive_sell_volume"][df["option_type"] == "CALL"].sum())
    agg_put_buys = int(df["aggressive_buy_volume"][df["option_type"] == "PUT"].sum())
    agg_put_sells = int(df["aggressive_sell_volume"][df["option_type"] == "PUT"].sum())

    return {
        'call_buys': call_buys,
        'call_sells': call_sells,
        'put_buys': put_buys,
        'put_sells': put_sells,
        'call_net_flow': call_buys - call_sells,
        'put_net_flow': put_buys - put_sells,
        'agg_call_buys': agg_call_buys,
        'agg_call_sells': agg_call_sells,
        'agg_put_buys': agg_put_buys,
        'agg_put_sells': agg_put_sells,
        'agg_call_net': agg_call_buys - agg_call_sells,
        'agg_put_net': agg_put_buys - agg_put_sells,
    }


def build_enhanced_strike_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Build strike-level aggregations"""
    if df.empty:
        return pd.DataFrame()

    strike_agg = df.groupby(["strike", "option_type"]).agg({
        'qty': 'sum',
        'buy_volume': 'sum',
        'sell_volume': 'sum',
        'aggressive_buy_volume': 'sum',
        'aggressive_sell_volume': 'sum',
        'aggressive_net_volume': 'sum',
        'net_volume': 'sum',
        'confidence': 'mean',
    }).reset_index()

    strike_agg['total_aggressive'] = (
            strike_agg['aggressive_buy_volume'] +
            strike_agg['aggressive_sell_volume']
    )
    strike_agg['buy_dominance'] = (
            strike_agg['aggressive_buy_volume'] /
            strike_agg['total_aggressive'].clip(lower=1)
    )

    return strike_agg.sort_values(['option_type', 'strike'])


def get_available_expiries(db_path: str) -> List[Dict]:
    """
    Get available expiries from database with metadata
    Returns list of dicts: [{'value': '20251007', 'label': '2025-10-07 (0DTE) - 1,234 trades', 'dte': 0}, ...]
    """
    if not os.path.exists(db_path):
        return []

    try:
        con = sqlite3.connect(db_path)
        df = pd.read_sql_query(
            """
            SELECT DISTINCT expiry, COUNT(*) as count 
            FROM trades 
            WHERE expiry IS NOT NULL 
            GROUP BY expiry 
            ORDER BY expiry DESC
            """,
            con
        )
        con.close()

        expiries = []
        today = datetime.now().date()

        for _, row in df.iterrows():
            exp_str = row['expiry']
            count = row['count']

            try:
                exp_date = datetime.strptime(exp_str, '%Y%m%d').date()
                dte = (exp_date - today).days
                label = f"{exp_date} ({dte}DTE) - {count:,} trades"
                expiries.append({
                    'value': exp_str,
                    'label': label,
                    'dte': dte
                })
            except Exception:
                # Fallback for malformed dates
                expiries.append({
                    'value': exp_str,
                    'label': f"{exp_str} - {count:,} trades",
                    'dte': None
                })

        return expiries

    except Exception as e:
        print(f"Error loading expiries: {e}")
        return []