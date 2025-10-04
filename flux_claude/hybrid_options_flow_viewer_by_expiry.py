# =============================================
# File: hybrid_options_flow_viewer_by_expiry.py
# Description: Hybrid viewer combining GPT-5 clarity with comprehensive analytics
# Features: Rich flow analysis + Advanced GEX + Market maker insights
# =============================================

# --- MUST BE FIRST: ensure asyncio loop exists for eventkit/ib_insync ---
import asyncio
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

try:
    import nest_asyncio
    nest_asyncio.apply()
except Exception:
    pass
# -----------------------------------------------------------------------

import os
import sqlite3
from typing import Optional, Dict
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import streamlit as st


from oi_utils import (
    ensure_oi_schema,
    fetch_oi_snapshot_for_expiry,
    merge_latest_oi_into_strike_metrics,
    get_oi_snapshot_meta,
)

# Import our enhanced GEX module
try:
    from enhanced_gex_analysis import render_enhanced_gex_component, EnhancedGEXAnalyzer
    HAS_ENHANCED_GEX = True
except Exception as e:
    HAS_ENHANCED_GEX = False
    st.warning(f"Enhanced GEX disabled: {e}")

# ============= Page Configuration =============
st.set_page_config(
    page_title="Hybrid SPX Options Flow Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============= Sidebar Configuration =============
st.sidebar.title("⚖️ Options Flow Analyzer")
st.sidebar.markdown("*Combining clarity with analytical depth*")

# Database and sampling settings
DEFAULT_DB_PATH = r"C:\Users\decle\PycharmProjects\flux_claude\db\optionflow.db"  # << UNIFIÉ
if "db_path" not in st.session_state:
    st.session_state["db_path"] = DEFAULT_DB_PATH

db_path = st.sidebar.text_input("Database Path", st.session_state["db_path"])
st.session_state["db_path"] = db_path

# ===== Expiry-Only Sidebar & Helpers =====
@st.cache_data(ttl=300)
def get_available_expiries(db_path: str) -> list:
    if not os.path.exists(db_path):
        return []
    try:
        con = sqlite3.connect(db_path)
        df = pd.read_sql_query(
            "SELECT DISTINCT expiry, COUNT(*) as count FROM trades WHERE expiry IS NOT NULL GROUP BY expiry ORDER BY expiry",
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
                expiries.append({'value': exp_str, 'label': label, 'dte': dte})
            except Exception:
                continue
        return expiries
    except Exception:
        return []

expiry_options = get_available_expiries(db_path)

if expiry_options:
    selected_expiry = st.sidebar.selectbox(
        "Select Expiry:",
        options=['ALL'] + [exp['value'] for exp in expiry_options],
        format_func=lambda x: "All Expiries" if x == 'ALL' else next(
            (exp['label'] for exp in expiry_options if exp['value'] == x), x
        )
    )
else:
    selected_expiry = 'ALL'
    st.sidebar.warning("No expiry data found")

st.sidebar.markdown("---")
use_sampling = st.sidebar.checkbox("Use sampling", value=True)
sample_size = st.sidebar.slider("Sample size", 10000, 500000, 100000) if use_sampling else None
min_volume = st.sidebar.number_input("Min volume per trade", 0, 1000, 0)
confidence_min = st.sidebar.slider("Min confidence", 0.0, 1.0, 0.6, 0.05)

# Map to original variable names used by the rest of the code to remain unchanged
min_volume_filter = int(min_volume)
confidence_threshold = float(confidence_min)

# Advanced settings
with st.sidebar.expander("🔧 Advanced Settings"):
    aggressive_threshold = st.sidebar.slider("Aggressive trade threshold", 0.0, 1.0, 0.80, 0.05)
    show_debug = st.sidebar.checkbox("Show debug info", value=False)
    enable_real_time = st.sidebar.checkbox("Enable real-time refresh", value=False)
    if enable_real_time:
        refresh_interval = st.sidebar.slider("Refresh interval (seconds)", 10, 300, 30)

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Force Refresh Data"):
    st.cache_data.clear()
    st.rerun()

# --- IB settings (pour relever l’OI) ---
with st.sidebar.expander("🔌 IB (relever l’OI)"):
    ib_host = st.text_input("Host", "127.0.0.1", key="ib_host")
    ib_port = st.number_input("Port", 7400, 9000, 7497, step=1, key="ib_port")
    ib_cid  = st.number_input("Client ID", 0, 50, 21, step=1, key="ib_cid")

if st.sidebar.button("📥 Fetch OI now (échéance sélectionnée)"):
    if selected_expiry == "ALL":
        st.error("Sélectionne une échéance précise pour relever l’OI.")
    else:
        with st.spinner(f"Relevé de l’OI en cours pour {selected_expiry}…"):
            try:
                ensure_oi_schema(db_path)
                ok = fetch_oi_snapshot_for_expiry(
                    db_path=db_path,
                    expiry=selected_expiry,
                    host=ib_host,
                    port=int(ib_port),
                    client_id=int(ib_cid),
                    exchange="CBOE",     # On commence par CBOE ; la fonction sait fallback si tu as mis try_smart_on_empty=True
                    batch_size=50,
                    pause=0.35,
                    timeout_s=8.0,
                    retries=1,
                    try_smart_on_empty=True,
                    debug=True,
                )
                if ok:
                    st.success("✅ OI relevé et enregistré. Rafraîchissement…")
                    st.cache_data.clear()
                    st.rerun()
                else:
                    st.warning("Aucun OI inséré (abonnement IB, marché fermé, ou aucun contrat qualifié).")
            except Exception as e:
                st.exception(e)



# ============= Enhanced Data Loading =============
@st.cache_data(ttl=60, show_spinner=False)
def load_comprehensive_data(db_path: str, selected_expiry: str, sample_size: Optional[int],
                            min_volume_filter: int, confidence_threshold: float) -> pd.DataFrame:
    """
    Charge les données en filtrant uniquement par échéance (expiry).
    - Si 'ALL' → pas de filtre.
    - Sinon → filtre sur les colonnes existantes : 'expiry' ET/OU 'lastTradeDateOrContractMonth'.
    - Normalise l'entrée (YYYY-MM-DD ou YYYYMMDD) en digits uniquement pour matcher toutes les formes.
    """
    if not os.path.exists(db_path):
        return pd.DataFrame()

    con = sqlite3.connect(db_path)
    try:
        cols = set(pd.read_sql("PRAGMA table_info(trades);", con)['name'].tolist())
        has_expiry = 'expiry' in cols
        has_ib = 'lastTradeDateOrContractMonth' in cols

        params = []
        if selected_expiry and selected_expiry != 'ALL':
            norm = "".join(ch for ch in str(selected_expiry) if ch.isdigit())
            where_clauses = []
            if has_expiry:
                where_clauses.append(
                    "REPLACE(REPLACE(REPLACE(COALESCE(expiry, ''), '-', ''), '/', ''), '.', '') = ?"
                )
                params.append(norm)
            if has_ib:
                where_clauses.append(
                    "REPLACE(REPLACE(REPLACE(COALESCE(lastTradeDateOrContractMonth, ''), '-', ''), '/', ''), '.', '') = ?"
                )
                params.append(norm)

            if where_clauses:
                query = "SELECT * FROM trades WHERE (" + " OR ".join(where_clauses) + ") ORDER BY ts DESC"
            else:
                query = "SELECT * FROM trades ORDER BY ts DESC"
        else:
            query = "SELECT * FROM trades ORDER BY ts DESC"

        if sample_size:
            query += f" LIMIT {int(sample_size)}"

        df = pd.read_sql(query, con, params=params)
    finally:
        con.close()

    if df.empty:
        return df

    df = _preprocess_data(df, min_volume_filter, confidence_threshold)
    return df

def _preprocess_data(df: pd.DataFrame, min_volume_filter: int, confidence_threshold: float) -> pd.DataFrame:
    """Comprehensive data preprocessing with advanced feature engineering"""
    # Temporal processing
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
        df["minute"] = df["ts"].dt.floor("min")
        df["hour"] = df["ts"].dt.floor("h")
        df["market_session"] = df["ts"].dt.hour.apply(_get_market_session)
    else:
        df["minute"] = pd.NaT
        df["hour"] = pd.NaT
        df["market_session"] = "unknown"

    # Standardize option type
    if "right" in df.columns:
        df["option_type"] = np.where(df["right"].astype(str).str.upper() == "C", "CALL", "PUT")
    elif "type" in df.columns:
        df["option_type"] = df["type"].astype(str).str.upper().map({
            "CALL": "CALL", "PUT": "PUT", "C": "CALL", "P": "PUT", "call": "CALL", "put": "PUT"
        })
    else:
        df["option_type"] = "UNKNOWN"

    # Clean and standardize estimation
    if "estimation" in df.columns:
        df["estimation"] = df["estimation"].astype(str).str.lower().replace({
            'indetermine': 'indéterminé',
            'indeterminate': 'indéterminé'
        })
    else:
        df["estimation"] = "indéterminé"

    # Ensure numeric columns
    numeric_cols = ["qty", "signed_qty", "strike", "spot", "confidence", "last", "bid", "ask", "delta", "gamma"]
    for col in numeric_cols:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    # Filters
    df = df.copy()
    if min_volume_filter > 0:
        df = df[df["qty"] >= float(min_volume_filter)]
    if confidence_threshold > 0:
        df = df[(df["estimation"].isin(["achat", "vente"])) & (df["confidence"] >= float(confidence_threshold))]

    # Features
    df = _add_advanced_features(df, confidence_threshold)
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
    """Add advanced features for comprehensive analysis"""
    is_call = df["option_type"] == "CALL"
    is_put  = df["option_type"] == "PUT"
    is_buy  = df["estimation"] == "achat"
    is_sell = df["estimation"] == "vente"

    # Basic directional components
    df.loc[:, "is_buy"]  = is_buy.astype(int)
    df.loc[:, "is_sell"] = is_sell.astype(int)

    # Side sign (+1 buy, -1 sell)
    df.loc[:, "side_sign"] = np.where(df["estimation"] == "achat", 1,
                               np.where(df["estimation"] == "vente", -1, 0))

    # Client delta/gamma & dealer gamma
    df.loc[:, "client_delta"] = df["delta"].astype(float) * df["side_sign"] * df["qty"] * 100.0
    df.loc[:, "client_gamma"] = df["gamma"].astype(float) * df["side_sign"] * df["qty"] * 100.0
    df.loc[:, "mm_gamma"]     = -df["client_gamma"]

    # Aggressiveness
    df.loc[:, "is_aggressive"]      = (df["confidence"] > 0.8).astype(int)
    df.loc[:, "is_aggressive_buy"]  = (is_buy  & (df["confidence"] > 0.8)).astype(int)
    df.loc[:, "is_aggressive_sell"] = (is_sell & (df["confidence"] > 0.8)).astype(int)

    # Volumes
    df.loc[:, "buy_volume"]             = df["qty"] * df["is_buy"]
    df.loc[:, "sell_volume"]            = df["qty"] * df["is_sell"]
    df.loc[:, "aggressive_buy_volume"]  = df["qty"] * df["is_aggressive_buy"]
    df.loc[:, "aggressive_sell_volume"] = df["qty"] * df["is_aggressive_sell"]

    # Net flows
    df.loc[:, "net_volume"]            = df["buy_volume"] - df["sell_volume"]
    df.loc[:, "aggressive_net_volume"] = df["aggressive_buy_volume"] - df["aggressive_sell_volume"]

    # Directional components
    df.loc[:, "bullish_component"] = (
        np.where(is_call & is_buy, df["qty"], 0) +
        np.where(is_put  & is_sell, df["qty"], 0)
    )
    df.loc[:, "bearish_component"] = (
        np.where(is_put & is_buy, df["qty"], 0) +
        np.where(is_call & is_sell, df["qty"], 0)
    )
    df.loc[:, "directional_signed"] = df["bullish_component"] - df["bearish_component"]

    # Notionals & microstructure
    df.loc[:, "notional"]                 = df["last"] * df["qty"] * 100.0
    df.loc[:, "aggressive_buy_notional"]  = df["notional"] * df["is_aggressive_buy"]
    df.loc[:, "aggressive_sell_notional"] = df["notional"] * df["is_aggressive_sell"]

    df.loc[:, "spread"]     = df["ask"] - df["bid"]
    df.loc[:, "spread_pct"] = df["spread"] / df["last"].clip(lower=0.01)
    df.loc[:, "mid"]        = (df["bid"] + df["ask"]) / 2.0

    return df

# ============= Enhanced Analytics Functions =============
@st.cache_data(show_spinner=False)
def compute_comprehensive_metrics(df: pd.DataFrame) -> Dict[str, any]:
    """Compute comprehensive flow metrics with enhanced market maker insights"""
    if df.empty:
        return {}
    call_buys  = int(df[(df["option_type"] == "CALL") & (df["estimation"] == "achat")]["qty"].sum())
    call_sells = int(df[(df["option_type"] == "CALL") & (df["estimation"] == "vente")]["qty"].sum())
    put_buys   = int(df[(df["option_type"] == "PUT")  & (df["estimation"] == "achat")]["qty"].sum())
    put_sells  = int(df[(df["option_type"] == "PUT")  & (df["estimation"] == "vente")]["qty"].sum())

    agg_call_buys  = int(df["aggressive_buy_volume"][df["option_type"] == "CALL"].sum())
    agg_call_sells = int(df["aggressive_sell_volume"][df["option_type"] == "CALL"].sum())
    agg_put_buys   = int(df["aggressive_buy_volume"][df["option_type"] == "PUT"].sum())
    agg_put_sells  = int(df["aggressive_sell_volume"][df["option_type"] == "PUT"].sum())

    call_net_flow = call_buys - call_sells
    put_net_flow  = put_buys  - put_sells
    agg_call_net  = agg_call_buys - agg_call_sells
    agg_put_net   = agg_put_buys  - agg_put_sells

    total_bullish = int(df["bullish_component"].sum())
    total_bearish = int(df["bearish_component"].sum())
    directional_flow = total_bullish - total_bearish

    total_volume = int(df["qty"].sum())
    total_aggressive_volume = int(df[df["is_aggressive"] == 1]["qty"].sum())
    aggressive_ratio = total_aggressive_volume / max(total_volume, 1)

    total_notional = df["notional"].sum()
    aggressive_notional = df["aggressive_buy_notional"].sum() + df["aggressive_sell_notional"].sum()

    session_metrics = df.groupby('market_session').agg({
        'qty': 'sum',
        'aggressive_net_volume': 'sum'
    }).to_dict()

    return {
        'call_buys': call_buys, 'call_sells': call_sells,
        'put_buys': put_buys, 'put_sells': put_sells,
        'call_net_flow': call_net_flow, 'put_net_flow': put_net_flow,
        'agg_call_buys': agg_call_buys, 'agg_call_sells': agg_call_sells,
        'agg_put_buys': agg_put_buys, 'agg_put_sells': agg_put_sells,
        'agg_call_net': agg_call_net, 'agg_put_net': agg_put_net,
        'total_bullish': total_bullish, 'total_bearish': total_bearish,
        'directional_flow': directional_flow,
        'total_volume': total_volume, 'total_aggressive_volume': total_aggressive_volume,
        'aggressive_ratio': aggressive_ratio,
        'total_notional': total_notional, 'aggressive_notional': aggressive_notional,
        'session_metrics': session_metrics
    }

@st.cache_data(show_spinner=False)
def build_enhanced_strike_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    strike_agg = df.groupby(["strike", "option_type"]).agg({
        # Volume metrics
        'qty': 'sum',
        'buy_volume': 'sum',
        'sell_volume': 'sum',
        'aggressive_buy_volume': 'sum',
        'aggressive_sell_volume': 'sum',
        'aggressive_net_volume': 'sum',
        'net_volume': 'sum',

        # Directional components
        'bullish_component': 'sum',
        'bearish_component': 'sum',
        'directional_signed': 'sum',

        # Quality metrics
        'confidence': 'mean',
        'is_aggressive': 'sum',

        # Notional
        'notional': 'sum',
        'aggressive_buy_notional': 'sum',
        'aggressive_sell_notional': 'sum',

        # Market structure
        'spread_pct': 'mean',

        # Count / Greeks
        'ts': 'count',
        'client_delta': 'sum',
        'client_gamma': 'sum',
        'mm_gamma': 'sum',
    }).reset_index()

    strike_agg = strike_agg.rename(columns={'ts': 'trade_count'})

    strike_agg['total_aggressive'] = strike_agg['aggressive_buy_volume'] + strike_agg['aggressive_sell_volume']
    strike_agg['buy_ratio'] = strike_agg['buy_volume'] / strike_agg['qty'].clip(lower=1)
    strike_agg['sell_ratio'] = strike_agg['sell_volume'] / strike_agg['qty'].clip(lower=1)
    strike_agg['aggressive_buy_ratio'] = strike_agg['aggressive_buy_volume'] / strike_agg['qty'].clip(lower=1)
    strike_agg['aggressive_sell_ratio'] = strike_agg['aggressive_sell_volume'] / strike_agg['qty'].clip(lower=1)
    strike_agg['aggressive_ratio'] = strike_agg['is_aggressive'] / strike_agg['qty'].clip(lower=1)

    strike_agg['buy_dominance'] = (
        strike_agg['aggressive_buy_volume'] /
        (strike_agg['total_aggressive']).clip(lower=1)
    )
    strike_agg['sell_dominance'] = 1 - strike_agg['buy_dominance']

    strike_agg['flow_intensity'] = strike_agg['confidence'] * strike_agg['qty']
    strike_agg['notional_per_contract'] = strike_agg['notional'] / strike_agg['qty'].clip(lower=1)

    return strike_agg.sort_values(['option_type', 'strike'])

# ============= Enhanced Visualization Functions =============
def create_comprehensive_flow_heatmap(df: pd.DataFrame) -> go.Figure:
    """Create enhanced flow heatmap with multiple dimensions"""
    if df.empty or 'minute' not in df.columns or df['minute'].isna().all():
        return go.Figure().add_annotation(text="No time data available for heatmap")

    heat_data = (
        df.groupby(['minute', 'strike', 'option_type'])
        .agg({'aggressive_net_volume': 'sum'})
        .reset_index()
    )

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("CALL Aggressive Net Flow", "PUT Aggressive Net Flow"),
        shared_xaxes=True,
        vertical_spacing=0.1
    )

    for i, option_type in enumerate(['CALL', 'PUT'], 1):
        data = heat_data[heat_data['option_type'] == option_type]
        if not data.empty:
            pivot_data = data.pivot_table(
                index='strike', columns='minute',
                values='aggressive_net_volume', fill_value=0
            )
            fig.add_trace(
                go.Heatmap(
                    z=pivot_data.values,
                    x=pivot_data.columns.astype(str),
                    y=pivot_data.index.astype(str),
                    colorscale='RdBu',
                    zmid=0,
                    name=f'{option_type} Flow',
                    showscale=True,
                    colorbar=dict(title="Net Flow", x=1.02 if i == 1 else 1.08)
                ),
                row=i, col=1
            )

    fig.update_layout(
        title="Enhanced Order Flow Heatmap (Green=Buy Pressure, Red=Sell Pressure)",
        height=700
    )
    return fig

def create_strike_analysis_chart(strike_metrics: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """Create comprehensive strike analysis visualization"""
    if strike_metrics.empty:
        return go.Figure().add_annotation(text="No strike data available")

    top_strikes = (
        strike_metrics.nlargest(top_n, 'total_aggressive')
        .sort_values('strike')
    )

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Volume Analysis by Strike", "Net Flow & Dominance Analysis"),
        specs=[[{"secondary_y": True}], [{"secondary_y": True}]],
        vertical_spacing=0.15
    )

    for option_type in ['CALL', 'PUT']:
        data = top_strikes[top_strikes['option_type'] == option_type]
        if data.empty:
            continue
        color = 'green' if option_type == 'CALL' else 'red'

        fig.add_trace(
            go.Bar(
                x=data['strike'],
                y=data['aggressive_buy_volume'],
                name=f'{option_type} Aggressive Buys',
                marker_color=color,
                opacity=0.7
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(
                x=data['strike'],
                y=-data['aggressive_sell_volume'],
                name=f'{option_type} Aggressive Sells',
                marker_color=color,
                opacity=0.5
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=data['strike'],
                y=data['aggressive_net_volume'],
                mode='lines+markers',
                name=f'{option_type} Net Flow',
                line=dict(color='blue' if option_type == 'CALL' else 'purple', width=3)
            ),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=data['strike'],
                y=data['buy_dominance'],
                mode='lines+markers',
                name=f'{option_type} Buy Dominance',
                line=dict(color='orange' if option_type == 'CALL' else 'brown', width=2),
            ),
            row=2, col=1, secondary_y=True
        )

    fig.update_layout(
        title=f"Comprehensive Strike Analysis - Top {top_n} Most Active",
        height=800,
        barmode='relative'
    )
    fig.update_yaxes(title_text="Volume", row=1, col=1)
    fig.update_yaxes(title_text="Net Flow", row=2, col=1)
    fig.update_yaxes(title_text="Buy Dominance %", secondary_y=True, row=2, col=1)
    return fig

# ============= Main Application =============
def main():
    st.title("Hybrid SPX Options Flow Analyzer")
    st.markdown("*Advanced analytics with clean interface*")

    # Load data
    with st.spinner("Loading comprehensive data..."):
        df = load_comprehensive_data(db_path, selected_expiry, sample_size, min_volume_filter, confidence_threshold)

    if df.empty:
        st.error("No data loaded. Check database path, date filter, and other settings.")
        st.stop()

    # Compute metrics
    metrics = compute_comprehensive_metrics(df)
    strike_metrics = build_enhanced_strike_metrics(df)

    # Merge latest OI (safe even if expiry == ALL)
    strike_metrics = merge_latest_oi_into_strike_metrics(
        strike_metrics=strike_metrics,
        db_path=db_path,
        expiry=selected_expiry
    )
    if "open_interest" not in strike_metrics.columns:
        strike_metrics["open_interest"] = np.nan

    # --- OI snapshot panel ---
    # --- OI snapshot panel (si une échéance précise est sélectionnée) ---
    if selected_expiry != "ALL":
        last_ts, n_total, n_calls, n_puts = get_oi_snapshot_meta(db_path, selected_expiry)
        with st.container():
            c1, c2, c3 = st.columns(3)
            c1.metric("Dernier snapshot OI", last_ts or "—")
            c2.metric("Contrats couverts (OI)", f"{n_total:,}")
            c3.metric("Répartition", f"C:{n_calls:,} / P:{n_puts:,}")

        oi_plot = strike_metrics.dropna(subset=["open_interest"])
        if not oi_plot.empty and oi_plot["open_interest"].abs().sum() > 0:
            oi_plot = (
                oi_plot.groupby(["strike", "option_type"])["open_interest"]
                .sum().reset_index()
            )
            fig_oi = go.Figure()
            for ot, name in [("CALL", "Calls"), ("PUT", "Puts")]:
                d = oi_plot[oi_plot["option_type"] == ot].sort_values("strike")
                if not d.empty:
                    fig_oi.add_trace(go.Bar(x=d["strike"], y=d["open_interest"], name=name))
            fig_oi.update_layout(
                title=f"Open Interest par strike — {selected_expiry}",
                barmode="overlay",
                height=420
            )
            st.plotly_chart(fig_oi, use_container_width=True)
        else:
            st.info("Pas d’OI fusionné pour afficher un graphe. Clique sur **📥 Fetch OI now** dans la sidebar.")

    # ============= Executive Dashboard =============
    st.header("Executive Flow Summary")

    # Spot block
    with st.container():
        c1, c2 = st.columns([1, 3])

        last_price = None
        if "spot" in df.columns:
            spot_series_desc = df["spot"].replace(0, np.nan).dropna()
            if not spot_series_desc.empty:
                last_price = float(spot_series_desc.iloc[0])

        with c1:
            st.metric("Dernier spot SPX", f"{last_price:,.2f}" if last_price is not None else "—")

        with c2:
            if "minute" in df.columns and "spot" in df.columns and not df["minute"].isna().all():
                price_df = (
                    df.dropna(subset=["minute", "spot"])
                    .replace({"spot": {0: np.nan}})
                    .dropna(subset=["spot"])
                    .groupby("minute", as_index=False)["spot"].median()
                    .sort_values("minute")
                )
                if not price_df.empty:
                    fig_price = go.Figure()
                    fig_price.add_trace(
                        go.Scatter(
                            x=price_df["minute"], y=price_df["spot"],
                            mode="lines", name="SPX Spot", line=dict(width=2)
                        )
                    )
                    if last_price is not None:
                        fig_price.add_hline(y=last_price, line_dash="dot",
                                            annotation_text="Dernier",
                                            annotation_position="top left")
                    fig_price.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10), title="SPX Spot (intraday)")
                    fig_price.update_xaxes(title=None)
                    fig_price.update_yaxes(title=None)
                    st.plotly_chart(fig_price, width="stretch")
            else:
                st.caption("Aucune donnée temporelle disponible pour tracer le spot.")

    # Top metrics
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.metric("CALL Buys", f"{metrics.get('call_buys', 0):,}",
                  f"{metrics.get('agg_call_buys', 0):,} aggressive")
    with col2:
        st.metric("CALL Sells", f"{metrics.get('call_sells', 0):,}",
                  f"{metrics.get('agg_call_sells', 0):,} aggressive")
    with col3:
        st.metric("CALL Net", f"{metrics.get('call_net_flow', 0):+,}",
                  f"{metrics.get('agg_call_net', 0):+,} aggressive")
    with col4:
        st.metric("PUT Buys", f"{metrics.get('put_buys', 0):,}",
                  f"{metrics.get('agg_put_buys', 0):,} aggressive")
    with col5:
        st.metric("PUT Sells", f"{metrics.get('put_sells', 0):,}",
                  f"{metrics.get('agg_put_sells', 0):,} aggressive")
    with col6:
        st.metric("PUT Net", f"{metrics.get('put_net_flow', 0):+,}",
                  f"{metrics.get('agg_put_net', 0):+,} aggressive")

    # ============= Main Analysis Tabs =============
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Flow Heatmap",
        "Strike Analysis",
        "Time Series",
        "Buy vs Sell",
        "GEX Analysis",
        "MM Insights"
    ])

    with tab1:
        st.subheader("Advanced Flow Heatmap")
        heatmap_fig = create_comprehensive_flow_heatmap(df)
        st.plotly_chart(heatmap_fig, width="stretch")

    with tab2:
        st.subheader("Comprehensive Strike Analysis")
        if not strike_metrics.empty:
            strike_fig = create_strike_analysis_chart(strike_metrics, 20)
            st.plotly_chart(strike_fig, width="stretch")

            st.subheader("Detailed Strike Metrics")
            display_cols = [
                'strike', 'option_type', 'qty', 'aggressive_buy_volume',
                'aggressive_sell_volume', 'aggressive_net_volume', 'buy_dominance',
                'aggressive_ratio', 'flow_intensity', 'open_interest'
            ]
            display_cols = [c for c in display_cols if c in strike_metrics.columns]

            top_strikes_display = strike_metrics.nlargest(25, 'total_aggressive')[display_cols]

            styled_strikes = top_strikes_display.style.format({
                'strike': '{:.0f}',
                'qty': '{:,}',
                'aggressive_buy_volume': '{:,}',
                'aggressive_sell_volume': '{:,}',
                'aggressive_net_volume': '{:+,}',
                'buy_dominance': '{:.1%}',
                'aggressive_ratio': '{:.1%}',
                'flow_intensity': '{:.0f}'
            }).background_gradient(
                subset=['aggressive_net_volume'], cmap='RdBu', vmin=-2000, vmax=2000
            ).background_gradient(
                subset=['buy_dominance'], cmap='RdYlGn', vmin=0, vmax=1
            )
            st.dataframe(styled_strikes, width="stretch")
        else:
            st.info("No strike metrics available with current filters.")

    with tab3:
        st.subheader("Time Series Analysis")
        if 'minute' in df.columns and not df['minute'].isna().all():
            time_agg = (
                df.groupby('minute')
                .agg({'aggressive_net_volume': 'sum', 'directional_signed': 'sum', 'qty': 'sum'})
                .reset_index()
            )
            time_agg['cumulative_flow'] = time_agg['aggressive_net_volume'].cumsum()
            time_agg['cumulative_directional'] = time_agg['directional_signed'].cumsum()

            fig_time = make_subplots(
                rows=3, cols=1,
                subplot_titles=("Aggressive Net Flow per Minute", "Cumulative Flows", "Volume per Minute"),
                shared_xaxes=True, vertical_spacing=0.08
            )
            fig_time.add_trace(
                go.Scatter(x=time_agg['minute'], y=time_agg['aggressive_net_volume'],
                           mode='lines+markers', name='Aggressive Net Flow',
                           line=dict(color='blue', width=2)),
                row=1, col=1
            )
            fig_time.add_trace(
                go.Scatter(x=time_agg['minute'], y=time_agg['cumulative_flow'],
                           mode='lines', name='Cumulative Aggressive',
                           line=dict(color='red', width=3)),
                row=2, col=1
            )
            fig_time.add_trace(
                go.Scatter(x=time_agg['minute'], y=time_agg['cumulative_directional'],
                           mode='lines', name='Cumulative Directional',
                           line=dict(color='green', width=3)),
                row=2, col=1
            )
            fig_time.add_trace(
                go.Bar(x=time_agg['minute'], y=time_agg['qty'], name='Volume per Minute', opacity=0.7),
                row=3, col=1
            )
            fig_time.update_layout(height=900, showlegend=True)
            st.plotly_chart(fig_time, width="stretch")
        else:
            st.info("Time series data not available.")

    with tab4:
        st.subheader("Buy vs Sell Analysis")
        if not strike_metrics.empty:
            fig_scatter = go.Figure()
            for option_type in ['CALL', 'PUT']:
                data = strike_metrics[strike_metrics['option_type'] == option_type]
                if not data.empty:
                    fig_scatter.add_trace(
                        go.Scatter(
                            x=data['aggressive_buy_volume'],
                            y=data['aggressive_sell_volume'],
                            mode='markers',
                            name=f'{option_type}s',
                            text=data['strike'],
                            marker=dict(
                                size=data['qty'] / 50,
                                color=data['aggressive_net_volume'],
                                colorscale='RdBu',
                                showscale=True,
                                colorbar=dict(title="Net Flow"),
                                line=dict(width=1, color='black')
                            ),
                            hovertemplate=(
                                f"<b>{option_type} %{{text}}</b><br>"
                                "Aggressive Buys: %{x}<br>"
                                "Aggressive Sells: %{y}<br>"
                                "Net Flow: %{marker.color}<br>"
                                "<extra></extra>"
                            )
                        )
                    )
            max_val = max(
                strike_metrics['aggressive_buy_volume'].max(),
                strike_metrics['aggressive_sell_volume'].max()
            )
            fig_scatter.add_shape(
                type="line", x0=0, y0=0, x1=max_val, y1=max_val,
                line=dict(color="gray", width=2, dash="dash")
            )
            fig_scatter.update_layout(
                title="Aggressive Buy vs Sell Comparison (Size=Volume, Color=Net Flow)",
                xaxis_title="Aggressive Buy Volume",
                yaxis_title="Aggressive Sell Volume",
                height=600
            )
            st.plotly_chart(fig_scatter, width="stretch")
        else:
            st.info("No data available for buy vs sell analysis.")

    with tab5:
        st.subheader("Gamma Exposure Analysis")
        if "spot" in df.columns and (df["spot"] != 0).any():
            current_spot = float(df["spot"].replace(0, np.nan).dropna().median())
        else:
            current_spot = float(df["strike"].median()) if "strike" in df.columns else 6500.0

        try:
            gex_df, crit, det = render_enhanced_gex_component(df, current_spot)

            have_ib = bool(det.get("have_ib_gamma", False))
            col_src, col_ib, col_iv = st.columns(3)
            with col_src:
                st.metric(
                    "Source gamma",
                    "IBKR" if have_ib else "Approx (BS)",
                    help="IBKR = colonnes 'gamma' issues du collecteur. Approx (BS) = gamma recalculé via IV + DTE."
                )
            with col_ib:
                try:
                    con = sqlite3.connect(db_path)
                    norm_exp = "".join(ch for ch in str(selected_expiry) if ch and ch.isdigit()) if selected_expiry != "ALL" else None
                    if norm_exp:
                        q = """SELECT COUNT(*) AS rows,
                                      SUM(CASE WHEN gamma IS NOT NULL THEN 1 ELSE 0 END) AS with_gamma
                               FROM trades
                               WHERE REPLACE(REPLACE(REPLACE(COALESCE(expiry,''),'-',''),'/',''),'.','') = ?"""
                        rows = pd.read_sql(q, con, params=[norm_exp])
                    else:
                        rows = pd.read_sql("""SELECT COUNT(*) AS rows,
                                                     SUM(CASE WHEN gamma IS NOT NULL THEN 1 ELSE 0 END) AS with_gamma
                                              FROM trades""", con)
                    total_rows = int(rows.loc[0, "rows"] or 0)
                    with_g = int(rows.loc[0, "with_gamma"] or 0)
                    pct = (with_g / total_rows) if total_rows else 0.0
                    st.metric("Lignes avec gamma (IBKR)", f"{with_g:,} / {total_rows:,}", f"{pct:.1%}")
                finally:
                    try: con.close()
                    except: pass
            with col_iv:
                st.metric("IV dispo (pour Approx)", "Oui" if det.get("have_iv", False) else "Non")

            st.markdown("---")

            force_mode = st.toggle("Comparer IB-only vs Approx (BS)", value=False,
                                   help="IB-only = ne garde que les lignes avec gamma non-null; Approx = recalcul via IV+DTE.")
            if force_mode and HAS_ENHANCED_GEX:
                analyzer_ib = EnhancedGEXAnalyzer(prefer_ib_gamma=True)
                analyzer_bs = EnhancedGEXAnalyzer(prefer_ib_gamma=False)
                gex_ib, crit_ib, det_ib = analyzer_ib.analyze(df, current_spot)
                gex_bs, crit_bs, det_bs = analyzer_bs.analyze(df, current_spot)

                c1, c2 = st.columns(2)
                with c1:
                    st.subheader("IB-only")
                    st.metric("Total GEX (IB)", f"{crit_ib.get('total_gex', 0):,.0f}")
                    st.write("Zero-Gamma:", crit_ib.get("zero_gamma"))
                    st.dataframe(gex_ib.sort_values("abs_gex", ascending=False).head(40), width="stretch")
                with c2:
                    st.subheader("Approx (BS)")
                    st.metric("Total GEX (BS)", f"{crit_bs.get('total_gex', 0):,.0f}")
                    st.write("Zero-Gamma:", crit_bs.get("zero_gamma"))
                    st.dataframe(gex_bs.sort_values("abs_gex", ascending=False).head(40), width="stretch")

            if gex_df.empty:
                st.info("Aucune donnée GEX calculable avec les colonnes disponibles.")
            else:
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Total GEX", f"{crit.get('total_gex', 0):,.0f}")
                with c2:
                    z = crit.get("zero_gamma")
                    st.metric("Zero-Gamma", f"{z:.0f}" if z is not None else "—")
                with c3:
                    st.metric("Spot utilisé", f"{det.get('spot_used', current_spot):,.2f}")

                fig = go.Figure()
                fig.add_trace(go.Bar(x=gex_df["strike"], y=gex_df["gex"], name="GEX"))
                if crit.get("zero_gamma") is not None:
                    fig.add_vline(x=crit["zero_gamma"], line_color="red", line_dash="dash", annotation_text="ZG")
                fig.update_layout(title="GEX par strike", xaxis_title="Strike", yaxis_title="GEX")
                st.plotly_chart(fig, width="stretch")

                st.subheader("Top Gamma Walls")
                st.write("Positifs :", crit.get("largest_pos_gamma_strikes", []))
                st.write("Négatifs :", crit.get("largest_neg_gamma_strikes", []))

                st.dataframe(gex_df.sort_values("abs_gex", ascending=False).head(100), width="stretch")

        except Exception as e:
            st.error(f"GEX error: {e}")

    with tab6:
        st.subheader("Market Maker Insights")
        if not strike_metrics.empty:
            most_active = strike_metrics.nlargest(15, 'total_aggressive')
            st.subheader("Most Active Strikes (MM Focus Areas)")

            for _, row in most_active.iterrows():
                option_type = row['option_type']
                strike = row['strike']
                net_flow = row['aggressive_net_volume']
                buy_dominance = row['buy_dominance']

                client_delta = float(row.get('client_delta', 0.0))
                if abs(client_delta) >= 1:
                    if client_delta > 0:
                        mm_position = "Clients long Δ → MM short Δ → Hedge: BUY underlying"
                        color = "🟢"
                    else:
                        mm_position = "Clients short Δ → MM long Δ → Hedge: SELL underlying"
                        color = "🔴"
                else:
                    if option_type == 'CALL':
                        mm_position = "Short Calls → Hedge: BUY underlying" if net_flow > 0 else "Long Calls → Hedge: SELL underlying"
                        color = "🟢" if net_flow > 0 else "🔴"
                    else:
                        mm_position = "Short Puts → Hedge: SELL underlying" if net_flow > 0 else "Long Puts → Hedge: BUY underlying"
                        color = "🔴" if net_flow > 0 else "🟢"

                st.metric(
                    f"{color} {option_type} ${strike:.0f}",
                    f"{net_flow:+,} net flow",
                    f"{buy_dominance:.1%} buy dominance • {mm_position}"
                )

            st.subheader("Overall Market Implications")
            tot_client_delta = float(df.get("client_delta", 0.0).sum())
            tot_mm_gamma = float(df.get("mm_gamma", 0.0).sum())

            if abs(tot_client_delta) >= 1:
                hedge_side = "BUY underlying" if tot_client_delta > 0 else "SELL underlying"
                st.caption(f"Δ net clients = {tot_client_delta:+,.0f} → Hedge MM: {hedge_side}")

            if abs(tot_mm_gamma) >= 1:
                if tot_mm_gamma > 0:
                    st.success(f"Dealer **long gamma** (Σγ_MM={tot_mm_gamma:,.0f}) → hedging contrarien → **stabilisateur**")
                else:
                    st.warning(f"Dealer **short gamma** (Σγ_MM={tot_mm_gamma:,.0f}) → hedging pro-cyclique → **amplificateur de volatilité**")
            else:
                st.caption("Signal gamma faible (≈0) — effet stabilisant/amplificateur limité.")




# Auto-refresh logic
if enable_real_time:
    time.sleep(refresh_interval)
    st.cache_data.clear()
    st.rerun()

if __name__ == "__main__":
    main()
