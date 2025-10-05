"""
enhanced_gex_analysis.py
------------------------
Lightweight GEX calculator used by the Streamlit app.

- Prefers **observed gamma** from the database if available (column "gamma").
- Falls back to a simple Black–Scholes gamma approximation using "iv" and "expiry".
- Assumes SPX multiplier = 100 (configurable).
- Returns (gex_by_strike_df, critical_levels_dict, detailed_dict).

Expected columns in df (best effort / optional):
    - 'strike' (float)
    - 'right' or 'option_type' (values: 'C'/'P' or 'CALL'/'PUT')
    - 'signed_qty' (dealer sign = -signed_qty). If absent, uses 'qty' + 'estimation' ('achat'/'vente')
    - 'gamma' (per-option gamma). If absent, will try to use 'iv' + 'expiry' to approximate.
    - 'iv' (implied vol as DECIMAL, e.g., 0.20)
    - 'expiry' (YYYYMMDD string)
    - 'ts' (timestamp text)  # optional, for time-to-expiry

Usage in the app:
    from enhanced_gex_analysis import render_enhanced_gex_component, EnhancedGEXAnalyzer

    gex_by_strike, critical_levels, detailed = render_enhanced_gex_component(df, spot)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd

# Optional: allow Streamlit warnings if available, but do not require it.
try:
    import streamlit as st
except Exception:
    st = None


# -------------------- Math helpers --------------------
SQRT_2PI = math.sqrt(2.0 * math.pi)

def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / SQRT_2PI

def _to_float(s):
    try:
        if s is None or (isinstance(s, float) and (np.isnan(s))):
            return None
        v = float(s)
        return v if np.isfinite(v) else None
    except Exception:
        return None


# -------------------- Analyzer --------------------
@dataclass
class EnhancedGEXAnalyzer:
    multiplier: float = 100.0           # SPX options
    prefer_ib_gamma: bool = True        # use gamma column if present
    min_T_days: float = 1.0             # minimal DTE to avoid div by zero
    min_sigma: float = 1e-6             # minimal vol to avoid div by zero
    clamp_sigma: float = 5.0            # cap extreme vols (e.g. 500%)

    def _ensure_option_type(self, df: pd.DataFrame) -> pd.Series:
        if "option_type" in df.columns:
            s = df["option_type"].astype(str).str.upper()
            s = s.replace({"C": "CALL", "P": "PUT"})
            return s
        if "right" in df.columns:
            s = df["right"].astype(str).str.upper()
            s = s.replace({"C": "CALL", "P": "PUT"})
            return s
        # default unknown -> treat as CALL to keep rows; will have no effect if gamma NaN
        return pd.Series(["CALL"] * len(df), index=df.index)

    def _d1(self, S: float, K: float, sigma: float, T: float, r: float = 0.0) -> Optional[float]:
        try:
            if S <= 0 or K <= 0 or sigma <= 0 or T <= 0:
                return None
            return (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
        except Exception:
            return None

    def _approx_gamma(self, S: float, K: float, sigma: float, T: float) -> Optional[float]:
        """
        Black–Scholes gamma for either call or put (same formula).
        Returns per-underlying-unit gamma (not multiplied by contract size).
        """
        try:
            sigma = max(min(float(sigma), self.clamp_sigma), self.min_sigma)
            T = max(float(T), self.min_T_days / 365.0)  # minimal T
            d1 = self._d1(S, K, sigma, T)
            if d1 is None:
                return None
            return _norm_pdf(d1) / (S * sigma * math.sqrt(T))
        except Exception:
            return None

    def _parse_expiry(self, s: str) -> Optional[datetime]:
        if not isinstance(s, str):
            return None
        for fmt in ("%Y%m%d", "%Y-%m-%d"):
            try:
                return datetime.strptime(s, fmt)
            except Exception:
                continue
        return None

    def _compute_T_years(self, expiry_str: Optional[str], ref_time: Optional[datetime]) -> float:
        """
        Compute year fraction to expiry using df['ts'] if available else current time.
        """
        if not expiry_str:
            return 0.0
        exp_dt = self._parse_expiry(expiry_str)
        if exp_dt is None:
            return 0.0
        now = ref_time or datetime.utcnow()
        dte_days = (exp_dt - now).days + max((exp_dt - now).seconds, 0) / 86400.0
        return max(dte_days, 0.0) / 365.0

    def _dealer_position(self, df: pd.DataFrame) -> pd.Series:
        # Prefer signed_qty if present
        if "signed_qty" in df.columns:
            return -pd.to_numeric(df["signed_qty"], errors="coerce").fillna(0.0)
        # Fallback: use qty + estimation (achat/vente)
        qty = pd.to_numeric(df.get("qty", 0.0), errors="coerce").fillna(0.0)
        est = df.get("estimation", "").astype(str).str.lower()
        # client buy -> dealers sell -> negative; client sell -> dealers buy -> positive
        sign = est.map({"achat": -1.0, "buy": -1.0, "vente": 1.0, "sell": 1.0}).fillna(0.0)
        return qty * sign

    def _pick_gamma(self, df: pd.DataFrame, S: float) -> pd.Series:
        """
        Returns a Series of per-contract gamma (per-underlying-unit) if available,
        else approximates with BS using iv + expiry.
        """
        g = None
        if self.prefer_ib_gamma and "gamma" in df.columns:
            g = pd.to_numeric(df["gamma"], errors="coerce")

        if g is None or not np.isfinite(g.fillna(np.nan)).any():
            # approximate gamma using iv + expiry
            iv = pd.to_numeric(df.get("iv", np.nan), errors="coerce")
            K = pd.to_numeric(df.get("strike", np.nan), errors="coerce")
            # choose a reference time per row
            if "ts" in df.columns:
                ts = pd.to_datetime(df["ts"], errors="coerce")
                ref_times = ts.where(ts.notna(), datetime.utcnow())
            else:
                ref_times = pd.Series([datetime.utcnow()] * len(df), index=df.index)

            T = []
            for i, exp_str in enumerate(df.get("expiry", "")):
                T.append(self._compute_T_years(str(exp_str), ref_times.iloc[i] if len(ref_times) > i else None))
            T = pd.Series(T, index=df.index)

            gamma_est = []
            for i in df.index:
                s_iv = _to_float(iv.loc[i]) if i in iv.index else None
                s_K = _to_float(K.loc[i]) if i in K.index else None
                s_T = _to_float(T.loc[i]) if i in T.index else None
                if s_iv and s_K and s_T and S:
                    gamma_est.append(self._approx_gamma(S, s_K, s_iv, s_T))
                else:
                    gamma_est.append(np.nan)
            g = pd.Series(gamma_est, index=df.index)

        return g

    def analyze(self, df: pd.DataFrame, spot: float) -> Tuple[pd.DataFrame, Dict, Dict]:
        """
        Main entry: compute per-strike GEX and key levels.
        """
        if df is None or df.empty:
            return pd.DataFrame(columns=["strike", "gex"]), {"zero_gamma": None}, {"total_gex": 0.0}

        work = df.copy()
        # normalize
        work["strike"] = pd.to_numeric(work.get("strike"), errors="coerce")
        work = work.dropna(subset=["strike"])

        work["option_type"] = self._ensure_option_type(work)
        S = float(spot) if _to_float(spot) else float(np.nanmean(work.get("spot", np.nan)))
        if not S or not np.isfinite(S) or S <= 0:
            # last resort: median strike as proxy
            S = float(np.nanmedian(work["strike"]))

        mm_position = self._dealer_position(work)  # dealers share (contracts), signed
        gamma_series = self._pick_gamma(work, S)

        # Contribution per row: gamma * S^2 * position * contract_multiplier
        gex_contrib = gamma_series * (S ** 2) * mm_position * float(self.multiplier)

        # Group by strike and type (and total)
        tmp = pd.DataFrame({
            "strike": work["strike"],
            "option_type": work["option_type"],
            "gex": gex_contrib
        })

        # Clean NaNs
        tmp = tmp.replace([np.inf, -np.inf], np.nan).dropna(subset=["gex", "strike"])

        by_strike_type = tmp.groupby(["strike", "option_type"], as_index=False)["gex"].sum()
        by_strike = by_strike_type.groupby("strike", as_index=False)["gex"].sum()
        by_strike["abs_gex"] = by_strike["gex"].abs()

        # Critical levels
        #  - Gamma walls: top +/- by absolute size
        top_pos = by_strike.sort_values("gex", ascending=False).head(3)["strike"].tolist()
        top_neg = by_strike.sort_values("gex", ascending=True).head(3)["strike"].tolist()

        #  - Zero gamma level: linear interp on cumulative GEX across strikes
        #    Sort by strike, compute cumulative sum and find sign change.
        cum = by_strike.sort_values("strike").copy()
        cum["cum_gex"] = cum["gex"].cumsum()
        zero_gamma = None
        for i in range(1, len(cum)):
            a, b = cum.iloc[i - 1], cum.iloc[i]
            if a["cum_gex"] == 0:
                zero_gamma = float(a["strike"])
                break
            if (a["cum_gex"] < 0 and b["cum_gex"] > 0) or (a["cum_gex"] > 0 and b["cum_gex"] < 0):
                # linear interpolation between a and b on cum_gex axis
                g1, g2 = float(a["cum_gex"]), float(b["cum_gex"])
                k1, k2 = float(a["strike"]), float(b["strike"])
                if g2 != g1:
                    w = abs(g1) / (abs(g1) + abs(g2))
                    zero_gamma = k1 * w + k2 * (1.0 - w)
                else:
                    zero_gamma = (k1 + k2) / 2.0
                break

        # Detailed summary
        total_gex = float(by_strike["gex"].sum()) if len(by_strike) else 0.0
        detailed = {
            "spot_used": float(S),
            "rows_used": int(len(tmp)),
            "have_ib_gamma": bool(("gamma" in df.columns) and pd.to_numeric(df["gamma"], errors="coerce").notna().any()),
            "have_iv": bool(("iv" in df.columns) and pd.to_numeric(df["iv"], errors="coerce").notna().any()),
            "total_gex": total_gex,
            "top_pos_walls": top_pos,
            "top_neg_walls": top_neg
        }

        crit = {
            "zero_gamma": zero_gamma,
            "largest_pos_gamma_strikes": top_pos,
            "largest_neg_gamma_strikes": top_neg,
            "total_gex": total_gex
        }

        # Return both by_strike and by_strike+type for richer usage if needed
        by_strike = by_strike.sort_values("strike").reset_index(drop=True)
        by_strike_type = by_strike_type.sort_values(["strike", "option_type"]).reset_index(drop=True)

        # attach convenience pivot for UI tools (optional usage)
        pivot = by_strike_type.pivot_table(index="strike", columns="option_type", values="gex", fill_value=0.0)
        pivot = pivot.reset_index().rename_axis(None, axis=1)

        # Pack primary df with both views
        out = by_strike.merge(pivot, on="strike", how="left", suffixes=("", ""))
        # Ensure columns exist even if one side missing
        if "CALL" not in out.columns:
            out["CALL"] = 0.0
        if "PUT" not in out.columns:
            out["PUT"] = 0.0

        return out[["strike", "gex", "CALL", "PUT", "abs_gex"]], crit, detailed


# -------------------- Facade for the app --------------------
def render_enhanced_gex_component(df: pd.DataFrame, current_spot: float):
    """
    Thin wrapper to be compatible with the app's expected import.
    """
    analyzer = EnhancedGEXAnalyzer(multiplier=100.0, prefer_ib_gamma=True)
    gex_by_strike, critical_levels, detailed = analyzer.analyze(df, current_spot)

    # Optional: small heads-up inside Streamlit if we had to approximate
    if st is not None:
        if not detailed["have_ib_gamma"] and detailed["have_iv"]:
            st.info("GEX calculé avec gamma **approximé** (BS) à partir d'IV et DTE.")
        elif not detailed["have_ib_gamma"] and not detailed["have_iv"]:
            st.warning("GEX calculé sans gamma observé ni IV : les niveaux peuvent être incomplets.")

    return gex_by_strike, critical_levels, detailed
