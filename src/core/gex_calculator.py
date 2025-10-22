"""
GEX Calculator
Extracted from enhanced_gex_analysis.py - Pure logic
"""
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Tuple, Optional
from src.core.oi_fetcher import latest_oi_frame


import numpy as np
import pandas as pd

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


@dataclass
class EnhancedGEXAnalyzer:
    multiplier: float = 100.0
    prefer_ib_gamma: bool = True
    min_T_days: float = 1.0
    min_sigma: float = 1e-6
    clamp_sigma: float = 5.0

    def _ensure_option_type(self, df: pd.DataFrame) -> pd.Series:
        if "option_type" in df.columns:
            s = df["option_type"].astype(str).str.upper()
            s = s.replace({"C": "CALL", "P": "PUT"})
            return s
        if "right" in df.columns:
            s = df["right"].astype(str).str.upper()
            s = s.replace({"C": "CALL", "P": "PUT"})
            return s
        return pd.Series(["CALL"] * len(df), index=df.index)

    def _d1(self, S: float, K: float, sigma: float, T: float, r: float = 0.0) -> Optional[float]:
        try:
            if S <= 0 or K <= 0 or sigma <= 0 or T <= 0:
                return None
            return (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
        except Exception:
            return None

    def _approx_gamma(self, S: float, K: float, sigma: float, T: float) -> Optional[float]:
        try:
            sigma = max(min(float(sigma), self.clamp_sigma), self.min_sigma)
            T = max(float(T), self.min_T_days / 365.0)
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
        if not expiry_str:
            return 0.0
        exp_dt = self._parse_expiry(expiry_str)
        if exp_dt is None:
            return 0.0
        now = ref_time or datetime.utcnow()
        dte_days = (exp_dt - now).days + max((exp_dt - now).seconds, 0) / 86400.0
        return max(dte_days, 0.0) / 365.0

    def _dealer_position(self, df: pd.DataFrame) -> pd.Series:
        if "signed_qty" in df.columns:
            return -pd.to_numeric(df["signed_qty"], errors="coerce").fillna(0.0)
        qty = pd.to_numeric(df.get("qty", 0.0), errors="coerce").fillna(0.0)
        est = df.get("estimation", "").astype(str).str.lower()
        sign = est.map({"achat": -1.0, "buy": -1.0, "vente": 1.0, "sell": 1.0}).fillna(0.0)
        return qty * sign

    def _pick_gamma(self, df: pd.DataFrame, S: float) -> pd.Series:
        g = None
        if self.prefer_ib_gamma and "gamma" in df.columns:
            g = pd.to_numeric(df["gamma"], errors="coerce")

        if g is None or not np.isfinite(g.fillna(np.nan)).any():
            iv = pd.to_numeric(df.get("iv", np.nan), errors="coerce")
            K = pd.to_numeric(df.get("strike", np.nan), errors="coerce")
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

    def analyze(self, df: pd.DataFrame, spot: float, use_real_oi: bool = True,
                db_path: Optional[str] = None, expiry: Optional[str] = None) -> Tuple[pd.DataFrame, Dict, Dict]:
        """
        Analyze GEX with optional real OI data

        Args:
            df: Options flow DataFrame
            spot: Current spot price
            use_real_oi: If True and OI data available, use real OI instead of qty
            db_path: Path to DB containing oi_snapshots (required if use_real_oi=True)
            expiry: Expiry to fetch OI for (required if use_real_oi=True)
        """
        if df is None or df.empty:
            return pd.DataFrame(columns=["strike", "gex"]), {"zero_gamma": None}, {"total_gex": 0.0}

        work = df.copy()
        work["strike"] = pd.to_numeric(work.get("strike"), errors="coerce")
        work = work.dropna(subset=["strike"])

        work["option_type"] = self._ensure_option_type(work)
        S = float(spot) if _to_float(spot) else float(np.nanmean(work.get("spot", np.nan)))
        if not S or not np.isfinite(S) or S <= 0:
            S = float(np.nanmedian(work["strike"]))

        # ✅ NEW: Use real OI if available
        # Dans analyze(), remplace le bloc merge OI par :
        if use_real_oi and db_path and expiry:
            try:
                oi_df = latest_oi_frame(db_path, expiry)
                if not oi_df.empty:
                    # Merge OI into work
                    work = work.merge(
                        oi_df,
                        on=['option_type', 'strike'],
                        how='left'
                    )
                    # Use OI as position size instead of qty
                    if 'open_interest' in work.columns:
                        work['position_for_gex'] = work['open_interest'].fillna(work['qty'])
                        print(f"[GEX] Using real OI: {len(oi_df)} strikes with OI data")
                    else:
                        work['position_for_gex'] = work['qty']
                else:
                    work['position_for_gex'] = work['qty']
            except Exception as e:
                print(f"[GEX] Could not load OI, using qty: {e}")
                work['position_for_gex'] = work.get('qty', 0)
        else:
            work['position_for_gex'] = work.get('qty', 0)

        if 'position_for_gex' in work.columns:
            mm_position = self._dealer_position_with_oi(work)
        else:
            mm_position = self._dealer_position(work)
        gamma_series = self._pick_gamma(work, S)

        gex_contrib = gamma_series * (S ** 2) * mm_position * float(self.multiplier)

        tmp = pd.DataFrame({
            "strike": work["strike"],
            "option_type": work["option_type"],
            "gex": gex_contrib
        })

        tmp = tmp.replace([np.inf, -np.inf], np.nan).dropna(subset=["gex", "strike"])

        by_strike_type = tmp.groupby(["strike", "option_type"], as_index=False)["gex"].sum()
        by_strike = by_strike_type.groupby("strike", as_index=False)["gex"].sum()
        by_strike["abs_gex"] = by_strike["gex"].abs()

        top_pos = by_strike.sort_values("gex", ascending=False).head(3)["strike"].tolist()
        top_neg = by_strike.sort_values("gex", ascending=True).head(3)["strike"].tolist()

        cum = by_strike.sort_values("strike").copy()
        cum["cum_gex"] = cum["gex"].cumsum()
        zero_gamma = None
        for i in range(1, len(cum)):
            a, b = cum.iloc[i - 1], cum.iloc[i]
            if a["cum_gex"] == 0:
                zero_gamma = float(a["strike"])
                break
            if (a["cum_gex"] < 0 and b["cum_gex"] > 0) or (a["cum_gex"] > 0 and b["cum_gex"] < 0):
                g1, g2 = float(a["cum_gex"]), float(b["cum_gex"])
                k1, k2 = float(a["strike"]), float(b["strike"])
                if g2 != g1:
                    w = abs(g1) / (abs(g1) + abs(g2))
                    zero_gamma = k1 * w + k2 * (1.0 - w)
                else:
                    zero_gamma = (k1 + k2) / 2.0
                break

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

        pivot = by_strike_type.pivot_table(index="strike", columns="option_type", values="gex", fill_value=0.0)
        pivot = pivot.reset_index().rename_axis(None, axis=1)

        out = by_strike.merge(pivot, on="strike", how="left", suffixes=("", ""))
        if "CALL" not in out.columns:
            out["CALL"] = 0.0
        if "PUT" not in out.columns:
            out["PUT"] = 0.0

        return out[["strike", "gex", "CALL", "PUT", "abs_gex"]], crit, detailed

    def _dealer_position_with_oi(self, df: pd.DataFrame) -> pd.Series:
        """
        Dealer position using real OI or signed_qty
        """
        if "position_for_gex" in df.columns:
            # OI represents total open interest (unsigned)
            # Dealer is short when clients buy, long when clients sell
            qty = pd.to_numeric(df["position_for_gex"], errors="coerce").fillna(0.0)

            if "estimation" in df.columns:
                est = df.get("estimation", "").astype(str).str.lower()
                sign = est.map({"achat": -1.0, "buy": -1.0, "vente": 1.0, "sell": 1.0}).fillna(0.0)
                return qty * sign
            else:
                # If no estimation, assume dealer is net short (conservative)
                return -qty

        # Fallback to original method
        if "signed_qty" in df.columns:
            return -pd.to_numeric(df["signed_qty"], errors="coerce").fillna(0.0)

        qty = pd.to_numeric(df.get("qty", 0.0), errors="coerce").fillna(0.0)
        est = df.get("estimation", "").astype(str).str.lower()
        sign = est.map({"achat": -1.0, "buy": -1.0, "vente": 1.0, "sell": 1.0}).fillna(0.0)
        return qty * sign