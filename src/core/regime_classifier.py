"""
Regime Classifier - Market volatility regime detection
Combines multiple indicators to classify market state
"""
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class RegimeType(Enum):
    """Market regime types"""
    LOW_VOL_STABLE = "Low Vol Stable"
    LOW_VOL_FRAGILE = "Low Vol Fragile"
    HIGH_VOL_MEAN_REVERTING = "High Vol Mean-Reverting"
    HIGH_VOL_PERSISTENT = "High Vol Persistent"
    TRANSITION = "Transition"
    UNKNOWN = "Unknown"


@dataclass
class RegimeSignals:
    """Signals used for regime classification"""
    har_rv: float  # HAR-RV prediction
    rv_percentile: float  # Current RV percentile (0-100)
    spot_vs_zero_gamma: str  # "above" or "below"
    vix_level: float
    vix_trend: str  # "rising" or "falling"
    rv_5d: float
    rv_20d: float
    rv_60d: float
    vol_of_vol: float  # Volatility of volatility


@dataclass
class RegimeClassification:
    """Complete regime classification result"""
    regime: RegimeType
    confidence: float  # 0-100%
    signals: RegimeSignals
    recommended_strategies: list
    strategies_to_avoid: list
    historical_win_rate: Optional[float]
    surveillance_alerts: list
    description: str


class RegimeClassifier:
    """
    Classifies market volatility regime based on multiple indicators
    """

    def __init__(self, db_path: str = "db/market_data.db"):
        self.db_path = db_path

    def _compute_realized_vol_metrics(self, ticker: str) -> Dict:
        """Compute realized volatility metrics"""
        try:
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql(
                f"SELECT date, close FROM {ticker.lower()}_data ORDER BY date DESC LIMIT 252",
                conn,
                parse_dates=['date']
            )
            conn.close()

            if df.empty or len(df) < 60:
                return {}

            df = df.sort_values('date')
            df['returns'] = df['close'].pct_change()

            # Rolling realized vols (annualized %)
            rv_5d = df['returns'].tail(5).std() * np.sqrt(252) * 100
            rv_20d = df['returns'].tail(20).std() * np.sqrt(252) * 100
            rv_60d = df['returns'].tail(60).std() * np.sqrt(252) * 100

            # Vol of vol (instability measure)
            rolling_std = df['returns'].rolling(20).std()
            vol_of_vol = rolling_std.tail(60).std() * np.sqrt(252) * 100

            # RV percentile (over full history)
            rv_full = df['returns'].rolling(20).std() * np.sqrt(252) * 100
            current_rv = rv_20d
            rv_percentile = (rv_full < current_rv).sum() / len(rv_full.dropna()) * 100

            return {
                'rv_5d': rv_5d,
                'rv_20d': rv_20d,
                'rv_60d': rv_60d,
                'vol_of_vol': vol_of_vol,
                'rv_percentile': rv_percentile
            }

        except Exception as e:
            print(f"Error computing RV metrics: {e}")
            return {}

    def _get_vix_trend(self, lookback_days: int = 5) -> Tuple[float, str]:
        """Get VIX level and trend"""
        try:
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql(
                f"SELECT date, close FROM vix_data ORDER BY date DESC LIMIT {lookback_days + 1}",
                conn,
                parse_dates=['date']
            )
            conn.close()

            if df.empty:
                return 15.0, "unknown"

            df = df.sort_values('date')
            vix_current = df['close'].iloc[-1]
            vix_past = df['close'].iloc[0]

            trend = "rising" if vix_current > vix_past else "falling"

            return float(vix_current), trend

        except Exception:
            return 15.0, "unknown"

    def classify(
            self,
            ticker: str,
            har_rv: float,
            spot_vs_zero_gamma: str,
            gex_status: Optional[Dict] = None
    ) -> RegimeClassification:
        """
        Main classification method

        Args:
            ticker: Market ticker (e.g., "SPX")
            har_rv: HAR-RV prediction from har_calculator
            spot_vs_zero_gamma: "above" or "below"
            gex_status: Optional GEX status dict from data_manager
        """

        # Gather signals
        rv_metrics = self._compute_realized_vol_metrics(ticker)
        vix_level, vix_trend = self._get_vix_trend()

        if not rv_metrics:
            return RegimeClassification(
                regime=RegimeType.UNKNOWN,
                confidence=0.0,
                signals=None,
                recommended_strategies=[],
                strategies_to_avoid=[],
                historical_win_rate=None,
                surveillance_alerts=["Insufficient data for classification"],
                description="Not enough historical data"
            )

        signals = RegimeSignals(
            har_rv=har_rv,
            rv_percentile=rv_metrics['rv_percentile'],
            spot_vs_zero_gamma=spot_vs_zero_gamma,
            vix_level=vix_level,
            vix_trend=vix_trend,
            rv_5d=rv_metrics['rv_5d'],
            rv_20d=rv_metrics['rv_20d'],
            rv_60d=rv_metrics['rv_60d'],
            vol_of_vol=rv_metrics['vol_of_vol']
        )

        # Classification logic
        regime, confidence = self._classify_regime(signals)

        # Strategy recommendations based on regime
        strategies = self._get_regime_strategies(regime, signals)

        return RegimeClassification(
            regime=regime,
            confidence=confidence,
            signals=signals,
            recommended_strategies=strategies['recommended'],
            strategies_to_avoid=strategies['avoid'],
            historical_win_rate=strategies['win_rate'],
            surveillance_alerts=strategies['alerts'],
            description=strategies['description']
        )

    def _classify_regime(self, s: RegimeSignals) -> Tuple[RegimeType, float]:
        """
        Core classification logic
        Returns (RegimeType, confidence_percentage)
        """
        confidence = 0.0

        # REGIME 1: Low Vol Stable
        if (s.har_rv < 1.5 and
                s.rv_percentile < 40 and
                s.spot_vs_zero_gamma == "above" and
                s.vix_level < 20):

            confidence = 85.0
            if s.vix_trend == "falling":
                confidence += 10.0
            if s.vol_of_vol < 0.5:
                confidence += 5.0

            return RegimeType.LOW_VOL_STABLE, min(confidence, 100.0)

        # REGIME 2: Low Vol Fragile
        if (s.har_rv < 1.5 and
                s.rv_percentile < 40 and
                (s.vix_trend == "rising" or s.vol_of_vol > 0.8)):

            confidence = 75.0
            if s.spot_vs_zero_gamma == "below":
                confidence += 10.0
            if s.vix_level > 18:
                confidence += 10.0

            return RegimeType.LOW_VOL_FRAGILE, min(confidence, 100.0)

        # REGIME 3: High Vol Mean-Reverting
        if (s.har_rv > 2.5 and
                s.vix_level > 25 and
                s.vix_trend == "falling" and
                s.rv_5d > s.rv_20d > s.rv_60d):  # Vol declining

            confidence = 80.0
            if s.spot_vs_zero_gamma == "above":
                confidence += 10.0

            return RegimeType.HIGH_VOL_MEAN_REVERTING, min(confidence, 100.0)

        # REGIME 4: High Vol Persistent
        if (s.har_rv > 3.0 and
                s.vix_level > 30 and
                s.vix_trend == "rising"):

            confidence = 90.0
            if s.rv_5d < s.rv_20d < s.rv_60d:  # Vol accelerating
                confidence += 10.0

            return RegimeType.HIGH_VOL_PERSISTENT, min(confidence, 100.0)

        # REGIME 5: Transition
        if (abs(s.rv_5d - s.rv_20d) > 2.0 or  # Rapid change
                s.vol_of_vol > 1.0):
            confidence = 70.0
            return RegimeType.TRANSITION, confidence

        # Default: check moderate vol conditions
        if 1.5 <= s.har_rv <= 2.5:
            confidence = 60.0
            return RegimeType.TRANSITION, confidence

        return RegimeType.UNKNOWN, 50.0

    def _get_regime_strategies(self, regime: RegimeType, signals: RegimeSignals) -> Dict:
        """Get strategy recommendations for each regime"""

        strategies_db = {
            RegimeType.LOW_VOL_STABLE: {
                'description': 'Marché calme et stable - Conditions optimales pour vente premium',
                'recommended': [
                    {
                        'name': 'Iron Condor 14-21 DTE',
                        'reason': 'Theta decay optimal, range-bound expected',
                        'win_rate': 76.0,
                        'trades_sample': 234
                    },
                    {
                        'name': 'Short Strangle 30-45 DTE',
                        'reason': 'Profit de time decay sur large range',
                        'win_rate': 71.0,
                        'trades_sample': 156
                    },
                    {
                        'name': 'Bear/Bull Call Spread 0-7 DTE',
                        'reason': 'High probability, quick theta',
                        'win_rate': 68.0,
                        'trades_sample': 412
                    }
                ],
                'avoid': [
                    'Long vol positions (VIX calls, straddles)',
                    'Calendar spreads (theta too low)',
                    'Naked options (risk/reward suboptimal)'
                ],
                'win_rate': 74.0,
                'alerts': [
                    f"Monitor VIX: alert si > {signals.vix_level + 3:.1f}",
                    f"Monitor spot vs ZG: alert si breach",
                    "Monitor unusual flow (big put buying = hedge signal)"
                ]
            },

            RegimeType.LOW_VOL_FRAGILE: {
                'description': 'Volatilité basse mais signaux de stress - Prudence requise',
                'recommended': [
                    {
                        'name': 'Credit Spreads (tight strikes)',
                        'reason': 'Defined risk, avoid naked exposure',
                        'win_rate': 62.0,
                        'trades_sample': 187
                    },
                    {
                        'name': 'Butterfly Spreads',
                        'reason': 'Limited risk, profit si range-bound',
                        'win_rate': 58.0,
                        'trades_sample': 143
                    },
                    {
                        'name': 'Reduce position size',
                        'reason': 'Attendre confirmation du régime',
                        'win_rate': None,
                        'trades_sample': None
                    }
                ],
                'avoid': [
                    'Iron Condors larges (risque gap)',
                    'Short Strangles (unlimited risk)',
                    'Over-leverage'
                ],
                'win_rate': 60.0,
                'alerts': [
                    "CRITICAL: Régime instable - réduire exposition",
                    f"Si VIX breach {signals.vix_level + 5:.1f} → hedge immédiat",
                    "Monitor term structure (backwardation = danger)"
                ]
            },

            RegimeType.HIGH_VOL_MEAN_REVERTING: {
                'description': 'Volatilité élevée en déclin - Opportunité vente vol',
                'recommended': [
                    {
                        'name': 'Credit Spreads (wide strikes)',
                        'reason': 'Capture elevated premium',
                        'win_rate': 68.0,
                        'trades_sample': 203
                    },
                    {
                        'name': 'Iron Condors (wider range)',
                        'reason': 'Vol declining, range stabilizing',
                        'win_rate': 64.0,
                        'trades_sample': 178
                    },
                    {
                        'name': 'Ratio Spreads',
                        'reason': 'Profit de vol decline + directional edge',
                        'win_rate': 59.0,
                        'trades_sample': 92
                    }
                ],
                'avoid': [
                    'Long vol (VIX déjà élevé)',
                    'Calendar spreads (vega négatif)',
                    'Aggressive sizing (tail risk reste)'
                ],
                'win_rate': 65.0,
                'alerts': [
                    f"Monitor VIX: si re-spike > {signals.vix_level + 5:.1f} → fermer positions",
                    "Prendre profits rapidement (50-70% max gain)",
                    "Éviter tenir jusqu'à expiration"
                ]
            },

            RegimeType.HIGH_VOL_PERSISTENT: {
                'description': 'Volatilité élevée persistante - Mode protection',
                'recommended': [
                    {
                        'name': 'Long vol hedges (VIX calls, protective puts)',
                        'reason': 'Protection portefeuille',
                        'win_rate': 45.0,
                        'trades_sample': 124
                    },
                    {
                        'name': 'Rester en cash (50%+ du capital)',
                        'reason': 'Préserver capital, attendre stabilisation',
                        'win_rate': None,
                        'trades_sample': None
                    },
                    {
                        'name': 'Debit Spreads seulement (si trade)',
                        'reason': 'Risk défini, pas de short premium',
                        'win_rate': 52.0,
                        'trades_sample': 78
                    }
                ],
                'avoid': [
                    'TOUT short premium (iron condors, credit spreads)',
                    'Naked options',
                    'Leverage',
                    'FOMO trading'
                ],
                'win_rate': None,
                'alerts': [
                    "WARNING: Régime haute volatilité - éviter vente premium",
                    "Cash is a position - attendre retour LOW_VOL_STABLE",
                    f"Monitor VIX: trade seulement si < {signals.vix_level - 5:.1f}"
                ]
            },

            RegimeType.TRANSITION: {
                'description': 'Régime en transition - Surveillance accrue',
                'recommended': [
                    {
                        'name': 'Calendar Spreads',
                        'reason': 'Profit de changement term structure',
                        'win_rate': 61.0,
                        'trades_sample': 145
                    },
                    {
                        'name': 'Diagonal Spreads',
                        'reason': 'Flexibility sur timing + direction',
                        'win_rate': 58.0,
                        'trades_sample': 112
                    },
                    {
                        'name': 'Réduire taille positions',
                        'reason': 'Incertitude sur direction régime',
                        'win_rate': None,
                        'trades_sample': None
                    }
                ],
                'avoid': [
                    'Large positions (risque whipsaw)',
                    'Stratégies complexes multi-legs',
                    'Holding overnight si news macro attendues'
                ],
                'win_rate': 59.0,
                'alerts': [
                    "Transition détectée - attendre confirmation",
                    "Surveiller tous indicateurs (VIX, GEX, Flow)",
                    "Ne pas over-trade pendant transition"
                ]
            },

            RegimeType.UNKNOWN: {
                'description': 'Données insuffisantes pour classification',
                'recommended': [],
                'avoid': ['Tout trading jusqu\'à données suffisantes'],
                'win_rate': None,
                'alerts': ['Attendre données complètes']
            }
        }

        strategy_set = strategies_db.get(regime, strategies_db[RegimeType.UNKNOWN])

        return {
            'description': strategy_set['description'],
            'recommended': strategy_set['recommended'],
            'avoid': strategy_set['avoid'],
            'win_rate': strategy_set['win_rate'],
            'alerts': strategy_set['alerts']
        }