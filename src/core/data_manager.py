"""
Data Manager - Central hub for all data sources
Aggregates volatility, flow, and GEX data for dashboard
"""
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Optional, Tuple

from src.core.har_calculator import HARCalculator
from src.core.options_flow_analyzer import (
    load_comprehensive_data,
    compute_comprehensive_metrics,
    build_enhanced_strike_metrics
)
from src.core.gex_calculator import EnhancedGEXAnalyzer
from src.core.regime_classifier import RegimeClassifier, RegimeType


class DataManager:
    """
    Central data aggregation for dashboard and analysis
    """

    def __init__(
            self,
            market_db_path: str = "db/market_data.db",
            flow_db_path: str = r"C:\Users\decle\PycharmProjects\flux_claude\db\optionflow.db"
    ):
        self.market_db_path = market_db_path
        self.flow_db_path = flow_db_path

        self.har_calc = HARCalculator(db_path=market_db_path)
        self.gex_analyzer = EnhancedGEXAnalyzer(multiplier=100.0, prefer_ib_gamma=True)
        self.regime_classifier = RegimeClassifier(db_path=market_db_path)

    def get_volatility_status(self, ticker: str = "SPX") -> Dict:
        """
        Get current volatility status for a ticker
        Returns dict with HAR-RV prediction, current levels, regime
        """
        try:
            df, results = self.har_calc.compute_har_prediction(
                ticker=ticker,
                period="1 an",
                mode="C2C"
            )

            if "error" in results:
                return {"error": results["error"], "status": "ERROR"}

            pred_vol = results.get("pred_vol", 0)

            # Determine status based on prediction
            if pred_vol < 1.0:
                status = "VERY LOW"
                color = "#4CAF50"
            elif pred_vol < 1.5:
                status = "LOW"
                color = "#4CAF50"
            elif pred_vol < 2.5:
                status = "MODERATE"
                color = "#FFA500"
            else:
                status = "HIGH"
                color = "#f44336"

            return {
                "har_rv_next": pred_vol,
                "status": status,
                "color": color,
                "r_squared": results.get("r_squared", 0),
                "ticker": ticker,
                "error": None
            }

        except Exception as e:
            return {
                "error": str(e),
                "status": "ERROR",
                "color": "#f44336"
            }

    def get_flow_status(self, expiry: str = "20251002") -> Dict:
        """
        Get current options flow status
        Returns dict with call/put net flows and sentiment
        """
        try:
            df = load_comprehensive_data(
                db_path=self.flow_db_path,
                selected_expiry=expiry,
                sample_size=100000,
                min_volume_filter=0,
                confidence_threshold=0.6
            )

            if df.empty:
                return {"error": "No flow data", "status": "NO DATA"}

            metrics = compute_comprehensive_metrics(df)

            call_net = metrics.get('agg_call_net', 0)
            put_net = metrics.get('agg_put_net', 0)

            # Determine sentiment
            if call_net > 100 and put_net < 50:
                status = "BULLISH"
                color = "#4CAF50"
            elif put_net > 100 and call_net < 50:
                status = "BEARISH"
                color = "#f44336"
            else:
                status = "NEUTRAL"
                color = "#FFA500"

            return {
                "call_net": call_net,
                "put_net": put_net,
                "status": status,
                "color": color,
                "call_buys": metrics.get('agg_call_buys', 0),
                "put_buys": metrics.get('agg_put_buys', 0),
                "error": None
            }

        except Exception as e:
            return {
                "error": str(e),
                "status": "ERROR",
                "color": "#f44336"
            }

    def get_gex_status(self, expiry: str = "20251002", use_real_oi: bool = True) -> Dict:
        """
        Get current GEX status with optional real OI
        """
        try:
            df = load_comprehensive_data(
                db_path=self.flow_db_path,
                selected_expiry=expiry,
                sample_size=100000,
                min_volume_filter=0,
                confidence_threshold=0.6
            )

            if df.empty:
                return {"error": "No GEX data", "status": "NO DATA"}

            spot = float(
                df["spot"].replace(0, float('nan')).dropna().median()) if "spot" in df.columns else 6700.0

            # ✅ NEW: Pass use_real_oi to analyzer
            gex_df, critical, detailed = self.gex_analyzer.analyze(
                df, spot,
                use_real_oi=use_real_oi,
                db_path=self.flow_db_path,
                expiry=expiry
            )

                    # ... reste du code inchangé

            zero_gamma = critical.get('zero_gamma')
            total_gex = critical.get('total_gex', 0)

            # Determine status
            if zero_gamma and spot > zero_gamma:
                status = "STABLE"
                color = "#4CAF50"
                spot_above_zg = True
            elif zero_gamma and spot < zero_gamma:
                status = "UNSTABLE"
                color = "#f44336"
                spot_above_zg = False
            else:
                status = "UNKNOWN"
                color = "#FFA500"
                spot_above_zg = False

            # Find max GEX strike
            max_gex_strike = None
            if not gex_df.empty:
                max_gex_strike = float(gex_df.loc[gex_df['abs_gex'].idxmax(), 'strike'])

            return {
                "zero_gamma": zero_gamma,
                "spot": spot,
                "spot_above_zg": spot_above_zg,
                "status": status,
                "color": color,
                "total_gex": total_gex,
                "max_gex_strike": max_gex_strike,
                "largest_pos_walls": critical.get('largest_pos_gamma_strikes', []),
                "largest_neg_walls": critical.get('largest_neg_gamma_strikes', []),
                "error": None
            }

        except Exception as e:
            return {
                "error": str(e),
                "status": "ERROR",
                "color": "#f44336"
            }

    def get_regime_classification(
            self,
            ticker: str = "SPX",
            vol_status: Optional[Dict] = None,
            gex_status: Optional[Dict] = None
    ) -> Dict:
        """
        Get complete regime classification
        """
        if vol_status is None:
            vol_status = self.get_volatility_status(ticker)

        if gex_status is None:
            gex_status = self.get_gex_status()

        har_rv = vol_status.get("har_rv_next", 1.5)
        spot_above_zg = "above" if gex_status.get("spot_above_zg", True) else "below"

        classification = self.regime_classifier.classify(
            ticker=ticker,
            har_rv=har_rv,
            spot_vs_zero_gamma=spot_above_zg,
            gex_status=gex_status
        )

        return {
            "regime": classification.regime.value,
            "confidence": classification.confidence,
            "description": classification.description,
            "recommended_strategies": classification.recommended_strategies,
            "strategies_to_avoid": classification.strategies_to_avoid,
            "win_rate": classification.historical_win_rate,
            "alerts": classification.surveillance_alerts,
            "signals": {
                "har_rv": classification.signals.har_rv if classification.signals else None,
                "rv_percentile": classification.signals.rv_percentile if classification.signals else None,
                "vix_level": classification.signals.vix_level if classification.signals else None,
                "vix_trend": classification.signals.vix_trend if classification.signals else None
            }
        }

    def generate_trade_recommendations(
            self,
            vol_status: Dict,
            gex_status: Dict,
            flow_status: Dict
    ) -> str:
        """
        DEPRECATED: Use regime-based recommendations instead
        Generate trading recommendations based on all three data sources
        """
        recommendations = []

        # Check for errors
        if vol_status.get("error") or gex_status.get("error") or flow_status.get("error"):
            return "<b>Insufficient data for recommendations</b><br>Check data sources in other tabs."

        har_rv = vol_status.get("har_rv_next", 999)
        spot_above_zg = gex_status.get("spot_above_zg", False)
        call_net = flow_status.get("call_net", 0)
        put_net = flow_status.get("put_net", 0)
        max_gex_strike = gex_status.get("max_gex_strike", 6700)

        # Bear Call Spread 0-3 DTE
        if har_rv < 1.5 and spot_above_zg and call_net < 0:
            short_strike = int(max_gex_strike + 50)
            long_strike = short_strike + 10

            recommendations.append(
                f"<b style='color: #4CAF50;'>Bear Call Spread 0-3 DTE</b><br>"
                f"Strikes suggeres: {short_strike}/{long_strike}<br>"
                f"Probabilite OTM: ~82%<br>"
                f"Credit attendu: $75-90<br>"
                f"Raison: Vol basse ({har_rv:.2f}%), spot > ZG, call selling<br>"
            )

        # Iron Condor 14 DTE
        if har_rv < 1.5 and gex_status.get("status") == "STABLE":
            call_short = int(max_gex_strike + 100)
            call_long = call_short + 50
            put_short = int(max_gex_strike - 100)
            put_long = put_short - 50

            recommendations.append(
                f"<b style='color: #4CAF50;'>Iron Condor 14 DTE</b><br>"
                f"Strikes: P:{put_long}/{put_short} C:{call_short}/{call_long}<br>"
                f"Probabilite profit: ~73%<br>"
                f"Credit attendu: $100-150<br>"
                f"Raison: Vol basse, GEX stable<br>"
            )

        # Bull Put Spread
        if put_net > 100 and har_rv < 2.0:
            short_strike = int(max_gex_strike - 50)
            long_strike = short_strike - 50

            recommendations.append(
                f"<b style='color: #2196F3;'>Bull Put Spread 7-14 DTE</b><br>"
                f"Strikes: {long_strike}/{short_strike}<br>"
                f"Credit attendu: $60-80<br>"
                f"Raison: Protection demand elevee (PUT buying)<br>"
            )

        # Warnings
        if not spot_above_zg:
            recommendations.append(
                f"<span style='color: #f44336;'><b>WARNING</b>: Spot sous Zero-Gamma<br>"
                f"Zone instable - eviter vente premium</span><br>"
            )

        if har_rv > 2.5:
            recommendations.append(
                f"<span style='color: #FFA500;'><b>CAUTION</b>: Vol elevee ({har_rv:.2f}%)<br>"
                f"Reduire taille positions ou passer en hedge</span><br>"
            )

        # Calendar Spread (low priority)
        if har_rv < 1.0:
            recommendations.append(
                f"<span style='color: #888;'>Calendar Spread: Vol trop stable actuellement</span><br>"
            )

        if not recommendations:
            return "<b>Aucune recommendation claire</b><br>Conditions de marche non optimales pour vente premium."

        return "<br>".join(recommendations)

    # ========================================================================
    # HISTORICAL DATA UPDATE METHODS (IB API)
    # ========================================================================

    def connect_ib(self, host: str = "127.0.0.1", port: int = 7497, client_id: int = 1) -> bool:
        """
        Connect to Interactive Brokers TWS/Gateway

        Args:
            host: IB host (default: localhost)
            port: IB port (7497 for TWS paper, 7496 for live, 4002 for Gateway)
            client_id: Unique client ID

        Returns:
            True if connected successfully
        """
        try:
            from ib_insync import IB

            self.ib = IB()
            self.ib.connect(host, port, clientId=client_id)

            if self.ib.isConnected():
                print(f"✓ Connected to IB @ {host}:{port} (clientId={client_id})")
                return True
            else:
                print(f"✗ Failed to connect to IB")
                return False

        except Exception as e:
            print(f"✗ Connection error: {e}")
            return False

    def disconnect_ib(self):
        """Disconnect from IB"""
        if hasattr(self, 'ib') and self.ib and self.ib.isConnected():
            self.ib.disconnect()
            print("✓ Disconnected from IB")

    def update_historical_data(
            self,
            symbol: str,
            duration: str = '1 M',
            bar_size: str = '1 day'
    ) -> bool:
        """
        Update historical data for a symbol via IB API

        Args:
            symbol: Ticker symbol (SPX, VIX, NDX, RUT, SPY, etc.)
            duration: How far back to fetch ('1 M', '6 M', '1 Y', '5 Y')
            bar_size: Bar size ('1 day', '1 hour', '1 min')

        Returns:
            True if successful
        """
        if not hasattr(self, 'ib') or not self.ib or not self.ib.isConnected():
            print(f"✗ Not connected to IB. Call connect_ib() first.")
            return False

        try:
            from ib_insync import Stock, Index, util

            # Create contract based on symbol type
            if symbol.upper() in ['SPX', 'VIX', 'RVX', 'NDX', 'RUT', 'DJX']:
                contract = Index(symbol.upper(), 'CBOE')
            elif symbol.upper() in ['ES', 'NQ', 'RTY', 'YM']:
                # Futures (need specific contract)
                print(f"⚠️  Futures not yet supported: {symbol}")
                return False
            else:
                # Stocks
                contract = Stock(symbol.upper(), 'SMART', 'USD')

            # Qualify contract
            print(f"⏳ Qualifying contract for {symbol}...")
            qualified = self.ib.qualifyContracts(contract)

            if not qualified:
                print(f"✗ Could not qualify contract for {symbol}")
                return False

            # Request historical data
            print(f"⏳ Fetching {duration} of data for {symbol}...")
            bars = self.ib.reqHistoricalData(
                qualified[0],
                endDateTime='',
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow='TRADES',
                useRTH=True,  # Regular trading hours only
                formatDate=1  # String format
            )

            if not bars:
                print(f"✗ No data received for {symbol}")
                return False

            # Convert to DataFrame
            df = util.df(bars)

            if df.empty:
                print(f"✗ Empty data for {symbol}")
                return False

            # Format date column
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')

            # Determine table name
            table_name = f"{symbol.lower()}_data"

            # Save to database
            conn = sqlite3.connect(self.market_db_path)

            # Create table if not exists
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    date TEXT PRIMARY KEY,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    average REAL,
                    barCount INTEGER
                )
            """)

            # Get existing data count
            cursor = conn.cursor()
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            existing_count = cursor.fetchone()[0]

            # Insert or replace data
            df.to_sql(table_name, conn, if_exists='replace', index=False)

            # Get new count
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            new_count = cursor.fetchone()[0]

            conn.commit()
            conn.close()

            print(f"✓ {symbol}: {new_count} bars saved (was {existing_count})")
            return True

        except Exception as e:
            print(f"✗ Error updating {symbol}: {e}")
            import traceback
            traceback.print_exc()
            return False