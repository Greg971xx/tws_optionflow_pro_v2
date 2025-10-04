from ui.components.volatility_base_viewer import VolatilityBaseViewer
from utils.db_helpers import get_available_tickers, read_volatility_stats
from core.strategies.regime_volatilite import classify_volatility_regime
import plotly.graph_objects as go
import pandas as pd

COLORS = {
    "🟢 calme": "#1f9d55",
    "🟡 latent": "#c2a200",
    "🔵 normal": "#2a6fdb",
    "🔴 stress": "#d64545",
    "🟠 retrait": "#e07a00",
    "❓ inconnu": "#888888",
}

class RegimeViewer(VolatilityBaseViewer):
    def __init__(self, parent=None):
        super().__init__(
            get_ticker_list_callable=get_available_tickers,
            parent=parent,
            title="Analyse des régimes de volatilité"
        )

    def load_dataframe(self, ticker: str) -> pd.DataFrame:
        df = read_volatility_stats(ticker)
        return df

    def generate_figure(self, df: pd.DataFrame, period_label: str) -> go.Figure:
        fig = go.Figure()
        print(f"[RegimeViewer] generate_figure: rows={0 if df is None else len(df)} period={period_label}")

        if df is None or df.empty:
            print("[RegimeViewer] DataFrame vide")
            fig.add_annotation(text="Aucune donnée", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig

        print(f"[RegimeViewer] Colonnes: {list(df.columns)}")

        # Assurer numeric
        for col in ["vr_c2c", "ma20_c2c"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Calcul ma20 si absente / vide
        if "ma20_c2c" not in df.columns or df["ma20_c2c"].isna().all():
            print("[RegimeViewer] ma20_c2c absente ou vide → recalcul via vr_c2c.rolling(20)")
            if "vr_c2c" in df.columns:
                df["ma20_c2c"] = pd.to_numeric(df["vr_c2c"], errors="coerce").rolling(20, min_periods=5).mean()
            else:
                print("[RegimeViewer] vr_c2c absente → rien à tracer")
                fig.add_annotation(text="Pas de séries exploitables", xref="paper", yref="paper",
                                   x=0.5, y=0.5, showarrow=False)
                return fig

        print(f"[RegimeViewer] ma20_c2c head:\n{df['ma20_c2c'].head()}")

        try:
            df2 = classify_volatility_regime(df, vol_column="ma20_c2c")
            counts = df2["regime_volatilite"].value_counts(dropna=False).to_dict()
            print(f"[RegimeViewer] Régimes détectés: {counts}")
        except Exception as e:
            print(f"[RegimeViewer] Erreur classify_volatility_regime: {e}")
            fig.add_annotation(text=f"Erreur régime: {e}", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
            return fig

        # Courbe MA20
        fig.add_trace(go.Scatter(x=df2["date"], y=df2["ma20_c2c"],
                                 name="MA20 C2C", mode="lines"))

        # Points par régime
        for regime, color in COLORS.items():
            sub = df2[df2["regime_volatilite"] == regime]
            print(f"[RegimeViewer] {regime}: {len(sub)} points")
            if not sub.empty:
                fig.add_trace(go.Scatter(
                    x=sub["date"], y=sub["ma20_c2c"], mode="markers",
                    name=regime, marker=dict(size=6, color=color)
                ))

        if len(fig.data) == 1:  # seulement la MA20
            fig.add_annotation(text="Aucun point catégorisé (trop de NaN ?)", x=0.5, y=0.9,
                               xref="paper", yref="paper", showarrow=False)

        return fig
