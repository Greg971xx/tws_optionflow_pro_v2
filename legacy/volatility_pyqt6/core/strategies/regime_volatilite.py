import pandas as pd

def classify_volatility_regime(df: pd.DataFrame, vol_column: str = "ma20_c2c") -> pd.DataFrame:
    if vol_column not in df.columns:
        raise ValueError(f"La colonne '{vol_column}' est absente du DataFrame.")
    out = df.copy()

    # 👉 Assurer numérique
    out[vol_column] = pd.to_numeric(out[vol_column], errors="coerce")

    seuil_bas = 0.0075
    seuil_haut = 0.015

    # 👉 diff() ne doit pas voir des objets
    out["pente_vol"] = out[vol_column].astype(float).diff()

    def assign_regime(row):
        niveau = row[vol_column]
        pente = row["pente_vol"]
        if pd.isna(niveau) or pd.isna(pente):
            return "❓ inconnu"
        if niveau < seuil_bas and pente < 0:
            return "🟢 calme"
        elif niveau < seuil_bas and pente >= 0:
            return "🟡 latent"
        elif seuil_bas <= niveau <= seuil_haut:
            return "🔵 normal"
        elif niveau > seuil_haut and pente > 0:
            return "🔴 stress"
        elif niveau > seuil_haut and pente <= 0:
            return "🟠 retrait"
        else:
            return "❓ inconnu"

    out["regime_volatilite"] = out.apply(assign_regime, axis=1)
    return out
