
import pandas as pd

def classify_volatility_regime(df: pd.DataFrame, vol_column: str = "ma20_c2c") -> pd.DataFrame:
    """
    Ajoute une colonne 'regime_volatilite' au DataFrame selon le niveau et la pente de la volatilité.
    - vol_column : colonne à utiliser pour classifier (ex: ma20_c2c)
    """

    # Vérifie la présence de la colonne
    if vol_column not in df.columns:
        raise ValueError(f"La colonne '{vol_column}' est absente du DataFrame.")

    df = df.copy()

    # Seuils de niveau (adaptables)
    seuil_bas = 0.0075  # Volatilité faible
    seuil_haut = 0.015  # Volatilité élevée

    # Calcul de la pente (variation jour / jour)
    df["pente_vol"] = df[vol_column].diff()

    # Classification
    def assign_regime(row):
        niveau = row[vol_column]
        pente = row["pente_vol"]

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

    df["regime_volatilite"] = df.apply(assign_regime, axis=1)
    return df
