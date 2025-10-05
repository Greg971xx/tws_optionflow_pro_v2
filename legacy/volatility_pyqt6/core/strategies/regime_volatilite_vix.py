
import pandas as pd

def classify_volatility_regime_with_vix(df: pd.DataFrame,
                                        vol_column: str = "ma20_c2c",
                                        vix_rank_column: str = "rank_ma20_c2c_vix",
                                        seuil_bas=0.0075,
                                        seuil_haut=0.015,
                                        seuil_vix_rank=0.3) -> pd.DataFrame:
    """
    Classifie les régimes de volatilité basés sur ma20_c2c et ajoute un filtre VIX intégré.
    Retourne un DataFrame avec les colonnes :
    - "pente_vol"
    - "regime_volatilite"
    - "vix_filter_pass" (booléen)
    """

    df = df.copy()

    if vol_column not in df.columns:
        raise ValueError(f"Colonne '{vol_column}' manquante dans le DataFrame.")
    if vix_rank_column not in df.columns:
        raise ValueError(f"Colonne '{vix_rank_column}' manquante dans le DataFrame.")

    # Calcul de la pente
    df["pente_vol"] = df[vol_column].diff()

    # Attribution du régime
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

    # Ajout du filtre VIX : True si rank VIX < 0.3
    df["vix_filter_pass"] = df[vix_rank_column] < seuil_vix_rank

    return df
