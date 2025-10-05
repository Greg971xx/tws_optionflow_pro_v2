# cleanup_by_expiry.py
# ------------------------------------------------------
# Supprime UNIQUEMENT les lignes correspondant à une expiration donnée,
# dans toutes les tables qui possèdent une colonne d'expiration.
#
# - Détection auto des colonnes expiration courantes:
#   ["expiry", "expiration", "exp_date", "maturity", "maturity_date"]
# - Option --tables pour restreindre aux tables listées (séparées par des virgules)
# - Dry-run par défaut ; utiliser --force pour exécuter
# - Sauvegarde .bak_<expiry> avant suppression (désactivable via --no-backup)
#
# Exemples :
#   Dry-run :
#     python cleanup_by_expiry.py --db db/optionflow.db --expiry 2025-09-16
#   Exécuter :
#     python cleanup_by_expiry.py --db optionflow.db --expiry 20250919 --force ( faire cd db avant)
#   Limiter aux tables 'trades' :
#     python cleanup_by_expiry.py --db db/optionflow.db --expiry 20250916 --tables trades --force
#
import argparse
import os
import shutil
import sqlite3
from typing import List, Tuple

DEFAULT_DB = "optionflow.db"

# Noms de colonnes candidates à l'expiration (ordre de préférence)
CANDIDATE_EXPIRY_COLS = ["expiry", "expiration", "exp_date", "maturity", "maturity_date"]

def normalize_expiry(s: str) -> str:
    """Retourne l'expiration au format YYYYMMDD (chiffres uniquement)."""
    digits = "".join(ch for ch in s if ch.isdigit())
    if len(digits) != 8:
        raise ValueError("Expiration invalide. Fournis une date au format YYYYMMDD (ou YYYY-MM-DD).")
    return digits

def ensure_backup(db_path: str, tag: str):
    bak = f"{db_path}.bak_{tag}"
    if not os.path.exists(bak):
        shutil.copy2(db_path, bak)
        print(f"[OK] Sauvegarde créée: {bak}")
    else:
        print(f"[INFO] Sauvegarde déjà existante: {bak}")

def get_all_tables(con) -> List[str]:
    rows = con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
    ).fetchall()
    return [r[0] for r in rows]

def table_has_col(con, table: str, col: str) -> bool:
    cur = con.execute(f"PRAGMA table_info({table});")
    return any(r[1] == col for r in cur.fetchall())

def discover_expiry_targets(con, allowed_tables: List[str] = None) -> List[Tuple[str, str]]:
    """
    Retourne [(table, expiry_col)] pour chaque table qui contient une colonne d'expiration.
    Si allowed_tables est fourni, restreint la détection à ces tables.
    """
    targets = []
    tables = allowed_tables if allowed_tables else get_all_tables(con)
    for t in tables:
        # Cherche la première colonne candidate existante dans l'ordre
        for c in CANDIDATE_EXPIRY_COLS:
            if table_has_col(con, t, c):
                targets.append((t, c))
                break
    return targets

def count_expiry(con, table: str, expiry_col: str, expiry_norm: str) -> int:
    q = f"""
    SELECT COUNT(*) FROM "{table}"
    WHERE REPLACE(REPLACE(REPLACE("{expiry_col}", '-', ''), '/', ''), '.', '') = ?
    """
    return con.execute(q, (expiry_norm,)).fetchone()[0]

def delete_expiry(con, table: str, expiry_col: str, expiry_norm: str) -> int:
    q = f"""
    DELETE FROM "{table}"
    WHERE REPLACE(REPLACE(REPLACE("{expiry_col}", '-', ''), '/', ''), '.', '') = ?
    """
    return con.execute(q, (expiry_norm,)).rowcount

def main():
    parser = argparse.ArgumentParser(
        description="Supprimer uniquement les données d'une EXPIRATION donnée dans la base SQLite."
    )
    parser.add_argument("--db", default=DEFAULT_DB, help="Chemin vers la base (ex: db/optionflow.db)")
    parser.add_argument("--expiry", required=True, help="Expiration ciblée (YYYYMMDD ou YYYY-MM-DD)")
    parser.add_argument(
        "--tables",
        help="Liste de tables à traiter (séparées par des virgules). "
             "Par défaut: toutes les tables qui possèdent une colonne d'expiration."
    )
    parser.add_argument("--force", action="store_true", help="Exécuter la suppression (sinon dry-run).")
    parser.add_argument("--no-backup", action="store_true", help="Ne pas créer de sauvegarde .bak.")
    args = parser.parse_args()

    if not os.path.exists(args.db):
        raise SystemExit(f"Base introuvable: {args.db}")

    try:
        expiry_norm = normalize_expiry(args.expiry)
    except ValueError as e:
        raise SystemExit(str(e))

    allowed_tables = None
    if args.tables:
        allowed_tables = [t.strip() for t in args.tables.split(",") if t.strip()]

    con = sqlite3.connect(args.db)
    try:
        con.execute("PRAGMA foreign_keys = ON;")
        print(f"[INFO] Base: {args.db}")
        print(f"[INFO] Expiration ciblée: {expiry_norm}\n")

        targets = discover_expiry_targets(con, allowed_tables=allowed_tables)
        if not targets:
            if allowed_tables:
                raise SystemExit("[ERREUR] Aucune des tables fournies n'a de colonne d'expiration reconnue.")
            raise SystemExit("[ERREUR] Aucune table avec colonne d'expiration détectée dans la base.")

        # Comptage (dry-run)
        total = 0
        for table, col in targets:
            cnt = count_expiry(con, table, col, expiry_norm)
            total += cnt
            print(f"[CHECK] {table}.{col}: {cnt} ligne(s) correspondant à expiry={expiry_norm}")

        if not args.force:
            print("\n[DRY-RUN] Aucune suppression effectuée. Ajoute --force pour exécuter.")
            return

        # Sauvegarde
        if not args.no_backup:
            ensure_backup(args.db, f"expiry_{expiry_norm}")

        # Suppression effective
        deleted_total = 0
        with con:
            for table, col in targets:
                d = delete_expiry(con, table, col, expiry_norm)
                print(f"[DELETE] {table}: {d} ligne(s) supprimée(s) (expiry={expiry_norm})")
                deleted_total += d

        # Compactage
        con.execute("VACUUM;")
        print(f"\n[OK] Terminé. Total supprimé (par expiration {expiry_norm}) : {deleted_total}")

    finally:
        con.close()

if __name__ == "__main__":
    main()
