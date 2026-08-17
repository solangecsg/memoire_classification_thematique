"""
recap_runs.py : tableau récapitulatif de tous les runs de modélisation

CE QUE FAIT CE SCRIPT

Parcourt le dossier resultats/, lit les fichiers descriptifs de chaque run et
écrit un tableau unique dans RESULTATS-CAMPAGNE.md. Le fichier sert de relevé de
campagne : il permet de retrouver les conditions exactes d'une mesure citée dans
le mémoire, et de comparer des runs sans ouvrir leurs dossiers un par un.

Deux indicateurs sont calculés ici, car ils ne figurent dans aucun des fichiers
produits par les scripts d'entraînement.

  themes_bruit  nombre de thèmes occupés par des débris d'océrisation. Un thème
                de ce genre réunit des fragments de mots que la reconnaissance
                de caractères a produits, et capte les documents les plus
                abîmés. Son apparition signale une qualité de texte insuffisante.
  rapport       effectif du thème le plus fourni divisé par celui du moins
                fourni. Il décrit la forme de la partition : une valeur voisine
                de 1 signale des thèmes de fréquence égale, hypothèse que la loi
                a priori symétrique impose et qu'une collection de presse
                contredit.

ENTRÉES

  resultats/{run}/meta.json          paramètres du run
  resultats/{run}/topics.json        thèmes et mots de tête
  resultats/{run}/metrics_brut.json  métriques sur la référence non filtrée
  resultats/{run}/metrics_ref.json   métriques sur la référence filtrée
  resultats/{run}/metrics.json       métriques anciennes, référence propre au run

Les trois fichiers de métriques sont cherchés dans cet ordre. Le premier trouvé
est employé, metrics_brut.json fournissant les valeurs retenues dans le mémoire.

SORTIE

  RESULTATS-CAMPAGNE.md    un tableau en Markdown, une ligne par run

PAQUETS EMPLOYÉS

  argparse    analyse des arguments, bibliothèque standard
  json        lecture des fichiers de chaque run, bibliothèque standard
  subprocess  appel de metrics_lda.py pour les runs dont les métriques
              manquent, bibliothèque standard
  sys         chemin de l'interpréteur courant, pour que le sous-processus
              emploie le même environnement, bibliothèque standard
  pathlib     manipulation des chemins, bibliothèque standard

Aucune dépendance extérieure n'est nécessaire.

USAGE

    python3 recap_runs.py
    python3 recap_runs.py --min-docs 1000   # écarte les essais sur un fascicule
    python3 recap_runs.py --calculer        # calcule les métriques absentes
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ICI = Path(__file__).resolve().parent
RESULTATS = ICI / "resultats"
SORTIE = ICI / "RESULTATS-CAMPAGNE.md"


def theme_de_bruit(top_words: str) -> bool:
    """Dit si un thème est occupé par des débris d'océrisation.

    Le critère retenu est qu'au moins cinq des dix mots de tête comptent deux
    caractères ou moins. Les fragments que produit une reconnaissance de
    caractères défaillante sont en effet très courts, quand les mots français
    utiles au classement le sont rarement.

    Le seuil est une convention propre à ce travail. Un autre seuil donnerait un
    autre compte. Sa valeur tient à sa constance : appliqué de la même manière à
    tous les runs, il rend leurs comptes comparables entre eux.
    """
    mots = top_words.split()[:10]
    return bool(mots) and sum(1 for m in mots if len(m) <= 2) >= 5


def lire(dossier: Path) -> dict | None:
    """Lit un dossier de run et rend un dictionnaire prêt pour le tableau.

    Trois fichiers sont consultés, dont un seul est obligatoire.

    meta.json     paramètres du run. Son absence signale un dossier incomplet,
                  et la fonction rend alors None.
    topics.json   thèmes et mots de tête. Il sert à calculer le nombre de thèmes
                  de bruit et le rapport des tailles.
    metrics       métriques déjà calculées. Le NPMI provient de
                  metrics_brut.json lorsqu'il existe, la diversité, l'entropie
                  et la taille du vocabulaire de metrics_ref.json ou du plus
                  ancien metrics.json. Ces trois dernières mesures ne dépendent
                  pas du corpus de référence.

    Les valeurs par défaut passées à get correspondent aux réglages en vigueur
    avant que le paramètre ne soit rendu explicite : granularité du bloc,
    longueur minimale de 2, aucun seuil de fréquence. Elles permettent de lire
    les runs les plus anciens sans les rejouer.
    """
    meta = dossier / "meta.json"
    if not meta.exists():
        return None
    m = json.loads(meta.read_text(encoding="utf-8"))
    d = {
        "run": dossier.name,
        "source": m.get("source"),
        "granularite": m.get("granularite", "bloc"),
        "k": m.get("k"),
        "docs": m.get("n_docs"),
        "jetons": m.get("n_tokens"),
        "iterations": m.get("iterations"),
        "graine": m.get("graine"),
        "optimisation": m.get("optimisation", 0),
        "burn_in": m.get("burn_in"),
        "longueur_min": m.get("longueur_min", 2),
        "freq_min": m.get("freq_min", 0),
        "duree": m.get("duree_s"),
    }
    topics = dossier / "topics.json"
    if topics.exists():
        tp = json.loads(topics.read_text(encoding="utf-8"))
        tailles = sorted(t.get("size", 0) for t in tp) or [0]
        bruit = [t for t in tp if theme_de_bruit(t.get("top_words", ""))]
        d["themes_bruit"] = len(bruit)
        d["docs_bruit"] = sum(t.get("size", 0) for t in bruit)
        d["rapport"] = round(tailles[-1] / tailles[0], 1) if tailles[0] else None
    # Deux fichiers de métriques sont lus, car ils ne portent pas la même chose.
    # metrics_brut.json ne contient que le NPMI recalculé sur la référence non
    # filtrée, qui fournit les valeurs retenues dans le mémoire. La diversité,
    # l'entropie et la taille du vocabulaire ne dépendent pas du corpus de
    # référence et se lisent dans le fichier complet.
    complet = next((dossier / n for n in ("metrics_ref.json", "metrics.json")
                    if (dossier / n).exists()), None)
    if complet is not None:
        mt = json.loads(complet.read_text(encoding="utf-8"))
        d["npmi"] = round(mt["npmi"]["npmi_mean"], 4)
        d["diversite"] = mt["diversity"]["diversity"]
        d["entropie"] = round(mt["entropy"]["entropy_normalized"], 4)
        d["vocabulaire"] = mt["diversity"].get("tokens_distincts")

    brut = dossier / "metrics_brut.json"
    if brut.exists():
        d["npmi"] = round(json.loads(brut.read_text(encoding="utf-8"))
                          ["npmi"]["npmi_mean"], 4)
    return d


def main() -> None:
    """Assemble le tableau et l'écrit dans RESULTATS-CAMPAGNE.md.

    Le seuil --min-docs écarte par défaut les runs portant sur moins de mille
    documents. Ceux-ci sont les essais de juillet conduits sur un seul
    fascicule, dont le mémoire établit que la variance entre exécutions dépasse
    les écarts mesurés. Les faire figurer inviterait à les lire comme des
    résultats.

    L'option --calculer lance metrics_lda.py sur les runs dont les métriques
    manquent. Le sous-processus emploie sys.executable afin de tourner dans le
    même environnement que le script appelant.

    Le tri range les lignes par granularité, source, filtrage, optimisation,
    nombre de thèmes puis graine. Cet ordre rassemble les runs comparables, de
    sorte qu'un écart se lise sur des lignes voisines.
    """
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-docs", type=int, default=1000,
                    help="écarte les runs portant sur moins de N documents")
    ap.add_argument("--calculer", action="store_true",
                    help="calcule les métriques absentes avant le récapitulatif")
    args = ap.parse_args()

    dossiers = sorted(d for d in RESULTATS.iterdir() if d.is_dir())
    runs = []
    for d in dossiers:
        r = lire(d)
        if r and (r["docs"] or 0) >= args.min_docs:
            if args.calculer and not (d / "metrics.json").exists():
                print(f"  métriques de {d.name}…")
                subprocess.run([sys.executable, str(ICI / "metrics_lda.py"),
                                "--run", str(d), "--json"], capture_output=True)
                r = lire(d)
            runs.append(r)

    runs.sort(key=lambda r: (r["granularite"], r["source"], r["freq_min"],
                             r["optimisation"], r["k"] or 0, r["graine"] or 0))

    lignes = [
        "# Campagne de modélisation thématique : récapitulatif",
        "",
        f"Régénéré par `recap_runs.py`. {len(runs)} runs portant sur au moins "
        f"{args.min_docs} documents.",
        "",
        "Colonnes : `f` seuil de fréquence minimale, `l` longueur minimale d'une "
        "forme, `opt` intervalle d'optimisation et amorce, `bruit` nombre de "
        "thèmes occupés par des débris et documents qu'ils captent, `rapp.` "
        "rapport entre le plus gros et le plus petit thème.",
        "",
        "| Granularité | Source | K | Docs | Vocab. | it. | f | l | opt | graine "
        "| NPMI | Divers. | Entropie | bruit | rapp. |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in runs:
        # Colonne opt : intervalle d'optimisation et amorce, ou rien lorsque la
        # loi a priori reste symétrique.
        opt = (f"{r['optimisation']}/{r['burn_in']}" if r["optimisation"] else ".")
        bruit = (f"{r['themes_bruit']} ({r['docs_bruit']})"
                 if r.get("themes_bruit") is not None else "?")
        lignes.append(
            f"| {r['granularite']} | {r['source']} | {r['k']} | {r['docs']} "
            f"| {r.get('vocabulaire','?')} | {r['iterations']} | {r['freq_min']} "
            f"| {r['longueur_min']} | {opt} | {r['graine']} "
            f"| {r.get('npmi','?')} | {r.get('diversite','?')} "
            f"| {r.get('entropie','?')} | {bruit} | {r.get('rapport','?')} |")

    SORTIE.write_text("\n".join(lignes) + "\n", encoding="utf-8")
    print(f"{len(runs)} runs → {SORTIE}")


if __name__ == "__main__":
    main()
