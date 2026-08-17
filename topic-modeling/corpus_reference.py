"""
corpus_reference.py : corpus de référence filtré pour le calcul du NPMI

CE QUE FAIT CE SCRIPT

Le NPMI mesure la cohérence d'un thème par la fréquence à laquelle ses mots de
tête paraissent ensemble dans les documents d'un corpus de référence. Ce corpus
sert uniquement à estimer des probabilités de cooccurrence. Il n'entre pas dans
l'entraînement des modèles.

Tant que chaque run écrivait son propre corpus de référence, deux familles de
méthodes n'étaient pas comparables. Celui de la modélisation thématique comptait
21 355 formes, filtrées par le seuil de fréquence. Celui du regroupement de
plongements en comptait 81 704, le même filtrage n'ayant pas été appliqué.
Comparer 0,250 et 0,276 revenait à mesurer avec deux règles différentes.

Ce module produit donc un corpus de référence unique par couple (source,
granularité), soit quatre fichiers, et l'enregistre dans reference/.

LIMITE CONNUE DE CE FICHIER

La normalisation appliquée ici est celle de la configuration retenue : listes
d'arrêt, seuil de fréquence de 5, longueur minimale de trois caractères. Un mot
de tête absent de cette référence n'y a pas de probabilité estimable, et
metrics_lda.py lui attribue alors un NPMI de -1 contre tous les autres mots de
son thème. Les configurations entraînées sous d'autres filtres se trouvent donc
écrasées par un effet de plancher.

Pour cette raison, les valeurs rapportées dans le mémoire proviennent de
reference_brute.py, qui construit une référence sans aucun filtrage. Le présent
module est conservé pour la reproduction des mesures intermédiaires.

ENTRÉES

  ../re-ocr/corpus/original/{fascicule}/          texte hérité de l'océrisation
  ../re-ocr/corpus/reocr_mistral/{fascicule}_reocr/  texte ré-océrisé
  stopwords_fr_presse.txt, stopwords_extra.txt   formes à retirer
  stoplocs_fr_presse.txt                         locutions à retirer

SORTIES

  reference/{source}_{granularite}.txt    un document par ligne, texte normalisé
  reference/{source}_{granularite}.json   nombre de documents, taille du
                                          vocabulaire et filtres appliqués

PAQUETS EMPLOYÉS

  argparse   analyse des arguments de la ligne de commande, bibliothèque
             standard
  json       écriture du fichier descriptif accompagnant chaque corpus,
             bibliothèque standard
  pathlib.Path         manipulation des chemins, bibliothèque standard

  lda_mallet_corpus    module du même dossier. Il fournit la lecture des
                       fichiers ALTO et METS, la normalisation du texte et le
                       chargement des listes d'arrêt. L'importer garantit que le
                       corpus de référence subit exactement le même traitement
                       que le corpus d'entraînement.

Aucune dépendance extérieure n'est nécessaire.

USAGE

    python3 corpus_reference.py                  # construit ce qui manque
    python3 corpus_reference.py --refaire        # reconstruit les quatre corpus
"""

import argparse
import json
from pathlib import Path

import lda_mallet_corpus as base

ICI = Path(__file__).resolve().parent
REFERENCE = ICI / "reference"

# ── Paramètres de normalisation ───────────────────────────────────────────────
# Ces trois valeurs reprennent celles de la configuration retenue au chapitre 1,
# de sorte que le corpus de référence et le corpus d'entraînement portent le même
# vocabulaire.

# Longueur minimale d'une forme, en caractères. À 3, les débris d'océrisation
# d'une ou deux lettres sont écartés sans perdre de mot français utile.
LONGUEUR_MIN = 3

# Nombre minimal de documents où une forme doit paraître pour être conservée.
# À 5, les formes attestées une ou deux fois sont retirées. Elles proviennent
# presque toutes d'erreurs d'océrisation, ne se répètent jamais, et gonflent le
# vocabulaire sans porter d'information.
FREQ_MIN = 5

# Nombre minimal de mots qu'un document doit conserver après filtrage pour être
# retenu. Le seuil diffère selon la granularité parce que les deux unités n'ont
# pas le même ordre de grandeur : l'article compte 114 mots en médiane, le bloc
# 17. Un seuil unique écarterait soit trop de blocs, soit aucun article vide.
MIN_MOTS = {"article": 20, "bloc": 5}


def chemin(source: str, granularite: str) -> Path:
    """Chemin du fichier de corpus pour un couple (source, granularité).

    Le fichier descriptif porte le même nom avec l'extension .json.
    """
    return REFERENCE / f"{source}_{granularite}.txt"


def construire(source: str, granularite: str) -> Path:
    """Construit un corpus de référence et l'écrit sur disque.

    Le traitement se fait en quatre temps.

    1. Lister les fascicules présents dans le dossier de la source. Le nom du
       dossier porte un suffixe pour la source ré-océrisée, retiré ici pour
       retrouver l'identifiant du fascicule.
    2. Appeler lda_mallet_corpus.constituer, qui lit les fichiers ALTO et METS,
       reconstitue les unités demandées et normalise leur texte.
    3. Écrire un document par ligne dans le fichier texte. metrics_lda.py lit ce
       format directement.
    4. Écrire à côté un fichier JSON qui consigne le nombre de documents, la
       taille du vocabulaire et les filtres employés, afin qu'une mesure puisse
       toujours être rapportée aux conditions de sa production.

    Rend le chemin du fichier texte écrit.
    """
    racine = base.SOURCES[source]["racine"]
    suffixe = base.SOURCES[source]["suffixe"]
    fascicules = sorted(
        d.name[:-len(suffixe)] if suffixe and d.name.endswith(suffixe) else d.name
        for d in racine.iterdir() if d.is_dir())

    docs = base.constituer(source, granularite, fascicules,
                           MIN_MOTS[granularite], base.charger_stopwords(),
                           base.charger_stoplocs(), LONGUEUR_MIN, FREQ_MIN)

    REFERENCE.mkdir(exist_ok=True)
    f = chemin(source, granularite)
    # La clé "traite" porte le texte normalisé. La clé "texte" porte le texte
    # brut, employé par les modèles de plongement et sans usage ici.
    f.write_text("\n".join(d["traite"] for d in docs), encoding="utf-8")

    vocab = len({j for d in docs for j in d["traite"].split()})
    meta = {"source": source, "granularite": granularite,
            "n_docs": len(docs), "vocabulaire": vocab,
            "longueur_min": LONGUEUR_MIN, "freq_min": FREQ_MIN,
            "min_mots": MIN_MOTS[granularite]}
    f.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    print(f"  {f.name} : {len(docs)} documents, {vocab} formes distinctes")
    return f


def charger(source: str, granularite: str) -> list[list[str]]:
    """Rend le corpus de référence sous forme de liste de listes de mots.

    Le corpus est construit à la demande s'il manque, ce qui évite d'imposer un
    ordre d'exécution aux scripts qui l'emploient. Les lignes vides sont
    ignorées.
    """
    f = chemin(source, granularite)
    if not f.exists():
        construire(source, granularite)
    return [l.split() for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> None:
    """Construit les quatre corpus de référence.

    Sans argument, seuls les corpus manquants sont construits, et les autres
    sont décrits d'après leur fichier JSON. L'option --refaire reconstruit tout,
    ce qui est nécessaire après une modification des listes d'arrêt ou des
    seuils déclarés en tête de ce fichier.
    """
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refaire", action="store_true",
                    help="reconstruire les corpus déjà présents")
    args = ap.parse_args()

    for source in ("bnf", "mistral"):
        for granularite in ("article", "bloc"):
            if args.refaire or not chemin(source, granularite).exists():
                print(f"\n{source} / {granularite}")
                construire(source, granularite)
            else:
                m = json.loads(chemin(source, granularite)
                               .with_suffix(".json").read_text(encoding="utf-8"))
                print(f"  déjà là : {source}/{granularite}, {m['n_docs']} documents, "
                      f"{m['vocabulaire']} formes")


if __name__ == "__main__":
    main()
