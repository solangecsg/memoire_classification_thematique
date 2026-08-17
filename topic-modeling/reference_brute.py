"""
reference_brute.py : corpus de référence non filtré pour le calcul du NPMI

CE QUE FAIT CE SCRIPT

Construit le corpus de référence sur lequel toutes les valeurs de NPMI du
mémoire sont calculées, puis recalcule ces valeurs pour l'ensemble des runs.

POURQUOI UNE RÉFÉRENCE NON FILTRÉE

Le corpus produit par corpus_reference.py applique les filtres de la
configuration retenue : listes d'arrêt, seuil de fréquence de 5, longueur
minimale de trois caractères. Il mesure correctement les runs entraînés sous ces
mêmes filtres.

Il pénalise les autres. Un mot de tête absent de la référence n'y a pas de
probabilité estimable, et metrics_lda.py lui attribue alors un NPMI de -1 contre
tous les autres mots de son thème. Cette valeur signifie "jamais co-occurrents",
quand la situation est "non mesurable". Le run à filtrage grammatical en donne
l'exemple : il conserve des noms que la liste d'arrêt écarte, 26 pour cent de
ses mots de tête manquaient à la référence, et sa valeur tombait à -0,33 sans
que rien de tel ne s'observe dans ses thèmes.

La référence construite ici ne retire rien : ni mots vides, ni formes rares, ni
formes courtes. Elle compte 83 682 formes contre 21 355 pour la précédente. Tout
run y trouve son vocabulaire, quel que soit le filtrage employé à
l'entraînement, et les cent onze runs se mesurent sur un instrument unique.

Le procédé suit la pratique du domaine. Le corpus de référence sert à estimer
des probabilités de cooccurrence et gagne à être large : Lau, Newman et Baldwin,
qui ont établi que le NPMI est la mesure la plus proche du jugement humain,
emploient l'encyclopédie Wikipédia entière.

Les mots vides que cette référence contient ne faussent aucune mesure. Le
dénombrement des cooccurrences, dans metrics_lda.build_cooc, ne porte que sur le
vocabulaire des thèmes évalués. Un mot vide qui ne figure dans aucun thème n'est
jamais compté.

ENTRÉES

  ../re-ocr/corpus/original/, ../re-ocr/corpus/reocr_mistral/   fascicules
  resultats/{run}/topics.json    thèmes dont le NPMI est recalculé
  resultats/{run}/meta.json      source et granularité du run, qui déterminent
                                 la référence à employer

SORTIES

  reference/{source}_{granularite}_brut.txt    un document par ligne
  reference/{source}_{granularite}_brut.json   nombre de documents et
                                               vocabulaire
  resultats/{run}/metrics_brut.json            NPMI recalculé, et nombre de mots
                                               de tête qui manquaient à la
                                               référence filtrée

PAQUETS EMPLOYÉS

  argparse, json, pathlib   bibliothèque standard
  lda_mallet_corpus         lecture des fichiers ALTO et METS, normalisation
  metrics_lda               calcul du NPMI, réemployé tel quel afin que les
                            valeurs restent comparables à celles des autres
                            scripts

Aucune dépendance extérieure n'est nécessaire.

USAGE

    python3 reference_brute.py              # construit les quatre références
    python3 reference_brute.py --mesurer    # recalcule le NPMI de tous les runs
"""

import argparse
import json
from pathlib import Path

import lda_mallet_corpus as base
import metrics_lda as M

ICI = Path(__file__).resolve().parent
REFERENCE = ICI / "reference"
RESULTATS = ICI / "resultats"
MIN_MOTS = {"article": 20, "bloc": 5}


def chemin(source: str, granularite: str) -> Path:
    """Chemin du corpus non filtré pour un couple (source, granularité).

    Le suffixe _brut distingue ces fichiers de ceux que produit
    corpus_reference.py, les deux jeux coexistant dans le même dossier.
    """
    return REFERENCE / f"{source}_{granularite}_brut.txt"


def construire(source: str, granularite: str) -> Path:
    """Construit un corpus de référence non filtré et l'écrit sur disque.

    L'appel à constituer diffère de celui de corpus_reference.py sur trois
    points, qui sont l'objet même de ce module.

      stopwords, stoplocs   deux ensembles vides sont passés, de sorte
                            qu'aucune forme ne soit retirée
      longueur_min = 1      les formes d'un seul caractère sont conservées
      freq_min = 0          aucun seuil de fréquence n'est appliqué

    Le seuil min_mots demeure en revanche celui de la configuration retenue,
    20 mots pour l'article et 5 pour le bloc. Il ne porte pas sur le vocabulaire
    mais sur les documents, et le modifier changerait la composition du corpus
    plutôt que son vocabulaire.

    Rend le chemin du fichier texte écrit.
    """
    racine = base.SOURCES[source]["racine"]
    suffixe = base.SOURCES[source]["suffixe"]
    fascicules = sorted(
        d.name[:-len(suffixe)] if suffixe and d.name.endswith(suffixe) else d.name
        for d in racine.iterdir() if d.is_dir())
    # Aucune liste d'arrêt, aucun seuil de fréquence, longueur minimale de 1.
    docs = base.constituer(source, granularite, fascicules,
                           MIN_MOTS[granularite], set(), set(), 1, 0)
    REFERENCE.mkdir(exist_ok=True)
    f = chemin(source, granularite)
    f.write_text("\n".join(d["traite"] for d in docs), encoding="utf-8")
    vocab = len({j for d in docs for j in d["traite"].split()})
    f.with_suffix(".json").write_text(json.dumps(
        {"source": source, "granularite": granularite, "n_docs": len(docs),
         "vocabulaire": vocab, "filtrage": "aucun",
         "min_mots": MIN_MOTS[granularite]}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"  {f.name} : {len(docs)} documents, {vocab} formes")
    return f


def charger(source: str, granularite: str) -> list[list[str]]:
    """Rend le corpus non filtré sous forme de liste de listes de mots.

    Le corpus est construit à la demande s'il manque.
    """
    f = chemin(source, granularite)
    if not f.exists():
        construire(source, granularite)
    return [l.split() for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]


def mesurer() -> None:
    """Recalcule le NPMI de tous les runs sur la référence non filtrée.

    Le traitement parcourt resultats/ et, pour chaque run muni d'un meta.json et
    d'un topics.json, procède ainsi.

    1. Lire la source et la granularité dans meta.json, qui déterminent laquelle
       des quatre références employer.
    2. Charger cette référence, en la construisant si elle manque. Les corpus
       chargés sont gardés en mémoire dans le dictionnaire caches, chacun étant
       employé par des dizaines de runs et sa lecture prenant plusieurs secondes.
    3. Prendre les dix premiers mots de chaque thème. Cette profondeur est celle
       qu'emploient les travaux du domaine, ce qui rend les valeurs comparables
       à la littérature.
    4. Calculer le NPMI moyen par metrics_lda.mean_npmi.
    5. Compter les mots de tête absents de la référence. Le compte doit valoir
       zéro pour tous les runs, et sa consignation permet de le vérifier.
    6. Écrire metrics_brut.json dans le dossier du run.

    Un tableau comparatif est imprimé en fin de traitement, qui met en regard la
    valeur obtenue sur la référence filtrée et celle obtenue ici.
    """
    caches: dict[tuple, list] = {}
    lignes = []
    for d in sorted(RESULTATS.iterdir()):
        meta_f, top_f = d / "meta.json", d / "topics.json"
        if not (meta_f.exists() and top_f.exists()):
            continue
        meta = json.loads(meta_f.read_text(encoding="utf-8"))
        src, gran = meta.get("source"), meta.get("granularite")
        if src not in base.SOURCES or gran not in MIN_MOTS:
            continue
        cle = (src, gran)
        if cle not in caches:
            print(f"référence {src} {gran}…")
            caches[cle] = charger(src, gran)
        corpus = caches[cle]
        vocab = {w for doc in corpus for w in doc}
        tops = [t["top_words"].split()[:10]
                for t in json.loads(top_f.read_text(encoding="utf-8"))
                if t["top_words"].strip()]
        if not tops:
            continue
        mots = [w for t in tops for w in t]
        npmi = M.mean_npmi(tops, corpus)
        hors = sum(1 for w in mots if w not in vocab)
        (d / "metrics_brut.json").write_text(json.dumps(
            {"reference": "brute", "npmi": npmi,
             "mots_hors_reference": hors, "mots_total": len(mots)},
            ensure_ascii=False, indent=1), encoding="utf-8")
        ancien = None
        if (d / "metrics_ref.json").exists():
            ancien = json.loads((d / "metrics_ref.json").read_text(
                encoding="utf-8"))["npmi"]["npmi_mean"]
        lignes.append((d.name, ancien, npmi["npmi_mean"], hors, len(mots)))

    print(f"\n{len(lignes)} runs mesurés sur référence non filtrée\n")
    print(f"{'run':56}{'filtrée':>9}{'brute':>9}{'écart':>8}{'hors réf.':>11}")
    for nom, a, b, h, n in lignes:
        av = f"{a:+.3f}" if a is not None else "   .  "
        ec = f"{b-a:+.3f}" if a is not None else "   .  "
        print(f"{nom[:56]:56}{av:>9}{b:>+9.3f}{ec:>8}{100*h/n:>10.0f} %")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mesurer", action="store_true")
    args = ap.parse_args()
    if args.mesurer:
        mesurer()
    else:
        for s in sorted(base.SOURCES):
            for g in MIN_MOTS:
                construire(s, g)
