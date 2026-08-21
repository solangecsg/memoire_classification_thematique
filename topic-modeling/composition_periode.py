"""
composition_periode.py : composition thématique du corpus par tranche de vingt ans

CE QUE FAIT CE SCRIPT

Agrège les affectations d'un modèle thématique par période de publication, et
rend le tableau que reprend le mémoire. Aucun entraînement n'est refait :
l'affectation des documents porte l'identifiant du fascicule, et il suffit de
rattacher chaque fascicule à sa date.

D'OÙ VIENNENT LES DATES

Chaque fascicule du corpus est accompagné de son manifeste, dont l'élément
dc:date porte la date de publication. La date est donc lue dans le corpus
lui-même plutôt que dans une source extérieure, ce qui rend le calcul
reproductible à partir des seuls fichiers versés.

Trois formes se rencontrent dans cet élément et sont toutes trois reconnues :
la date simple, la date écrite avec des barres obliques, et l'intervalle de deux
dates, dont la première est retenue.

Un contrôle facultatif confronte ces dates au tableur des URLs Gallica, qui
recense les cent fascicules avec leur titre et leur date. Les deux sources
s'accordent sur les cent, les seules divergences relevées tenant à l'écriture.

POURQUOI CE SCRIPT EXISTE

Une première version du relevé ne portait que sur soixante-sept fascicules, le
tiers restant étant réputé sans date. Le contrôle a montré que les cent dates
étaient disponibles, dans le manifeste comme dans le tableur. Le présent script
refait le calcul sur l'ensemble du corpus.

ENTRÉES

  {corpus}/{fascicule}_reocr/manifest.xml                  dates de publication
  resultats/lda_corpus_bnf_article_k10_g1_*/span_topic.json  affectations
  resultats/lda_corpus_bnf_article_k10_g1_*/topics.json      mots de tête
  ../../100_urls_gallica_recap.xlsx                        contrôle facultatif

SORTIES

  Un tableau imprimé sur la sortie standard, en LaTeX avec --latex.

PAQUETS EMPLOYÉS

  openpyxl                          lecture du tableur de contrôle, facultative
  json, re, glob, collections, argparse, pathlib   bibliothèque standard

Le manifeste est lu par expression régulière plutôt que par un analyseur XML.
Un seul élément est cherché, dont la forme est stable, et cette lecture évite
d'analyser cent fichiers de cent kilooctets pour en extraire une ligne.

USAGE

    python3 composition_periode.py
    python3 composition_periode.py --latex
    python3 composition_periode.py --controle
"""

import argparse
import collections
import glob
import json
import re
from pathlib import Path

ICI = Path(__file__).resolve().parent

# Tranches de vingt ans. Le corpus s'étend de 1819 à 1953, et ce pas donne des
# effectifs comparables d'une tranche à l'autre sans lisser les mouvements que
# le mémoire commente.
PAS = 20
DEBUT = 1800

# Intitulés donnés aux dix thèmes du modèle repris dans le mémoire. Ils sont
# lus par un humain sur les mots de tête, opération que le chapitre décrit comme
# le point faible de la méthode : ils servent à la lecture du tableau et ne sont
# produits par aucun calcul.
NOMS = {0: "locale", 1: "bruit", 2: "sport", 3: "bourse", 4: "annonces",
        5: "guerre", 6: "politique", 7: "spect.", 8: "divers", 9: "chron."}
ORDRE = ["locale", "bruit", "sport", "bourse", "annonces", "guerre",
         "politique", "spect.", "chron.", "divers"]


def fascicules() -> Path:
    """Dossier qui contient les fascicules ré-océrisés, un par sous-dossier.

    Deux emplacements sont possibles selon le contexte. Dans le dossier de
    travail, les fascicules se trouvent sous resultats_mistral/. Dans le dépôt,
    ils sont versés sous re-ocr/corpus/reocr_mistral/. Les deux portent la même
    structure interne, un dossier {identifiant}_reocr contenant son manifeste et
    son sous-dossier ocr/, et le premier qui existe est retenu.
    """
    for base in [ICI, *ICI.parents]:
        for rel in (Path("resultats_mistral"),
                    Path("re-ocr") / "corpus" / "reocr_mistral"):
            if (base / rel).is_dir():
                return base / rel
    raise SystemExit(
        "fascicules ré-océrisés introuvables. Ils sont attendus sous "
        "resultats_mistral/ dans le dossier de travail, ou sous "
        "re-ocr/corpus/reocr_mistral/ dans le dépôt.")


def dates():
    """Rend la date de publication de chaque fascicule, lue dans son manifeste.

    L'élément dc:date accepte trois écritures dans ce corpus. La date simple,
    2018-11-07. La date à barres obliques, 1899/12/31. Et l'intervalle,
    1953-03-01/1953-03-02, pour un fascicule couvrant plusieurs jours, dont la
    première date est retenue puisque c'est celle de parution.
    """
    out = {}
    for d in sorted(fascicules().glob("*_reocr")):
        fid = d.name.split("_")[0]
        f = d / "manifest.xml"
        if not f.is_file():
            continue
        t = f.read_text(encoding="utf-8", errors="ignore")
        m = re.search(r"<dc:date[^>]*>([^<]+)</dc:date>", t)
        if not m:
            continue
        brut = m.group(1).strip().split("/") if "/" not in m.group(1)[:8] \
            else [m.group(1).strip()]
        premiere = m.group(1).strip().split("/")[0] if m.group(1).count("/") > 1 \
            else m.group(1).strip().split("/")[0]
        an = re.match(r"(1[89]\d{2}|20\d{2})", premiere.replace("/", "-"))
        if an:
            out[fid] = int(an.group(1))
    return out


def controle(par_fascicule):
    """Confronte les dates du manifeste au tableur des URLs Gallica.

    Le tableur donne l'ARK de chaque fascicule, dont les sept chiffres qui
    suivent bpt6k forment l'identifiant employé dans le corpus, la dernière
    position de l'ARK étant un caractère de contrôle.
    """
    try:
        import openpyxl
    except ImportError:
        return "openpyxl absent, contrôle non conduit"
    cands = [c / "100_urls_gallica_recap.xlsx" for c in fascicules().parents]
    f = next((c for c in cands if c.is_file()), None)
    if f is None:
        return "tableur absent, contrôle non conduit"
    w = openpyxl.load_workbook(f)
    tab = {}
    for _, _titre, date, url in w[w.sheetnames[0]].iter_rows(min_row=2, values_only=True):
        m = re.search(r"bpt6k(\d{7})", str(url))
        if m:
            tab[m.group(1)] = int(str(date).split("/")[-1])
    communs = set(tab) & set(par_fascicule)
    acc = sum(1 for k in communs if tab[k] == par_fascicule[k])
    return (f"{len(tab)} fascicules au tableur, {len(communs)} communs, "
            f"{acc} années identiques")


def affectations():
    """Rend le fascicule et le thème de chaque document, pour le run du mémoire."""
    dossiers = sorted(glob.glob(str(ICI / "resultats" /
                                    "lda_corpus_bnf_article_k10_g1_2*")))
    if not dossiers:
        raise SystemExit("exécution introuvable : resultats/lda_corpus_bnf_article_k10_g1_*")
    run = dossiers[0]
    f = Path(run, "span_topic.json")
    if not f.is_file():
        raise SystemExit(
            f"affectation des documents absente : {f.name}\n"
            "Elle n'est pas versée dans le dépôt, étant volumineuse et "
            "régénérable. La reconstituer par :\n"
            "    python3 lda_mallet_corpus.py --source bnf --granularite article "
            "--k 10 --graine 1")
    d = json.loads(f.read_text(encoding="utf-8"))
    if isinstance(d, dict):
        d = d.get("spans", list(d.values()))
    return [(x["fascicule"], int(x["topic_id"])) for x in d
            if isinstance(x, dict) and x.get("fascicule") is not None]


def tableau():
    """Croise les périodes et les thèmes, en pourcentage des articles de la période."""
    an = dates()
    docs = affectations()
    sans = sorted({f for f, _ in docs} - set(an))
    par_periode = collections.defaultdict(collections.Counter)
    for f, t in docs:
        if f in an:
            par_periode[(an[f] - DEBUT) // PAS][t] += 1
    lignes = []
    for tranche in sorted(par_periode):
        c = par_periode[tranche]
        n = sum(c.values())
        deb = DEBUT + tranche * PAS
        lignes.append((f"{deb}-{deb + PAS - 1}", n,
                       {NOMS[t]: round(100 * v / n) for t, v in c.items()}))
    return lignes, len(an), sans, sum(n for _, n, _ in lignes), len(docs)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--latex", action="store_true", help="écrire le corps du tableau LaTeX")
    p.add_argument("--controle", action="store_true", help="confronter au tableur Gallica")
    args = p.parse_args()

    lignes, n_dates, sans, couverts, total = tableau()
    if args.controle:
        print(controle(dates()))
        return
    if args.latex:
        for periode, n, parts in lignes:
            cases = " & ".join(f"{parts.get(k, 0)}" for k in ORDRE)
            print(f"{periode} & {n:<5} & {cases} \\\\")
        return
    print(f"fascicules datés : {n_dates}")
    print(f"articles couverts : {couverts} sur {total}")
    if sans:
        print(f"fascicules sans date : {', '.join(sans)}")
    entete = f"{'Période':<10}{'n':>6}  " + "".join(f"{k:>10}" for k in ORDRE)
    print("\n" + entete)
    for periode, n, parts in lignes:
        print(f"{periode:<10}{n:>6}  " + "".join(f"{parts.get(k, 0):>10}" for k in ORDRE))


if __name__ == "__main__":
    main()
