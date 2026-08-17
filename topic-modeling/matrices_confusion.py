"""
matrices_confusion.py : cartes de chaleur en TikZ pour deux confrontations

CE QUE FAIT CE SCRIPT

Produit le code TikZ de deux matrices, à coller dans le mémoire.

  1. ENCODEUR CONTRE MODÈLE DE LANGUE. Croise les dix-sept catégories de premier
     niveau que prédit le classifieur affiné avec les ancêtres de premier niveau
     des étiquettes que le modèle de langue attribue au troisième. La diagonale
     porte les accords, le reste montre où les deux dispositifs divergent.

  2. LES DEUX FAMILLES DE REGROUPEMENT. Croise les vingt thèmes de la
     modélisation probabiliste avec les vingt groupes du regroupement de
     plongements, sur les articles communs. Les deux partitions sont
     indépendantes et leurs numéros arbitraires : les lignes et les colonnes
     sont donc réordonnées pour amener les correspondances sur la diagonale,
     faute de quoi la carte serait illisible.

     L'information mutuelle normalisée accompagne la matrice. Elle vaut 0 quand
     deux partitions sont sans rapport et 1 quand elles coïncident, et se lit
     indépendamment du nombre de groupes.

POURQUOI UNE CARTE DE CHALEUR

Une matrice de dix-sept ou vingt colonnes remplie de nombres ne se lit pas.
L'intensité d'une case permet de repérer les zones sans lire les valeurs, et
c'est bien une structure que l'on cherche : une catégorie qui absorbe les cas
douteux se voit comme une ligne foncée, deux partitions qui se croisent comme
une diagonale absente.

L'échelle est racine carrée plutôt que linéaire. La diagonale écrase tout le
reste sur une échelle linéaire, et les cellules hors diagonale, qui portent
l'information, deviendraient invisibles.

ENTRÉES

  resultats/classla_{...}/predictions.json          prédictions de l'encodeur
  ../../../../github/classification-iptc/results/feuilles_mistral_batched/
                                                    classification au niveau 3
  ../../../../github/classification-iptc/classification/iptc_mediatopic_official.json
  resultats/lda_corpus_mistral_article_k20_.../span_topic.json
  resultats/bertopic_kmeans_mistral_article_k20_.../span_topic.json

SORTIES

  matrices/confusion_iptc.tex      figure TikZ, encodeur contre modèle de langue
  matrices/confusion_familles.tex  figure TikZ, les deux familles

PAQUETS EMPLOYÉS

  json, glob, math, collections, pathlib   bibliothèque standard

Aucune dépendance extérieure. Le tracé est confié à TikZ plutôt qu'à une
bibliothèque de graphiques, pour que les figures emploient les fontes du mémoire
et se règlent dans son code source.

USAGE

    python3 matrices_confusion.py
"""

import collections
import glob
import json
import math
from pathlib import Path

ICI = Path(__file__).resolve().parent
SORTIE = ICI / "matrices"


def depot() -> Path:
    """Racine du dépôt de classification, cherchée en remontant l'arborescence."""
    for base in [ICI, *ICI.parents]:
        for rel in (Path("github") / "classification-iptc", Path(".")):
            if (base / rel / "classification" / "iptc_mediatopic_official.json").is_file():
                return (base / rel).resolve()
    raise SystemExit("dépôt de classification introuvable")


def taxonomie(d: Path):
    """Rend les tables nécessaires à la remontée dans l'arbre IPTC."""
    t = json.loads((d / "classification" / "iptc_mediatopic_official.json")
                   .read_text(encoding="utf-8"))
    conc = {c["qcode"].removeprefix("medtop:"): c for c in t["conceptSet"]}
    racines = {u.rsplit("/", 1)[-1] for u in t["hasTopConcept"]}
    fr = {k: (c["prefLabel"].get("fr") or c["prefLabel"].get("en-GB"))
          for k, c in conc.items()}
    par_en = {c["prefLabel"]["en-GB"]: c["qcode"].removeprefix("medtop:")
              for c in t["conceptSet"]
              if c["qcode"].removeprefix("medtop:") in racines
              and c["prefLabel"].get("en-GB")}
    # Le classifieur rend « politics » là où le schéma écrit « politics and
    # government ». Sans cette correspondance, la catégorie la plus fournie du
    # corpus tomberait entièrement hors diagonale.
    par_en["politics"] = par_en["politics and government"]

    def remonter(code):
        vu = set()
        while code not in racines and code not in vu:
            vu.add(code)
            b = conc.get(code, {}).get("broader")
            if not b:
                return None
            code = b[0].rsplit("/", 1)[-1]
        return code if code in racines else None

    return remonter, fr, par_en


def carte(M, lignes, colonnes, etiq_l, etiq_c, taille=0.46):
    """Rend le code TikZ d'une carte de chaleur.

    L'intensité suit la racine carrée de l'effectif, rapportée au maximum hors
    diagonale lorsque celle-ci écrase le reste.
    """
    mx = max(M.values()) if M else 1
    out = [r"\begin{tikzpicture}[x=%.2fcm,y=%.2fcm]" % (taille, taille)]
    for j, c in enumerate(colonnes):
        for i, l in enumerate(lignes):
            v = M.get((l, c), 0)
            if not v:
                continue
            n = min(1.0, math.sqrt(v / mx))
            out.append(r"  \fill[black!%d] (%d,%d) rectangle ++(1,1);"
                       % (int(8 + 84 * n), j, -i))
            if v >= mx * 0.25:
                out.append(r"  \node[font=\tiny,white] at (%.1f,%.1f) {%d};"
                           % (j + 0.5, -i + 0.5, v))
    for i, l in enumerate(lignes):
        out.append(r"  \node[anchor=east,font=\scriptsize] at (0,%.1f) {%s};"
                   % (-i + 0.5, etiq_l(l)))
    for j, c in enumerate(colonnes):
        out.append(r"  \node[anchor=west,rotate=55,font=\scriptsize] at (%.1f,1.15) {%s};"
                   % (j + 0.2, etiq_c(c)))
    out.append(r"  \draw[gray!50] (0,%d) grid[step=%.2f] (%d,1);"
               % (-len(lignes) + 1, 1, len(colonnes)))
    out.append(r"\end{tikzpicture}")
    return "\n".join(out)


def matrice_iptc():
    """Encodeur affiné contre modèle de langue, dix-sept catégories."""
    D = depot()
    remonter, fr, par_en = taxonomie(D)
    mis = {}
    for f in sorted((D / "results" / "feuilles_mistral_batched").glob("*_themes.json")):
        x = json.loads(f.read_text(encoding="utf-8"))
        for a in x["articles"]:
            if a.get("themes"):
                mis[f"{x['fascicule']}:{a['article_id']}"] = a["themes"]
    run = sorted(glob.glob(str(ICI / "resultats" / "classla_*")))[-1]
    pred = {p["doc_id"]: p for p in
            json.loads(Path(run, "predictions.json").read_text(encoding="utf-8"))}

    M = collections.Counter()
    for k in set(pred) & set(mis):
        e = par_en.get(pred[k]["etiquette"])
        p = remonter(mis[k][0]["code"])
        if e and p:
            M[(e, p)] += 1
    ordre = sorted({a for a, _ in M} | {b for _, b in M},
                   key=lambda c: -sum(v for (x, _), v in M.items() if x == c))
    tot = sum(M.values())
    diag = sum(v for (a, b), v in M.items() if a == b)
    court = lambda c: fr[c].split(",")[0]
    return carte(M, ordre, ordre, court, court), tot, diag, len(M)


def partitions(run):
    """Affectation d'un document à un thème ou à un groupe, depuis span_topic.

    Le fichier span_topic.json pèse plus d'un gigaoctet sur l'ensemble des runs
    et ne figure pas dans le dépôt. Son absence est signalée plutôt que laissée
    remonter, la première matrice restant productible sans lui.
    """
    f = Path(run, "span_topic.json")
    if not f.is_file():
        raise FileNotFoundError(f)
    d = json.loads(f.read_text(encoding="utf-8"))
    if isinstance(d, dict):
        d = d.get("spans", list(d.values()))
    out = {}
    for x in d:
        if not isinstance(x, dict):
            continue
        k = x.get("doc_id") or x.get("span_id") or x.get("id")
        t = x.get("topic") if x.get("topic") is not None else x.get("topic_id")
        if k is not None and t is not None:
            out[k] = t
    return out


def matrice_familles():
    """Les deux familles de regroupement, sur les articles communs."""
    a = partitions(sorted(glob.glob(str(ICI / "resultats" /
        "lda_corpus_mistral_article_k20_g1_f5_l3_*")))[0])
    b = partitions(sorted(glob.glob(str(ICI / "resultats" /
        "bertopic_kmeans_mistral_article_k20_e5_g1_*")))[0])
    com = set(a) & set(b)
    M = collections.Counter((a[k], b[k]) for k in com)

    # Les numéros des deux partitions sont arbitraires. On apparie chaque thème
    # au groupe qui partage le plus d'articles avec lui, sans réemploi, ce qui
    # amène les correspondances sur la diagonale.
    lignes = sorted({x for x, _ in M}, key=lambda x: -sum(v for (i, _), v in M.items() if i == x))
    restants = list({y for _, y in M})
    colonnes = []
    for l in lignes:
        if not restants:
            break
        best = max(restants, key=lambda c: M.get((l, c), 0))
        colonnes.append(best)
        restants.remove(best)
    colonnes += restants

    n = len(com)
    ca = collections.Counter(a[k] for k in com)
    cb = collections.Counter(b[k] for k in com)
    Ha = -sum(v / n * math.log(v / n) for v in ca.values())
    Hb = -sum(v / n * math.log(v / n) for v in cb.values())
    I = sum(v / n * math.log((v / n) / ((ca[x] / n) * (cb[y] / n)))
            for (x, y), v in M.items())
    nmi = 2 * I / (Ha + Hb)
    return (carte(M, lignes, colonnes, lambda x: str(x), lambda y: str(y)),
            n, nmi, len(M))


def main():
    SORTIE.mkdir(exist_ok=True)
    tikz, tot, diag, cellules = matrice_iptc()
    (SORTIE / "confusion_iptc.tex").write_text(tikz, encoding="utf-8")
    print(f"confusion_iptc.tex     {tot} articles, diagonale {diag} "
          f"({diag/tot:.1%}), {cellules}/289 cellules")

    try:
        tikz, n, nmi, cellules = matrice_familles()
    except (FileNotFoundError, IndexError):
        print("confusion_familles.tex non produite : l'affectation des documents "
              "manque.\n  Elle n'est pas versée dans le dépôt, étant volumineuse "
              "et régénérable.\n  La reconstituer par :\n"
              "    python3 lda_mallet_corpus.py --source mistral "
              "--granularite article --k 20 --graine 1\n"
              "    python3 bertopic_corpus.py --regroupement kmeans --k 20 "
              "--modele e5 --graine 1")
        return
    (SORTIE / "confusion_familles.tex").write_text(tikz, encoding="utf-8")
    print(f"confusion_familles.tex {n} articles, information mutuelle "
          f"normalisée {nmi:.3f}, {cellules}/400 cellules")


if __name__ == "__main__":
    main()
