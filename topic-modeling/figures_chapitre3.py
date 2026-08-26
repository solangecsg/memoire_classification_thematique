"""
figures_chapitre3.py : quatre figures pour la classification contre un vocabulaire

CE QUE FAIT CE SCRIPT

Produit les quatre figures du chapitre consacré à la classification contre le
référentiel IPTC. Chacune remplace un passage où le texte demandait au lecteur
de tenir plusieurs nombres en tête.

  1. L'AMORTISSEMENT DE LA LISTE. Le coût par article, mesuré sur les 316 appels
     du corpus, en fonction du nombre d'articles que porte l'appel. La liste des
     567 étiquettes pèse un poids fixe que chaque appel paie une fois, et le
     grouper sur vingt-cinq articles divise ce poids par vingt-cinq. La courbe
     théorique et son asymptote accompagnent les points mesurés : l'asymptote
     est la part incompressible, celle du texte des articles eux-mêmes.

  2. LA COUVERTURE DU RÉFÉRENTIEL. Une case par étiquette candidate, groupée
     par branche, noircie lorsque l'étiquette est employée au moins une fois sur
     le corpus. La figure situe les étiquettes qui ne servent jamais, que le
     texte ne donnait que sous la forme d'un total.

  3. LES DEUX DISTRIBUTIONS. Pour chacune des dix-sept branches, le nombre
     d'articles que lui attribue le classifieur affiné et celui que lui attribue
     le modèle de langue.

     La comparaison demande une précaution. Le modèle de langue rend 2,11
     étiquettes par article là où le classifieur en rend une, et confronter
     directement les deux totaux gonflerait l'écart. Deux comptes sont donc
     portés pour le modèle de langue : sa première étiquette, seule
     comparable terme à terme, et l'ensemble de ses étiquettes, qui dit ce que
     la branche touche.

  4. LA QUEUE DES ÉTIQUETTES. Les 390 étiquettes employées par rang de
     fréquence, en échelle logarithmique. Les 62 employées une seule fois
     forment le plateau terminal.

ENTRÉES

  ../../../../github/classification-iptc/analyse_couts/cout_par_lot.csv
  ../../../../github/classification-iptc/results/feuilles_mistral_batched/
  ../../../../github/classification-iptc/classification/iptc_mediatopic_official.json
  ../../../../github/classification-iptc/classification/classify_iptc_mistral_batched.py
  resultats/classla_{...}/predictions.json

SORTIES

  figures_ch3/amortissement.pdf
  figures_ch3/couverture_referentiel.pdf
  figures_ch3/distributions_branches.pdf
  figures_ch3/queue_etiquettes.pdf

PAQUETS EMPLOYÉS

  matplotlib                          tracé des quatre figures
  csv, json, glob, collections, math, importlib, pathlib   bibliothèque standard

La liste des 567 étiquettes candidates n'est pas recopiée ici. Elle est
reconstruite en appelant la fonction du script de classification qui la fabrique,
chargée par son chemin puisque les deux scripts vivent dans des dossiers
différents du dépôt. Recopier la règle exposerait les deux définitions à diverger.

La fonte du mémoire est chargée par la fonction du script des projections, pour
que toutes les figures de la partie emploient les mêmes caractères.

USAGE

    python3 figures_chapitre3.py
"""

import collections
import csv
import glob
import importlib.util
import json
import math
from pathlib import Path

ICI = Path(__file__).resolve().parent
SORTIE = ICI / "figures_ch3"

# Tarifs du modèle employé, en dollars par million de jetons, tels qu'ils
# figurent dans les scripts de classification et dans le mémoire.
PRIX_ENTREE = 0.5
PRIX_SORTIE = 1.5

# Deux couleurs pour les deux dispositifs, tenues dans toutes les figures.
ENCODEUR = "#b03a2e"
LANGUE = "#1f4e8c"
NEUTRE = "#4d4d4d"
PALE = "#d9d9d9"


def depot() -> Path:
    """Racine du dépôt de classification, cherchée en remontant l'arborescence."""
    for base in [ICI, *ICI.parents]:
        for rel in (Path("github") / "classification-iptc", Path(".")):
            if (base / rel / "classification" / "iptc_mediatopic_official.json").is_file():
                return (base / rel).resolve()
    raise SystemExit("dépôt de classification introuvable")


def police():
    """Charge la fonte du mémoire, par la fonction du script des projections."""
    spec = importlib.util.spec_from_file_location(
        "projections_clusters", ICI / "projections_clusters.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.police()


def feuilles(D):
    """Rend les 567 étiquettes candidates, par la fonction du script de classification.

    Une étiquette candidate est un concept de niveau 3, ou un concept de niveau
    inférieur dont la branche s'arrête avant. La règle vit dans le script qui
    construit l'énoncé soumis au modèle, et c'est lui qu'on interroge.
    """
    chemin = D / "classification" / "classify_iptc_mistral_batched.py"
    spec = importlib.util.spec_from_file_location("classif_batched", chemin)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.build_leaves(D / "classification" / "iptc_mediatopic_official.json")


def taxonomie(D):
    """Rend la remontée vers la racine, les intitulés français et les racines."""
    t = json.loads((D / "classification" / "iptc_mediatopic_official.json")
                   .read_text(encoding="utf-8"))
    conc = {c["qcode"].removeprefix("medtop:"): c for c in t["conceptSet"]}
    racines = {u.rsplit("/", 1)[-1] for u in t["hasTopConcept"]}
    fr = {k: (c["prefLabel"].get("fr") or c["prefLabel"].get("en-GB"))
          for k, c in conc.items()}
    par_en = {c["prefLabel"]["en-GB"]: c["qcode"].removeprefix("medtop:")
              for c in t["conceptSet"]
              if c["qcode"].removeprefix("medtop:") in racines
              and c["prefLabel"].get("en-GB")}
    # Le classifieur affiné rend « politics » là où le schéma écrit « politics
    # and government ».
    par_en["politics"] = par_en["politics and government"]

    def remonter(code):
        """Remonte la chaîne `broader` du référentiel jusqu'à la catégorie de premier
        niveau dont ce code dépend. Rend None si la chaîne se rompt, et s'arrête
        sur les codes déjà vus, une hiérarchie SKOS pouvant se refermer sur
        elle-même."""
        vu = set()
        while code not in racines and code not in vu:
            vu.add(code)
            b = conc.get(code, {}).get("broader")
            if not b:
                return None
            code = b[0].rsplit("/", 1)[-1]
        return code if code in racines else None

    return remonter, fr, racines, par_en


def attributions(D, remonter):
    """Dépouille les sorties du modèle de langue.

    Rend la fréquence de chaque étiquette, les articles rattachés à chaque
    branche par leur première étiquette, ceux qui lui sont rattachés par l'une
    quelconque de leurs étiquettes, et le nombre moyen d'étiquettes par article.
    """
    freq = collections.Counter()
    premiere = collections.defaultdict(set)
    toutes = collections.defaultdict(set)
    n_etiq = []
    for f in sorted((D / "results" / "feuilles_mistral_batched").glob("*_themes.json")):
        x = json.loads(f.read_text(encoding="utf-8"))
        for a in x["articles"]:
            th = a.get("themes") or []
            if not th:
                continue
            cle = f"{x['fascicule']}:{a['article_id']}"
            n_etiq.append(len(th))
            for u in th:
                freq[u["code"]] += 1
                r = remonter(u["code"])
                if r:
                    toutes[r].add(cle)
            r = remonter(th[0]["code"])
            if r:
                premiere[r].add(cle)
    return freq, premiere, toutes, sum(n_etiq) / len(n_etiq)


def encodeur(par_en):
    """Rend le compte du classifieur affiné par branche, et les articles couverts."""
    run = sorted(glob.glob(str(ICI / "resultats" / "classla_*")))[-1]
    pred = json.loads(Path(run, "predictions.json").read_text(encoding="utf-8"))
    return (collections.Counter(par_en[p["etiquette"]] for p in pred),
            {p["doc_id"] for p in pred})


def figure_amortissement(D):
    """Le coût par article en fonction du nombre d'articles par appel."""
    import matplotlib.pyplot as plt

    lots = list(csv.DictReader(
        (D / "analyse_couts" / "cout_par_lot.csv").open(encoding="utf-8")))
    n = [int(l["n_articles"]) for l in lots]
    cout = [float(l["cout_sans_cache_moyen"]) / int(l["n_articles"]) for l in lots]

    # Décomposition du coût d'un appel. Le poids fixe est la liste des
    # étiquettes, transmise en entier à chaque appel. Le poids variable est le
    # texte des articles et la réponse rendue.
    fixe = sum(int(l["tokens_fixe"]) for l in lots) / len(lots)
    art = sum(int(l["tokens_articles"]) for l in lots) / sum(n)
    sortie = sum(int(l["tokens_output_moyen"]) for l in lots) / sum(n)
    A = fixe * PRIX_ENTREE / 1e6
    B = (art * PRIX_ENTREE + sortie * PRIX_SORTIE) / 1e6

    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    ax.scatter(n, cout, s=13, color=LANGUE, alpha=0.45, linewidths=0,
               label="316 appels mesurés")
    xs = [1 + i * 0.1 for i in range(241)]
    ax.plot(xs, [A / x + B for x in xs], color=NEUTRE, lw=1.1,
            label="coût attendu, poids fixe amorti")
    ax.axhline(B, color=ENCODEUR, lw=1.0, ls="--",
               label="part incompressible, le texte des articles")
    ax.set_yscale("log")
    # La notation scientifique par défaut rend « 6 x 10^-3 » là où le lecteur
    # attend un prix. Les graduations sont posées à la main, en écriture
    # décimale française.
    from matplotlib.ticker import FixedLocator, FixedFormatter
    tics = [0.0003, 0.0005, 0.001, 0.002, 0.005]
    ax.yaxis.set_major_locator(FixedLocator(tics))
    ax.yaxis.set_minor_locator(FixedLocator([]))
    ax.yaxis.set_major_formatter(FixedFormatter(
        [f"{t:.4f}".replace(".", ",") for t in tics]))
    ax.set_xlabel("articles portés par un même appel", fontsize=9)
    ax.set_ylabel("coût par article, en dollars", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.set_xlim(0, 26.5)
    ax.legend(fontsize=7.5, frameon=False, loc="upper right")
    ax.grid(True, which="major", lw=0.3, color=PALE)
    ax.set_axisbelow(True)
    for c in ("top", "right"):
        ax.spines[c].set_visible(False)
    fig.tight_layout(pad=0.5)
    fig.savefig(SORTIE / "amortissement.pdf", bbox_inches="tight")
    plt.close(fig)
    return A + B, A / 25 + B, B


def figure_couverture(D, freq, fr, racines):
    """Une case par étiquette candidate, noircie si elle est employée."""
    import matplotlib.pyplot as plt

    L = feuilles(D)
    par_branche = collections.defaultdict(list)
    for code, v in L.items():
        par_branche[v["l1_code"].replace("medtop:", "")].append(code)
    ordre = sorted(par_branche, key=lambda r: -len(par_branche[r]))

    # Deux colonnes de branches, pour que la figure tienne en hauteur. Les
    # branches sont réparties de sorte que les deux colonnes aient des hauteurs
    # voisines, la première prenant les plus fournies.
    PARLIGNE = 17
    hauteur = {r: math.ceil(len(par_branche[r]) / PARLIGNE) + 1.4 for r in ordre}
    gauche, droite, hg, hd = [], [], 0, 0
    for r in ordre:
        if hg <= hd:
            gauche.append(r); hg += hauteur[r]
        else:
            droite.append(r); hd += hauteur[r]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 0.19 * max(hg, hd) + 0.4))
    for ax, colonne in zip(axes, (gauche, droite)):
        y = 0
        for r in colonne:
            codes = sorted(par_branche[r], key=lambda c: -freq.get(c, 0))
            employees = sum(1 for c in codes if c in freq)
            for i, c in enumerate(codes):
                ax.add_patch(plt.Rectangle(
                    (i % PARLIGNE, -(y + i // PARLIGNE)), 0.82, 0.82,
                    facecolor=LANGUE if c in freq else PALE, linewidth=0))
            ax.text(-0.6, -y + 0.4, f"{fr[r].split(',')[0]}", fontsize=7.6,
                    ha="right", va="center")
            ax.text(-0.6, -y - 0.5, f"{employees}/{len(codes)}", fontsize=6.4,
                    ha="right", va="center", color=NEUTRE)
            y += hauteur[r]
        ax.set_xlim(-7.6, PARLIGNE + 0.5)
        ax.set_ylim(-max(hg, hd) + 0.5, 1.2)
        ax.set_aspect("equal")
        ax.axis("off")
    fig.tight_layout(pad=0.3)
    fig.savefig(SORTIE / "couverture_referentiel.pdf", bbox_inches="tight")
    plt.close(fig)
    return sum(1 for c in L if c not in freq), len(L)


def figure_distributions(cl, premiere, toutes, communs, fr, racines):
    """Les deux dispositifs branche par branche, sur base comparable."""
    import matplotlib.pyplot as plt

    lignes = sorted(racines, key=lambda r: len(premiere[r] & communs))
    y = range(len(lignes))
    a = [cl.get(r, 0) for r in lignes]
    b = [len(premiere[r] & communs) for r in lignes]
    c = [len(toutes[r] & communs) for r in lignes]

    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    for i, (x1, x2) in enumerate(zip(a, b)):
        ax.plot([x1, x2], [i, i], color=PALE, lw=1.6, zorder=1, solid_capstyle="round")
    ax.scatter(c, y, s=26, facecolors="none", edgecolors=LANGUE, linewidths=0.9,
               zorder=2, label="modèle de langue, toutes ses étiquettes")
    ax.scatter(b, y, s=30, color=LANGUE, linewidths=0, zorder=3,
               label="modèle de langue, première étiquette")
    ax.scatter(a, y, s=30, color=ENCODEUR, linewidths=0, zorder=3,
               label="classifieur affiné, une étiquette par article")
    ax.set_yticks(list(y))
    ax.set_yticklabels([fr[r].split(",")[0] for r in lignes], fontsize=7.5)
    ax.set_xscale("log")
    ax.set_xlabel("articles, sur les 5 361 du sous-corpus", fontsize=9)
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(True, axis="x", which="major", lw=0.3, color=PALE)
    ax.set_axisbelow(True)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left",
              bbox_to_anchor=(0.0, 1.02))
    ax.tick_params(axis="y", length=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(pad=0.5)
    fig.savefig(SORTIE / "distributions_branches.pdf", bbox_inches="tight")
    plt.close(fig)


def figure_queue(freq):
    """Les étiquettes employées par rang de fréquence."""
    import matplotlib.pyplot as plt

    v = sorted(freq.values(), reverse=True)
    rangs = range(1, len(v) + 1)
    seuil = next(i for i, x in enumerate(v, 1) if x == 1)
    tot = sum(v)
    vingt = sum(v[:20])

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.axvspan(seuil, len(v), color=PALE, lw=0, zorder=0)
    ax.plot(list(rangs), v, color=LANGUE, lw=1.3, zorder=2)
    ax.set_yscale("log")
    ax.set_xlabel("étiquettes, par rang de fréquence", fontsize=9)
    ax.set_ylabel("attributions", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.set_xlim(0, len(v) + 4)
    # Le séparateur des milliers est une espace ordinaire : les espaces fines
    # d'Unicode manquent dans la fonte Latin Modern et disparaissent au tracé.
    mille = lambda n: f"{n:,}".replace(",", " ")
    ax.annotate(f"les vingt premières portent {vingt / tot * 100:.0f} % des\n"
                f"{mille(tot)} attributions",
                xy=(20, v[19]), xytext=(52, 190), fontsize=7.5, color=NEUTRE,
                arrowprops=dict(arrowstyle="-", lw=0.6, color=NEUTRE))
    # Repère vertical plutôt qu'oblique : une flèche traversant la courbe sur
    # toute sa largeur gênerait la lecture de la décroissance.
    milieu = seuil + (len(v) - seuil) / 2
    ax.annotate(f"{len(v) - seuil + 1} étiquettes employées une fois",
                xy=(milieu, 1.25), xytext=(milieu, 9), fontsize=7.5,
                color=NEUTRE, ha="center",
                arrowprops=dict(arrowstyle="-", lw=0.6, color=NEUTRE))
    ax.grid(True, which="major", lw=0.3, color=PALE)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(pad=0.5)
    fig.savefig(SORTIE / "queue_etiquettes.pdf", bbox_inches="tight")
    plt.close(fig)
    return len(v), seuil, vingt / tot


def main():
    """Point d'entrée : construit les figures du chapitre consacré au vocabulaire
    contrôlé, depuis le référentiel et les attributions du dépôt."""
    police()
    SORTIE.mkdir(exist_ok=True)
    D = depot()
    remonter, fr, racines, par_en = taxonomie(D)
    freq, premiere, toutes, moyenne = attributions(D, remonter)
    cl, communs = encodeur(par_en)

    un, vingtcinq, plancher = figure_amortissement(D)
    print(f"amortissement.pdf          un article par appel \\${un:.5f}, "
          f"vingt-cinq \\${vingtcinq:.5f}, plancher \\${plancher:.5f} "
          f"(rapport {un / vingtcinq:.1f})")

    jamais, total = figure_couverture(D, freq, fr, racines)
    print(f"couverture_referentiel.pdf {total - jamais}/{total} étiquettes employées, "
          f"{jamais} jamais")

    figure_distributions(cl, premiere, toutes, communs, fr, racines)
    print(f"distributions_branches.pdf {moyenne:.2f} étiquettes par article "
          f"pour le modèle de langue, une pour le classifieur")

    n, seuil, part = figure_queue(freq)
    print(f"queue_etiquettes.pdf       {n} étiquettes, {n - seuil + 1} employées "
          f"une fois, les vingt premières portent {part:.1%}")


if __name__ == "__main__":
    main()
