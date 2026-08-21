"""
figures_comparaison.py : deux figures qui confrontent les chapitres entre eux

CE QUE FAIT CE SCRIPT

Produit deux figures dont l'objet n'appartient à aucun chapitre en propre, et
qui mettent en regard des mesures que le mémoire rapporte dans des tableaux
séparés.

  1. LE BALAYAGE DU NOMBRE DE GROUPES, LES DEUX FAMILLES SUR UN MÊME AXE. La
     modélisation probabiliste et le regroupement de plongements ont été
     éprouvés sur les mêmes valeurs de K, sur le même corpus. Leurs deux
     tableaux se trouvent à trente pages d'écart, et le texte affirme que les
     courbes s'opposent sans les montrer ensemble. La figure les superpose et
     fait apparaître leur point de croisement.

  2. LA DÉPENSE À L'ÉCHELLE DE LA PLATEFORME. Les trois méthodes ont été
     extrapolées aux quelque 214 millions d'articles que représente RetroNews,
     et leurs coûts ne s'expriment pas dans la même unité : un calcul local se
     compte en jours de machine, un service distant en dollars facturés. La
     figure les présente donc en deux panneaux, chacun sur une échelle
     logarithmique, plutôt que sur un axe commun qui suggérerait une
     commensurabilité qui n'existe pas.

D'OÙ VIENNENT LES VALEURS

Les NPMI sont lus dans les fichiers metrics_brut.json de chaque exécution, soit
la référence non filtrée, seule employée par le mémoire depuis la reprise des
mesures. Les durées de calcul et les coûts sont ceux que le mémoire établit,
repris ici comme constantes puisqu'ils procèdent de mesures ponctuelles plutôt
que de fichiers de résultats.

ENTRÉES

  resultats/lda_corpus_mistral_article_k{K}_g1_f5_l3_*/metrics_brut.json
  resultats/bertopic_kmeans_mistral_article_k{K}_e5_g1_*/metrics_brut.json

SORTIES

  comparaisons/balayage_k.pdf   les deux familles sur un même axe
  comparaisons/couts.pdf        la dépense à l'échelle de la plateforme

PAQUETS EMPLOYÉS

  matplotlib                             tracé des deux figures
  json, glob, argparse, pathlib          bibliothèque standard

La fonte du mémoire est chargée par la fonction du script des projections, pour
que toutes les figures de la partie emploient les mêmes caractères.

USAGE

    python3 figures_comparaison.py
"""

import argparse
import glob
import json
from pathlib import Path

ICI = Path(__file__).resolve().parent
SORTIE = ICI / "comparaisons"

# Valeurs de K éprouvées. Le regroupement de plongements a reçu une valeur de
# plus, trente-quatre, qui est le nombre de groupes que le regroupement par
# densité avait trouvé de lui-même et sur lequel la comparaison contrôlée du
# chapitre porte.
K_LDA = [6, 10, 20, 30, 40, 60]
K_BERTOPIC = [6, 10, 20, 30, 34, 40, 60]

# Extrapolation à la plateforme. Le sous-corpus classé compte 6 699 articles et
# le mémoire retient 214 millions d'articles pour RetroNews, d'après le rapport
# de 11,9 articles par page sur dix-huit millions de pages.
ARTICLES_MESURES = 6699
ARTICLES_PLATEFORME = 214_000_000

# Coût mesuré des cinq stratégies d'appel sur le sous-corpus, en dollars, tel
# que le chapitre 3 l'établit avec le tokeniseur du modèle. Les valeurs se
# régénèrent par les trois scripts d'analyse_couts/ dans le dépôt : elles sont
# reprises ici en constantes plutôt que relues, ces scripts écrivant des CSV
# dont le format ne leur est pas commun.
STRATEGIES = [
    ("Un article par appel", 31.90),
    ("Un article par appel, avec cache", 17.02),
    ("Cascade en deux étages", 4.57),
    ("Groupage par lot", 3.63),
    ("Groupage et cache", 2.93),
]

# Durées de calcul local extrapolées à la plateforme, en jours, mesurées sur la
# machine de travail. Elles changent avec le matériel, et leurs rapports non.
DUREES = [
    ("Modélisation probabiliste", 51),
    ("Plongements camembert, 128 jetons", 52),
    ("Plongements e5, 512 jetons", 432),
]

BLEU = "#1f4e8c"
BRUN = "#b03a2e"
PALE = "#d9d9d9"


def police():
    """Charge la fonte du mémoire, par la fonction du script des projections."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "projections_clusters", ICI / "projections_clusters.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.police()


def npmi(motif):
    """Rend le NPMI moyen d'une exécution, lu sur la référence non filtrée.

    Le fichier metrics_brut.json porte la valeur que le mémoire retient. Le
    fichier metrics_ref.json, produit sur une référence filtrée, donne des
    valeurs systématiquement plus basses et n'est plus employé.
    """
    d = sorted(glob.glob(str(ICI / "resultats" / motif)))
    d = [x for x in d if not any(v in x for v in ("sansumap", "_v5d", "_v15d", "_v30d", "_v50d"))]
    if not d:
        return None
    m = json.loads(Path(d[0], "metrics_brut.json").read_text(encoding="utf-8"))
    return m["npmi"]["npmi_mean"]


def balayages():
    """Rend les deux séries du balayage, lues dans les exécutions."""
    lda = [(k, npmi(f"lda_corpus_mistral_article_k{k}_g1_f5_l3_*")) for k in K_LDA]
    bt = [(k, npmi(f"bertopic_kmeans_mistral_article_k{k}_e5_g1_*")) for k in K_BERTOPIC]
    return ([(k, v) for k, v in lda if v is not None],
            [(k, v) for k, v in bt if v is not None])


def croisement(a, b):
    """Rend l'intervalle de K où les deux courbes se croisent.

    Les deux séries ne partagent pas toutes leurs abscisses. Le croisement est
    cherché sur les seules valeurs communes, entre deux points consécutifs où
    l'ordre des deux courbes s'inverse.
    """
    da, db = dict(a), dict(b)
    communs = sorted(set(da) & set(db))
    for i in range(len(communs) - 1):
        k1, k2 = communs[i], communs[i + 1]
        if (da[k1] - db[k1]) * (da[k2] - db[k2]) < 0:
            return k1, k2
    return None


def figure_balayage():
    """Trace les deux balayages sur un même axe."""
    import matplotlib.pyplot as plt

    lda, bt = balayages()
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot([k for k, _ in bt], [v for _, v in bt], "o-", color=BLEU, lw=1.4,
            ms=4.5, label="Regroupement de plongements, e5")
    ax.plot([k for k, _ in lda], [v for _, v in lda], "s-", color=BRUN, lw=1.4,
            ms=4.2, label="Modélisation probabiliste")
    c = croisement(lda, bt)
    if c:
        ax.axvspan(c[0], c[1], color=PALE, lw=0, zorder=0)
        ax.annotate(f"les deux courbes\nse croisent entre\n{c[0]} et {c[1]} groupes",
                    xy=((c[0] + c[1]) / 2, 0.2405), xytext=((c[0] + c[1]) / 2, 0.2405),
                    fontsize=7.2, ha="center", va="bottom", color="#4d4d4d")
    ax.set_xlabel("nombre de groupes demandé", fontsize=9)
    ax.set_ylabel("NPMI", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.set_xticks([6, 10, 20, 30, 40, 60])
    ax.grid(True, lw=0.3, color=PALE)
    ax.set_axisbelow(True)
    ax.legend(fontsize=7.5, frameon=False, loc="upper right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(pad=0.5)
    fig.savefig(SORTIE / "balayage_k.pdf", bbox_inches="tight")
    plt.close(fig)
    return lda, bt, c


def figure_couts():
    """Trace la dépense à l'échelle de la plateforme, en deux panneaux."""
    import matplotlib.pyplot as plt

    facteur = ARTICLES_PLATEFORME / ARTICLES_MESURES
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2),
                             gridspec_kw={"width_ratios": [1, 1.35]})

    noms = [n for n, _ in DUREES][::-1]
    vals = [v for _, v in DUREES][::-1]
    axes[0].barh(range(len(vals)), vals, color=BRUN, height=0.55)
    for i, v in enumerate(vals):
        axes[0].text(v * 1.12, i, f"{v} j", va="center", fontsize=7.5, color="#333333")
    axes[0].set_yticks(range(len(noms)))
    axes[0].set_yticklabels(noms, fontsize=7.5)
    axes[0].set_xscale("log")
    axes[0].set_xlim(10, 3000)
    axes[0].set_xlabel("jours de calcul continu", fontsize=8.5)
    axes[0].set_title("Calcul local", fontsize=9, pad=6)

    noms2 = [n for n, _ in STRATEGIES][::-1]
    vals2 = [c * facteur for _, c in STRATEGIES][::-1]
    axes[1].barh(range(len(vals2)), vals2, color=BLEU, height=0.55)
    for i, v in enumerate(vals2):
        axes[1].text(v * 1.1, i, f"{v/1000:.0f} k$", va="center", fontsize=7.5,
                     color="#333333")
    axes[1].set_yticks(range(len(noms2)))
    axes[1].set_yticklabels(noms2, fontsize=7.5)
    axes[1].set_xscale("log")
    axes[1].set_xlim(5e4, 4e6)
    axes[1].set_xlabel("dollars facturés", fontsize=8.5)
    axes[1].set_title("Service distant, classification contre le vocabulaire",
                      fontsize=9, pad=6)

    for ax in axes:
        ax.tick_params(axis="x", labelsize=7.5)
        ax.grid(True, axis="x", lw=0.3, color=PALE)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
    fig.tight_layout(pad=0.6)
    fig.savefig(SORTIE / "couts.pdf", bbox_inches="tight")
    plt.close(fig)
    return facteur, vals2


def main():
    argparse.ArgumentParser(description=__doc__.split("\n")[1]).parse_args()
    police()
    SORTIE.mkdir(exist_ok=True)

    lda, bt, c = figure_balayage()
    print("balayage_k.pdf")
    print("  probabiliste  " + "  ".join(f"K{k}={v:.3f}" for k, v in lda))
    print("  plongements   " + "  ".join(f"K{k}={v:.3f}" for k, v in bt))
    print(f"  croisement entre K={c[0]} et K={c[1]}" if c else "  aucun croisement")

    facteur, vals = figure_couts()
    print(f"\ncouts.pdf     facteur d'extrapolation {facteur:.0f}")
    for (n, _), v in zip(STRATEGIES, vals[::-1]):
        print(f"  {n:34} {v:>10,.0f} \\$".replace(",", " "))


if __name__ == "__main__":
    main()
