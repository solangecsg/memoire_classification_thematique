"""
pages_thematiques.py : une page de presse coloriée par la présence d'un thème

CE QUE FAIT CE SCRIPT

Dessine la mise en page d'un fascicule à ses coordonnées réelles, un rectangle
par bloc de texte, et remplit chaque bloc d'une intensité proportionnelle au
poids que le modèle thématique lui donne sur un thème choisi.

La figure répond à une question que les tableaux de mesures ne posent pas : où
un thème se trouve-t-il sur la page ? Elle donne à voir ce que le chapitre
premier affirme du mélange, un document se répartissant entre les thèmes plutôt
que d'appartenir à l'un d'eux, et ce que le second reproche au bloc, qui découpe
la page en rectangles sans égard pour les unités de discours.

D'OÙ VIENNENT LES DEUX SOURCES

La géométrie vient du fichier ALTO de la page, où chaque TextBlock porte ses
quatre coordonnées en pixels, dans le repère de l'image numérisée. Le poids
thématique vient de l'exécution du modèle à la granularité du bloc, dont le
fichier span_topic conserve la distribution complète sur les vingt thèmes.

Les deux se joignent par l'identifiant du bloc, de la forme PAG_1_TB000015, que
l'ALTO écrit dans son attribut ID et que le modèle reprend tel quel.

TROIS ÉTATS POUR UN BLOC

Un bloc de la page se trouve dans l'un de trois états, que la figure distingue.

  - Modélisé et chargé du thème : le remplissage suit le poids.
  - Modélisé et sans rapport avec lui : le remplissage reste très clair.
  - Jamais modélisé : le bloc est laissé blanc et sa bordure est tiretée. Ce
    sont les titres, écartés du corpus, et les blocs trop courts pour atteindre
    le seuil de cinq mots. Les montrer importe : ils occupent une part
    considérable d'une page de presse, et une figure qui les tairait donnerait
    une idée fausse de ce que la modélisation couvre.

ENTRÉES

  ../../resultats_mistral/{fascicule}_reocr/ocr/{page}.xml
  resultats/lda_corpus_mistral_bloc_k20_g1_f5_l3_*/span_topic.json
  resultats/lda_corpus_mistral_bloc_k20_g1_f5_l3_*/metrics_brut.json

SORTIES

  pages/pages_thematiques.pdf   figure à trois panneaux, un thème par page
  pages/pages_iptc.pdf          figure à deux panneaux, une couleur par catégorie
  Le rattachement des blocs aux articles du METS, avec --verifier.
  Le relevé des pages les plus chargées de chaque thème, avec --chercher.

PAQUETS EMPLOYÉS

  matplotlib                        tracé des rectangles
  json, re, glob, collections, argparse, pathlib   bibliothèque standard

L'ALTO est lu par expression régulière plutôt que par un analyseur XML. Seul
l'élément TextBlock est cherché, avec ses quatre attributs de position, et cette
lecture évite de construire l'arbre complet d'un fichier de quatre cent
kilooctets dont on n'emploie que cent lignes.

USAGE

    python3 pages_thematiques.py
    python3 pages_thematiques.py --chercher 15    # pages chargées du thème 15
"""

import argparse
import collections
import glob
import json
import re
from pathlib import Path

ICI = Path(__file__).resolve().parent
SORTIE = ICI / "pages"

# Les trois panneaux de la figure. Chaque entrée donne le fascicule, la page,
# le thème et l'intitulé lu sur ses mots de tête. Les pages sont choisies pour
# opposer une page thématiquement homogène à des pages où le thème n'occupe
# qu'une région, opposition dont le mémoire tire son argument sur la
# granularité.
PANNEAUX = [
    ("4791550", "X0000004", 0, "Bourse et cotes", "1898"),
    ("4715618", "X0000002", 3, "Guerre et diplomatie", "1877"),
    ("4610674", "X0000005", 17, "Sport", "1934"),
]

# Seconde figure : les mêmes pages classées contre le vocabulaire contrôlé. La
# première reprend la page de 1877 du panneau central ci-dessus, ce qui met les
# deux dispositifs en regard sur une géométrie identique. La seconde est la page
# du corpus qui porte le plus de catégories distinctes.
PANNEAUX_IPTC = [
    ("4715618", "X0000002", "2", "Le Petit Marseillais, 1877"),
    ("4135298", "X0000004", "4", "La Dépêche, 1903"),
]

# Nombre de blocs à partir duquel l'intitulé de l'article est écrit sur la page.
# Au-dessous, l'article est trop petit pour porter un texte lisible.
BLOCS_INTITULE = 5

RUN = "lda_corpus_mistral_bloc_k20_g1_f5_l3_*"

# Bornes de l'échelle de remplissage. Le poids d'un bloc sur un thème dépasse
# rarement 0,6 sur ce corpus, un bloc court se répartissant toujours un peu
# entre les vingt thèmes. Plafonner à cette valeur emploie toute l'étendue de la
# palette plutôt que d'en laisser la moitié inutilisée.
POIDS_MAX = 0.6

TIRETS = "#9a9a9a"


def corpus() -> Path:
    """Racine du dossier de travail, cherchée en remontant l'arborescence."""
    for base in [ICI, *ICI.parents]:
        if (base / "resultats_mistral").is_dir():
            return base
    raise SystemExit("dossier resultats_mistral introuvable")


def run() -> Path:
    """Dossier de l'exécution du modèle à la granularité du bloc."""
    d = sorted(glob.glob(str(ICI / "resultats" / RUN)))
    if not d:
        raise SystemExit(f"exécution introuvable : resultats/{RUN}")
    return Path(d[0])


def blocs_alto(fascicule: str, page: str):
    """Rend la géométrie de chaque bloc de la page, et le format de la page.

    Les coordonnées sont en pixels de l'image numérisée, l'origine étant le coin
    supérieur gauche. L'axe vertical descend, contrairement à celui d'un
    graphique, et le tracé en tiendra compte.
    """
    f = corpus() / "resultats_mistral" / f"{fascicule}_reocr" / "ocr" / f"{page}.xml"
    if not f.is_file():
        raise SystemExit(f"page introuvable : {f}")
    t = f.read_text(encoding="utf-8", errors="ignore")
    dim = re.search(r"<[a-z0-9]*:?Page[^>]*HEIGHT=\"(\d+)\"[^>]*WIDTH=\"(\d+)\"", t)
    hauteur, largeur = (int(dim.group(1)), int(dim.group(2))) if dim else (0, 0)
    out = {}
    for m in re.finditer(r"<[a-z0-9]*:?TextBlock([^>]*)>", t):
        a = m.group(1)
        val = dict(re.findall(r'(\w+)="([^"]*)"', a))
        if {"HPOS", "VPOS", "WIDTH", "HEIGHT", "ID"} <= set(val):
            out[val["ID"]] = (int(val["HPOS"]), int(val["VPOS"]),
                              int(val["WIDTH"]), int(val["HEIGHT"]))
    return out, largeur, hauteur


def poids():
    """Rend, pour chaque bloc modélisé, sa distribution sur les vingt thèmes."""
    d = json.loads((run() / "span_topic.json").read_text(encoding="utf-8"))
    if isinstance(d, dict):
        d = d.get("spans", list(d.values()))
    return {(x["fascicule"], x["page"], x["unite"]): x["dist"]
            for x in d if isinstance(x, dict) and x.get("dist")}


def intitules():
    """Rend les mots de tête de chaque thème, pour vérifier un intitulé."""
    d = json.loads((run() / "metrics_brut.json").read_text(encoding="utf-8"))
    return {t["topic_id"]: t["top_words"] for t in d["npmi"]["npmi_per_topic"]}


def chercher(theme: int, combien: int = 12):
    """Imprime les pages où le thème pèse le plus, pour choisir un panneau."""
    par_page = collections.defaultdict(list)
    for (f, p, _), dist in poids().items():
        par_page[(f, p)].append(dist[theme])
    lignes = sorted(par_page.items(), key=lambda kv: -sum(kv[1]) / len(kv[1]))
    print(f"thème {theme} : {' '.join(intitules()[theme].split()[:9])}\n")
    for (f, p), v in lignes[:combien]:
        if len(v) >= 20:
            print(f"  {f} {p}  {len(v):4} blocs modélisés, "
                  f"poids moyen {sum(v) / len(v):.2f}, maximum {max(v):.2f}")


def articles(fascicule):
    """Rend, pour chaque bloc du fascicule, l'article auquel le METS le rattache.

    La carte logique déclare ses articles par des divisions typées ARTICLE, dont
    les descendantes portent les renvois aux blocs. Le rattachement se lit en
    parcourant le texte d'une division ARTICLE à la suivante et en relevant les
    identifiants de bloc rencontrés, procédé qui suit l'ordre du document sans
    construire son arbre.
    """
    f = corpus() / "resultats_mistral" / f"{fascicule}_reocr" / "toc" / f"T{fascicule}.xml"
    if not f.is_file():
        return {}
    t = f.read_text(encoding="utf-8", errors="ignore")
    bornes = [(m.start(), m.group(1), m.group(2)) for m in re.finditer(
        r'<div ID="([^"]+)" [^>]*TYPE="ARTICLE"[^>]*LABEL="([^"]*)"', t)]
    out = {}
    for i, (pos, aid, lab) in enumerate(bornes):
        fin = bornes[i + 1][0] if i + 1 < len(bornes) else len(t)
        for b in re.findall(r'BEGIN="([A-Z0-9_]+)"', t[pos:fin]):
            out[b] = (aid, lab)
    return out


def verifier(seuil=0.30):
    """Confronte l'étendue du thème sur la page au découpage en articles du METS.

    La figure montre où le thème se pose ; ce contrôle dit à quoi cet endroit
    correspond dans la description reçue. Il rapporte les articles auxquels
    appartiennent les blocs chargés du thème, avec leur intitulé, ce qui permet
    de juger si le modèle a retrouvé une unité éditoriale ou dispersé sa masse.
    """
    p = poids()
    for fascicule, page, theme, nom, an in PANNEAUX:
        art = articles(fascicule)
        sur = [(b, d) for (f, pg, b), d in p.items()
               if f == fascicule and pg == page]
        forts = [b for b, d in sur if d[theme] >= seuil]
        c = collections.Counter(art.get(b, ("hors article", "")) for b in forts)
        vus = {art[b] for b, _ in sur if b in art}
        print(f"\n{nom}, {an} ({fascicule} {page})")
        print(f"  {len(forts)} blocs sur {len(sur)} au-dessus de {seuil:.2f}, "
              f"{len(vus)} articles sur la page")
        for (aid, lab), n in c.most_common(6):
            print(f"    {n:2} blocs  {lab[:56]}")


def iptc():
    """Rend l'étiquette de chaque article, et la remontée vers sa catégorie.

    La classification contre le vocabulaire contrôlé travaille au niveau de
    l'article et rend jusqu'à cinq étiquettes de troisième niveau. La première
    est retenue pour la couleur, ramenée à sa catégorie de tête par la relation
    broader, et son intitulé fin sert à écrire l'article sur la page.
    """
    d = depot_iptc()
    t = json.loads((d / "classification" / "iptc_mediatopic_official.json")
                   .read_text(encoding="utf-8"))
    conc = {c["qcode"].removeprefix("medtop:"): c for c in t["conceptSet"]}
    racines = {u.rsplit("/", 1)[-1] for u in t["hasTopConcept"]}
    fr = {k: (c["prefLabel"].get("fr") or c["prefLabel"].get("en-GB"))
          for k, c in conc.items()}

    def remonter(code):
        vu = set()
        while code not in racines and code not in vu:
            vu.add(code)
            b = conc.get(code, {}).get("broader")
            if not b:
                return None
            code = b[0].rsplit("/", 1)[-1]
        return code if code in racines else None

    par_article = {}
    for f in sorted((d / "results" / "feuilles_mistral_batched").glob("*_themes.json")):
        x = json.loads(f.read_text(encoding="utf-8"))
        for a in x["articles"]:
            if a.get("themes"):
                c = a["themes"][0]
                par_article[(x["fascicule"], a["article_id"])] = (
                    remonter(c["code"]), c["label_fr"])
    return par_article, fr


def depot_iptc() -> Path:
    """Racine du dépôt de classification, cherchée en remontant l'arborescence."""
    for base in [ICI, *ICI.parents]:
        for rel in (Path("github") / "classification-iptc", Path(".")):
            if (base / rel / "classification" / "iptc_mediatopic_official.json").is_file():
                return (base / rel).resolve()
    raise SystemExit("dépôt de classification introuvable")


def panneau_iptc(ax, fascicule, page, numero, titre, couleurs, par_article, fr):
    """Trace une page dont chaque article reçoit la couleur de sa catégorie."""
    import matplotlib.pyplot as plt

    geo, largeur, hauteur = blocs_alto(fascicule, page)
    art = articles(fascicule)
    par_art = collections.defaultdict(list)
    for bid, boite in geo.items():
        a = art.get(bid)
        if a and (fascicule, a[0]) in par_article:
            par_art[a[0]].append((bid, boite))

    dessines = set()
    for aid, blocs in par_art.items():
        racine, fin = par_article[(fascicule, aid)]
        c = couleurs.get(racine, "#dddddd")
        for bid, (x, y, w, h) in blocs:
            ax.add_patch(plt.Rectangle((x, -y - h), w, h, facecolor=c,
                                       edgecolor="#4d4d4d", linewidth=0.2,
                                       alpha=0.85))
            dessines.add(bid)
        if len(blocs) >= BLOCS_INTITULE:
            cx = sum(b[1][0] + b[1][2] / 2 for b in blocs) / len(blocs)
            cy = sum(-b[1][1] - b[1][3] / 2 for b in blocs) / len(blocs)
            ax.text(cx, cy, fin, fontsize=5.2, ha="center", va="center",
                    color="#111111", bbox=dict(boxstyle="round,pad=0.14",
                                               fc="white", ec="none", alpha=0.82))
    for bid, (x, y, w, h) in geo.items():
        if bid not in dessines:
            ax.add_patch(plt.Rectangle((x, -y - h), w, h, facecolor="none",
                                       edgecolor=TIRETS, linewidth=0.3,
                                       linestyle=(0, (1.6, 1.6))))
    ax.set_xlim(0, largeur)
    ax.set_ylim(-hauteur, 0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for c in ax.spines.values():
        c.set_color("#999999")
        c.set_linewidth(0.5)
    ax.set_title(f"{titre}\n{len(par_art)} articles classés, "
                 f"{len(geo) - len(dessines)} blocs hors article",
                 fontsize=8, pad=5)
    return {par_article[(fascicule, a)][0] for a in par_art}


def figure_iptc():
    """Les mêmes pages, classées contre le vocabulaire contrôlé."""
    import matplotlib.pyplot as plt
    from matplotlib import colormaps

    importlib_police()
    par_article, fr = iptc()
    presentes = []
    for fascicule, page, numero, _t in PANNEAUX_IPTC:
        art = articles(fascicule)
        for bid in blocs_alto(fascicule, page)[0]:
            a = art.get(bid)
            if a and (fascicule, a[0]) in par_article:
                r = par_article[(fascicule, a[0])][0]
                if r and r not in presentes:
                    presentes.append(r)
    palette = list(colormaps["tab20"].colors)
    couleurs = {r: palette[i % len(palette)] for i, r in enumerate(presentes)}

    fig, axes = plt.subplots(1, len(PANNEAUX_IPTC), figsize=(7.2, 5.6))
    for ax, (f, p, n, t) in zip(axes, PANNEAUX_IPTC):
        panneau_iptc(ax, f, p, n, t, couleurs, par_article, fr)
    for r in presentes:
        axes[-1].add_patch(plt.Rectangle((0, 0), 0, 0, facecolor=couleurs[r],
                                         label=fr[r].split(",")[0]))
    axes[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=6.6,
                    frameon=False, labelspacing=0.35, handlelength=1.1)
    fig.tight_layout(pad=0.5)
    fig.savefig(SORTIE / "pages_iptc.pdf", bbox_inches="tight")
    plt.close(fig)
    return len(presentes)


def panneau(ax, fascicule, page, theme, titre, carte):
    """Trace une page, un rectangle par bloc, rempli selon le poids du thème."""
    import matplotlib.pyplot as plt

    geo, largeur, hauteur = blocs_alto(fascicule, page)
    p = poids()
    modelises = sans = 0
    for bid, (x, y, w, h) in geo.items():
        d = p.get((fascicule, page, bid))
        if d is None:
            # Bloc jamais modélisé : titre écarté du corpus, ou bloc trop court.
            ax.add_patch(plt.Rectangle((x, -y - h), w, h, facecolor="none",
                                       edgecolor=TIRETS, linewidth=0.3,
                                       linestyle=(0, (1.6, 1.6))))
            sans += 1
            continue
        modelises += 1
        v = 0.10 + 0.90 * min(1.0, d[theme] / POIDS_MAX)
        ax.add_patch(plt.Rectangle((x, -y - h), w, h, facecolor=carte(v),
                                   edgecolor="#6b6b6b", linewidth=0.18))
    ax.set_xlim(0, largeur)
    ax.set_ylim(-hauteur, 0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for c in ax.spines.values():
        c.set_color("#999999")
        c.set_linewidth(0.5)
    ax.set_title(f"{titre}\n{modelises} blocs modélisés, {sans} écartés",
                 fontsize=8, pad=5)


def figure():
    """Assemble les trois panneaux et la barre de couleur commune."""
    import matplotlib.pyplot as plt
    from matplotlib import colormaps
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    spec = importlib_police()
    carte = colormaps["YlOrRd"]
    fig, axes = plt.subplots(1, len(PANNEAUX), figsize=(7.2, 5.4))
    for ax, (f, p, t, nom, an) in zip(axes, PANNEAUX):
        panneau(ax, f, p, t, f"{nom}, {an}", carte)
    barre = fig.colorbar(ScalarMappable(norm=Normalize(0, POIDS_MAX), cmap=carte),
                         ax=axes, fraction=0.020, pad=0.02)
    barre.set_label("poids du bloc sur le thème", fontsize=8)
    barre.ax.tick_params(labelsize=7)
    barre.outline.set_visible(False)
    SORTIE.mkdir(exist_ok=True)
    fig.savefig(SORTIE / "pages_thematiques.pdf", bbox_inches="tight")
    plt.close(fig)
    return spec


def importlib_police():
    """Charge la fonte du mémoire, par la fonction du script des projections."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "projections_clusters", ICI / "projections_clusters.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.police()
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--chercher", type=int, metavar="THÈME",
                   help="lister les pages où ce thème pèse le plus")
    p.add_argument("--verifier", action="store_true",
                   help="confronter l'étendue du thème au découpage en articles")
    args = p.parse_args()
    if args.chercher is not None:
        chercher(args.chercher)
        return
    if args.verifier:
        verifier()
        return
    figure()
    n = figure_iptc()
    print(f"pages_iptc.pdf         {n} catégories de premier niveau représentées")
    mots = intitules()
    for f, p, t, nom, _an in PANNEAUX:
        geo, _, _ = blocs_alto(f, p)
        print(f"{nom:24} {f} {p}  {len(geo)} blocs sur la page  "
              f"thème {t} : {' '.join(mots[t].split()[:6])}")
    print(f"\npages_thematiques.pdf écrit dans {SORTIE.name}/")


if __name__ == "__main__":
    main()
