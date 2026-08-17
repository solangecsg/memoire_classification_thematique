"""
projections_clusters.py : nuages de points des regroupements

CE QUE FAIT CE SCRIPT

Projette les 5 361 articles sur un plan et les colorie de trois manières, pour
donner à voir ce que les tableaux de mesures ne montrent pas : la forme des
groupes et leur position les uns par rapport aux autres.

  1. LES DEUX REGROUPEMENTS CÔTE À CÔTE. Une figure à deux panneaux, même
     projection des deux côtés, même nombre de groupes, trente-quatre. À gauche
     le partitionnement en k moyennes, qui affecte chaque article. À droite le
     regroupement par densité, qui laisse en gris les 1 733 articles qu'aucune
     zone dense ne réclame. La figure montre où se trouve la matière rejetée.

  2. LES CATÉGORIES DU VOCABULAIRE CONTRÔLÉ SUR LE MÊME PLAN. Les mêmes
     articles, coloriés cette fois par la catégorie de premier niveau que leur
     attribue la classification contre le vocabulaire contrôlé. La question
     posée est celle de l'accord entre une géométrie construite sans consigne
     et un découpage documentaire construit sans elle.

TROIS STATISTIQUES ACCOMPAGNENT LES FIGURES

Une image suggère, elle ne mesure pas. Trois quantités sont donc calculées dans
l'espace des plongements complet, et non sur la projection :

  - la distance moyenne aux quinze plus proches voisins, comparée entre les
    articles retenus par le regroupement par densité et ceux qu'il écarte. Elle
    dit si la matière rejetée se tient à l'écart des autres ;
  - le nombre de groupes distincts parmi ces quinze voisins, comparé entre les
    mêmes deux ensembles. Il dit si la matière rejetée se tient là où plusieurs
    matières se rencontrent. Les deux quantités ne répondent pas à la même
    question et leurs réponses diffèrent ;
  - la pureté du voisinage des catégories du vocabulaire contrôlé, soit la part
    des dix plus proches voisins d'un article qui portent sa catégorie,
    rapportée au niveau qu'atteindrait un étiquetage sans rapport avec la
    géométrie.

POURQUOI UNE PROJECTION SÉPARÉE DE CELLE DU REGROUPEMENT

La chaîne de regroupement réduit les plongements à cinq dimensions avant de
former les groupes. Cinq dimensions ne se dessinent pas. Une seconde réduction
est donc conduite vers deux dimensions, avec le même voisinage et la même
graine, à seule fin d'affichage. Les groupes dessinés sont bien ceux qu'a formés
la chaîne, mais leurs positions sur le plan ne sont pas celles où elle les a
trouvés. Une projection déforme toujours : deux points voisins sur la figure le
sont dans l'espace d'origine, deux points éloignés ne le sont pas
nécessairement.

ENTRÉES

  embeddings/mistral_article_e5_5361.npy        plongements des articles
  embeddings/mistral_article_e5_5361.ids.json   identifiants, dans le même ordre
  resultats/bertopic_kmeans_mistral_article_k34_e5_g1_*/span_topic.json
  resultats/bertopic_hdbscan_mistral_article_mt20_e5_g1_*/span_topic.json
  ../../../../github/classification-iptc/results/feuilles_mistral_batched/
  ../../../../github/classification-iptc/classification/iptc_mediatopic_official.json

SORTIES

  projections/umap2d_mistral_article_e5.npy   projection mise en cache
  projections/regroupements.pdf               figure à deux panneaux
  projections/categories_iptc.pdf             figure à un panneau

PAQUETS EMPLOYÉS

  numpy                     tableaux de plongements et calculs de distances
  scipy                     test de Welch sur les voisinages
  umap-learn                réduction de dimension vers le plan
  scikit-learn              recherche des plus proches voisins
  matplotlib                tracé des nuages de points
  json, glob, collections, pathlib   bibliothèque standard

Les deux figures sont enregistrées avec un cadrage serré. La légende de la
seconde est posée hors du cadre des axes, et la mise en page automatique de
matplotlib ne la compte pas dans ses marges : sans ce cadrage, les intitulés les
plus longs se trouvent rognés.

Les figures sont écrites en PDF, format vectoriel : les points restent nets à
toute échelle d'impression, contrairement à une image en pixels. Les textes
emploient la fonte du mémoire, chargée depuis la distribution TeX.

USAGE

    python3 projections_clusters.py

La première exécution calcule la projection et prend environ une minute. Les
suivantes la relisent depuis le cache. L'option --recalculer force le calcul.
"""

import argparse
import collections
import glob
import json
from pathlib import Path

import numpy as np

ICI = Path(__file__).resolve().parent
SORTIE = ICI / "projections"

# Réglages de la projection d'affichage. Le voisinage et la graine reprennent
# ceux de la chaîne de regroupement, pour que la figure montre l'espace à la
# même échelle que celle où les groupes ont été formés. La distance minimale
# est le seul écart : la chaîne emploie zéro, qui tasse les points les uns sur
# les autres et convient à un algorithme de densité, alors qu'une figure
# demande que les points se distinguent à l'œil.
VOISINS = 15
DISTANCE_MIN = 0.10
GRAINE = 1

# Nombre de voisins retenus pour les statistiques. Quinze pour la caractérisation
# des voisinages, par cohérence avec le voisinage de la projection ; dix pour la
# pureté des catégories, valeur usuelle qui reste petite devant l'effectif de la
# plus rare des catégories retenues, trente-quatre articles.
VOISINS_DENSITE = 15
VOISINS_PURETE = 10

GRIS = "#c8c8c8"


def police():
    """Charge la fonte du mémoire dans matplotlib.

    Les figures d'un mémoire composé en Latin Modern jurent lorsqu'elles
    emploient la fonte par défaut d'une bibliothèque de tracé. Les fichiers
    OpenType de la distribution TeX sont donc déclarés directement. Si la
    distribution est absente, le tracé se poursuit avec une fonte à empattements
    quelconque.
    """
    import matplotlib
    from matplotlib import font_manager

    racine = Path("/usr/local/texlive")
    fichiers = sorted(
        p for p in racine.glob("*/texmf-dist/fonts/opentype/public/lm/lmroman10-*.otf")
        if p.stem.endswith(("regular", "italic")))
    for f in fichiers:
        font_manager.fontManager.addfont(str(f))
    familles = ["Latin Modern Roman"] if fichiers else []
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": familles + ["DejaVu Serif"],
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
    })


def plongements(nom="mistral_article_e5_5361"):
    """Rend la matrice des plongements et la liste des identifiants.

    Les deux fichiers sont écrits ensemble par la chaîne de regroupement et
    partagent leur ordre : la ligne i de la matrice décrit l'article i de la
    liste.

    Le cache de plongements ne figure pas dans le dépôt, pesant plusieurs
    centaines de mégaoctets. Son absence est signalée avec la commande qui le
    reconstitue, plutôt que laissée remonter sous forme de trace d'exécution.
    """
    if not (ICI / "embeddings" / f"{nom}.npy").is_file():
        raise SystemExit(
            f"cache de plongements absent : embeddings/{nom}.npy\n"
            "Il n'est pas versé dans le dépôt, étant volumineux et "
            "régénérable. Le reconstituer par :\n"
            "    python3 bertopic_corpus.py --regroupement kmeans --k 20 "
            "--modele e5 --graine 1")
    X = np.load(ICI / "embeddings" / f"{nom}.npy")
    ids = json.loads((ICI / "embeddings" / f"{nom}.ids.json")
                     .read_text(encoding="utf-8"))
    if isinstance(ids, dict):
        ids = ids.get("ids", list(ids.values()))
    return X, list(ids)


def projeter(X, recalculer=False):
    """Réduit les plongements à deux dimensions, avec mise en cache.

    Le calcul dure environ une minute et son résultat ne dépend que des
    plongements et de la graine. Il est donc conservé sur disque, ce qui rend
    les exécutions suivantes immédiates et garantit que les deux figures
    reposent sur exactement la même projection.
    """
    cache = SORTIE / "umap2d_mistral_article_e5.npy"
    if cache.is_file() and not recalculer:
        return np.load(cache)
    from umap import UMAP
    # La distance cosinus est celle pour laquelle le modèle de plongement a été
    # entraîné : deux articles proches de sens ont des vecteurs de même
    # direction, indépendamment de leur norme.
    xy = UMAP(n_neighbors=VOISINS, n_components=2, min_dist=DISTANCE_MIN,
              metric="cosine", random_state=GRAINE).fit_transform(X)
    SORTIE.mkdir(exist_ok=True)
    np.save(cache, xy)
    return xy


def affectations(motif):
    """Rend l'affectation de chaque article à un groupe, pour une exécution.

    Le fichier span_topic consigne, pour chaque document, le groupe qui lui a
    été attribué. La valeur -1 signale un article qu'aucun groupe ne réclame,
    convention du regroupement par densité.
    """
    run = sorted(glob.glob(str(ICI / "resultats" / motif)))[0]
    d = json.loads(Path(run, "span_topic.json").read_text(encoding="utf-8"))
    if isinstance(d, dict):
        d = d.get("spans", list(d.values()))
    out = {}
    for x in d:
        if not isinstance(x, dict):
            continue
        k = x.get("doc_id") or x.get("span_id") or x.get("id")
        t = x.get("topic") if x.get("topic") is not None else x.get("topic_id")
        if k is not None and t is not None:
            out[k] = int(t)
    return out


def depot():
    """Racine du dépôt de classification, cherchée en remontant l'arborescence."""
    for base in [ICI, *ICI.parents]:
        for rel in (Path("github") / "classification-iptc", Path(".")):
            if (base / rel / "classification" / "iptc_mediatopic_official.json").is_file():
                return (base / rel).resolve()
    raise SystemExit("dépôt de classification introuvable")


def categories_iptc():
    """Rend la catégorie de premier niveau attribuée à chaque article.

    La classification contre le vocabulaire contrôlé travaille au troisième
    niveau, trop fin pour une figure. Chaque étiquette est donc remontée jusqu'à
    sa racine par la relation broader, ce qui ramène les 567 intitulés possibles
    aux 17 catégories de tête. Seule la première étiquette de chaque article est
    retenue, celle que le modèle a placée en tête.
    """
    D = depot()
    t = json.loads((D / "classification" / "iptc_mediatopic_official.json")
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

    out = {}
    for f in sorted((D / "results" / "feuilles_mistral_batched").glob("*_themes.json")):
        x = json.loads(f.read_text(encoding="utf-8"))
        for a in x["articles"]:
            if a.get("themes"):
                r = remonter(a["themes"][0]["code"])
                if r:
                    out[f"{x['fascicule']}:{a['article_id']}"] = r
    return out, fr


def palette(n):
    """Rend n couleurs distinctes.

    Les palettes qualitatives de matplotlib s'arrêtent à vingt teintes. Deux
    d'entre elles sont mises bout à bout pour couvrir trente-quatre groupes. Au
    delà d'une vingtaine de couleurs l'œil ne les distingue plus une à une, et
    ce n'est pas ce qu'on lui demande ici : la couleur sert à séparer les
    régions voisines, pas à identifier un groupe précis.
    """
    from matplotlib import colormaps
    couleurs = list(colormaps["tab20"].colors) + list(colormaps["tab20b"].colors)
    return [couleurs[i % len(couleurs)] for i in range(n)]


def apparier(a, b, ids):
    """Fait correspondre les groupes de deux partitions par recouvrement.

    Les numéros que deux algorithmes donnent à leurs groupes sont arbitraires et
    sans rapport entre eux. Colorier les deux panneaux au hasard rendrait leur
    comparaison impossible. Chaque groupe de la seconde partition reçoit donc la
    couleur du groupe de la première avec lequel il partage le plus d'articles,
    sans qu'une couleur serve deux fois.
    """
    M = collections.Counter((a[k], b[k]) for k in ids
                            if k in a and k in b and a[k] >= 0 and b[k] >= 0)
    gauche = sorted({x for x, _ in M},
                    key=lambda x: -sum(v for (i, _), v in M.items() if i == x))
    libres = list(gauche)
    corr = {}
    for d in sorted({y for _, y in M},
                    key=lambda y: -sum(v for (_, j), v in M.items() if j == y)):
        if not libres:
            corr[d] = gauche[len(corr) % len(gauche)]
            continue
        meilleur = max(libres, key=lambda g: M.get((g, d), 0))
        corr[d] = meilleur
        libres.remove(meilleur)
    return {g: i for i, g in enumerate(gauche)}, corr


def nuage(ax, xy, groupes, rang, couleurs, titre, taille=3.2):
    """Trace un panneau : un point par article, une couleur par groupe.

    Les articles sans groupe sont tracés en premier et en gris clair, pour que
    les groupes se détachent au-dessus d'eux plutôt que d'être recouverts.
    """
    sans = np.array([g < 0 for g in groupes])
    if sans.any():
        ax.scatter(xy[sans, 0], xy[sans, 1], s=taille, c=GRIS,
                   linewidths=0, alpha=0.75, rasterized=True)
    for g in sorted({g for g in groupes if g >= 0}):
        m = np.array([x == g for x in groupes])
        ax.scatter(xy[m, 0], xy[m, 1], s=taille, color=couleurs[rang[g]],
                   linewidths=0, alpha=0.80, rasterized=True)
    # Les deux axes portent la même unité, une distance dans l'espace réduit.
    # Les étirer indépendamment donnerait à voir des rapprochements que la
    # projection n'a pas produits.
    ax.set_aspect("equal")
    if titre:
        ax.set_title(titre, fontsize=9, pad=6)
    ax.set_xticks([])
    ax.set_yticks([])
    for c in ax.spines.values():
        c.set_color("#999999")


def figure_regroupements(xy, ids, recalculer=False):
    """Les deux algorithmes sur la même projection, à trente-quatre groupes."""
    import matplotlib.pyplot as plt

    km = affectations("bertopic_kmeans_mistral_article_k34_e5_g1_*")
    hd = affectations("bertopic_hdbscan_mistral_article_mt20_e5_g1_*")
    gk = [km.get(i, -1) for i in ids]
    gh = [hd.get(i, -1) for i in ids]

    rang_km, corr = apparier(km, hd, ids)
    couleurs = palette(len(rang_km))
    rang_hd = {d: rang_km[g] for d, g in corr.items()}

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.0))
    nuage(axes[0], xy, gk, rang_km, couleurs,
          "Partitionnement en $k$ moyennes, $K = 34$")
    nuage(axes[1], xy, gh, rang_hd, couleurs,
          "Regroupement par densité, 34 groupes trouvés")
    sans = sum(1 for g in gh if g < 0)
    axes[1].scatter([], [], s=12, c=GRIS, linewidths=0,
                    label=f"sans groupe ({sans // 1000}\u202f{sans % 1000:03d} articles)")
    axes[1].legend(loc="lower right", fontsize=7.5, frameon=False,
                   handletextpad=0.4, borderpad=0.2)
    fig.tight_layout(pad=0.6)
    fig.savefig(SORTIE / "regroupements.pdf", dpi=400, bbox_inches="tight")
    plt.close(fig)
    return np.array(gh), np.array(gk)


def figure_iptc(xy, ids):
    """Les catégories du vocabulaire contrôlé sur la même projection."""
    import matplotlib.pyplot as plt

    cats, fr = categories_iptc()
    g = [cats.get(i) for i in ids]
    freq = collections.Counter(c for c in g if c)
    ordre = [c for c, _ in freq.most_common()]
    rang = {c: i for i, c in enumerate(ordre)}
    couleurs = palette(len(ordre))
    codes = [rang[c] if c else -1 for c in g]

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    nuage(ax, xy, codes, {i: i for i in range(len(ordre))}, couleurs,
          "", taille=3.6)
    for i, c in enumerate(ordre):
        ax.scatter([], [], s=14, color=couleurs[i], linewidths=0,
                   label=f"{fr[c].split(',')[0]} ({freq[c]})")
    ax.legend(loc="center left", bbox_to_anchor=(1.005, 0.5), fontsize=7.5,
              frameon=False, handletextpad=0.4, labelspacing=0.32)
    fig.tight_layout(pad=0.6)
    fig.savefig(SORTIE / "categories_iptc.pdf", dpi=400, bbox_inches="tight")
    plt.close(fig)
    return g


def voisinage(X, gh, gk):
    """Caractérise le voisinage des articles retenus et des articles écartés.

    Deux quantités sont tirées des quinze plus proches voisins de chaque
    article, dans l'espace des plongements complet où le regroupement a eu lieu
    et non sur la projection, qui déforme les distances.

    La distance moyenne à ces voisins dit si l'article se tient à l'écart. Le
    nombre de groupes distincts qu'ils occupent dit s'il se tient là où
    plusieurs matières se rencontrent. La partition employée pour ce second
    compte est celle du partitionnement en k moyennes, qui affecte tous les
    articles : celle du regroupement par densité laisserait les articles écartés
    sans groupe et rendrait le compte impossible.

    Le test de Welch accompagne le second écart. Il ne suppose pas que les deux
    ensembles aient même variance, ce qui n'a pas de raison d'être vrai ici.
    """
    from scipy import stats
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=VOISINS_DENSITE + 1, metric="cosine").fit(X)
    d, v = nn.kneighbors(X)
    dist = d[:, 1:].mean(axis=1)
    melange = np.array([len(set(gk[r])) for r in v[:, 1:]])
    ret = gh >= 0
    t = stats.ttest_ind(melange[ret], melange[~ret], equal_var=False)
    return (dist[ret].mean(), dist[~ret].mean(),
            melange[ret].mean(), melange[~ret].mean(), abs(t.statistic))


def purete_voisinage(X, g):
    """Part des dix plus proches voisins qui partagent la catégorie d'un article.

    La valeur brute ne se lit pas seule : un corpus dont la moitié des articles
    porterait la même catégorie afficherait une pureté élevée sans qu'aucune
    géométrie l'explique. Elle est donc rapportée au niveau qu'atteindrait un
    étiquetage indépendant de la position des points, soit la probabilité que
    deux articles tirés au hasard portent la même catégorie.
    """
    from sklearn.neighbors import NearestNeighbors
    idx = [i for i, c in enumerate(g) if c]
    Y = X[idx]
    lab = np.array([g[i] for i in idx])
    nn = NearestNeighbors(n_neighbors=VOISINS_PURETE + 1, metric="cosine").fit(Y)
    _, v = nn.kneighbors(Y)
    part = (lab[v[:, 1:]] == lab[:, None]).mean()
    freq = collections.Counter(lab)
    n = len(lab)
    hasard = sum((c / n) ** 2 for c in freq.values())
    return part, hasard, n


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--recalculer", action="store_true",
                   help="refait la projection au lieu de relire le cache")
    args = p.parse_args()

    police()
    SORTIE.mkdir(exist_ok=True)
    X, ids = plongements()
    xy = projeter(X, args.recalculer)
    print(f"projection    {xy.shape[0]} articles")

    gh, gk = figure_regroupements(xy, ids)
    dr, de, mr, me, t = voisinage(X, gh, gk)
    print(f"regroupements.pdf   articles retenus contre articles écartés, "
          f"sur {VOISINS_DENSITE} voisins")
    print(f"                    distance moyenne  {dr:.3f} contre {de:.3f} "
          f"({de / dr - 1:+.1%})")
    print(f"                    groupes distincts {mr:.2f} contre {me:.2f} "
          f"(Welch t = {t:.1f})")

    g = figure_iptc(xy, ids)
    part, hasard, n = purete_voisinage(X, g)
    print(f"categories_iptc.pdf {n} articles étiquetés, pureté du voisinage "
          f"{part:.3f} contre {hasard:.3f} attendus sans rapport à la géométrie "
          f"(rapport {part / hasard:.1f})")


if __name__ == "__main__":
    main()
