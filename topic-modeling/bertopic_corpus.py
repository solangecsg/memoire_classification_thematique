"""
bertopic_corpus.py : regroupement de plongements de phrases sur le sous-corpus

CE QUE FAIT CE SCRIPT

Représente chaque document par un vecteur produit par un modèle de phrases,
réduit la dimension de ces vecteurs, puis les regroupe. Chaque groupe est décrit
par les mots qui lui sont propres, mesurés par une pondération c-TF-IDF.

Le script est le pendant de lda_mallet_corpus.py, dont il reprend la
constitution du corpus afin que les deux chapitres portent sur les mêmes
documents. La comparaison entre regroupements est contrôlée : la représentation
vectorielle est calculée une seule fois puis mise en cache, et seul l'algorithme
de regroupement varie.

LA CHAÎNE, ÉTAPE PAR ÉTAPE

  1. Plongement. Chaque document devient un vecteur de 768 dimensions. Le calcul
     est long, et le cache le rend inutile à répéter.
  2. Réduction de dimension par UMAP. Les algorithmes de regroupement se
     comportent mal en grande dimension, où les distances se ressemblent toutes.
     L'option --sans-umap permet d'éprouver cette justification, qui ne se
     vérifie pas sur ce corpus.
  3. Regroupement, par k-means ou par densité.
  4. Description des groupes par c-TF-IDF, qui pondère chaque forme par sa
     fréquence dans le groupe et sa rareté dans le reste du corpus.

LES DEUX ALGORITHMES DE REGROUPEMENT

  --regroupement kmeans    demande un nombre de groupes, comme la modélisation
                           thématique, et affecte tout document à un groupe.
  --regroupement hdbscan   déduit le nombre de groupes de la densité des points,
                           et laisse sans groupe ceux qu'aucune zone dense ne
                           réclame. Ce rejet est la propriété qui l'oppose au
                           précédent.

ENTRÉES

  ../re-ocr/corpus/reocr_mistral/, ../re-ocr/corpus/original/   fascicules
  embeddings/{...}.npy      vecteurs mis en cache, recalculés s'ils manquent
  embeddings/{...}.ids.json manifeste des identifiants, qui valide le cache

SORTIES, dans resultats/{nom_du_run}/

  topics.json        groupes, effectifs et mots de tête
  span_topic.json    affectation de chaque document
  meta.json          paramètres exacts du run
  training_data.txt  corpus prétraité, employé par metrics_lda.py

Ce format est celui que lit metrics_lda.py, de sorte que les deux familles de
méthodes soient évaluées par le même code.

PAQUETS EMPLOYÉS

  argparse, json, sys, time, datetime, pathlib   bibliothèque standard
  numpy                     tableaux de vecteurs et cache sur disque
  sentence_transformers     calcul des plongements
  bertopic                  assemblage de la chaîne et pondération c-TF-IDF
  umap                      réduction de dimension
  sklearn                   k-means, et dénombrement des formes pour la
                            description des groupes
  hdbscan                   regroupement par densité, employé par bertopic
  lda_mallet_corpus         constitution du corpus, importée pour que les
                            documents soient identiques à ceux du chapitre 1

USAGE

    python3 bertopic_corpus.py --regroupement kmeans  --k 20
    python3 bertopic_corpus.py --regroupement hdbscan --min-taille 30
    python3 bertopic_corpus.py --regroupement kmeans  --k 20 --sans-umap
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import lda_mallet_corpus as base   # constitution du corpus, ressources, chemins

ICI = Path(__file__).resolve().parent
OUTPUT_DIR = ICI / "resultats"
CACHE = ICI / "embeddings"
MODELES = {
    # nom court : (identifiant, longueur maximale, préfixe éventuel)
    "minilm":    ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", 128, ""),
    "camembert": ("dangvantuan/sentence-camembert-base", 128, ""),
    # e5 demande un préfixe ; « query: » est celui que ses auteurs prescrivent
    # pour les tâches symétriques, dont le regroupement.
    "e5":        ("intfloat/multilingual-e5-base", 512, "query: "),
}


def charger_documents(source: str, granularite: str, min_mots: int,
                      longueur_min: int) -> list[dict]:
    """Reprend la constitution de lda_mallet_corpus, sans filtrage de fréquence :
    les plongements travaillent sur le texte, pas sur un sac de mots."""
    racine = base.SOURCES[source]["racine"]
    suffixe = base.SOURCES[source]["suffixe"]
    fascicules = sorted(
        d.name[:-len(suffixe)] if suffixe and d.name.endswith(suffixe) else d.name
        for d in racine.iterdir() if d.is_dir())
    stopwords, stoplocs = base.charger_stopwords(), base.charger_stoplocs()
    docs = base.constituer(source, granularite, fascicules, min_mots,
                           stopwords, stoplocs, longueur_min, 0)
    return docs


def plonger(docs: list[dict], source: str, granularite: str, modele_court: str):
    """Calcule les plongements, ou les relit si un cache existe. Le cache rend la
    comparaison entre regroupements exacte : les deux partent des mêmes vecteurs."""
    import numpy as np
    CACHE.mkdir(exist_ok=True)
    identifiant, longueur, prefixe = MODELES[modele_court]
    base_nom = f"{source}_{granularite}_{modele_court}_{len(docs)}"
    chemin = CACHE / f"{base_nom}.npy"
    manifeste = CACHE / f"{base_nom}.ids.json"
    ids = [d["doc_id"] for d in docs]
    if chemin.exists() and manifeste.exists():
        # Le nombre de documents ne suffit pas à identifier un corpus : deux
        # constitutions différentes peuvent y aboutir. Les identifiants sont donc
        # comparés avant de réutiliser des vecteurs.
        if json.loads(manifeste.read_text(encoding="utf-8")) == ids:
            print(f"  plongements relus depuis {chemin.name}")
            return np.load(chemin)
        print(f"  cache {chemin.name} écarté : les documents ont changé")
    from sentence_transformers import SentenceTransformer
    print(f"  calcul des plongements avec {identifiant}…")
    t0 = time.time()
    modele = SentenceTransformer(identifiant)
    vecteurs = modele.encode([prefixe + d["texte"] for d in docs],
                             batch_size=64, show_progress_bar=True,
                             convert_to_numpy=True)
    print(f"  {vecteurs.shape[0]} vecteurs de dimension {vecteurs.shape[1]} "
          f"en {time.time()-t0:.0f}s")
    np.save(chemin, vecteurs)
    manifeste.write_text(json.dumps(ids, ensure_ascii=False), encoding="utf-8")
    print(f"  enregistrés dans {chemin.name} avec leur manifeste d'identifiants")
    return vecteurs


def main() -> None:
    """Assemble la chaîne, lance le regroupement et enregistre les sorties.

    Les valeurs par défaut sont celles de la configuration retenue au
    chapitre 2. Chacune est commentée à l'endroit où elle est déclarée dans
    l'analyseur d'arguments ci-dessous.

    Le vectoriseur de description mérite une note. Il ne sert pas au
    regroupement, qui porte sur les vecteurs, mais uniquement à nommer les
    groupes une fois formés. Ses trois réglages sont : la liste d'arrêt du
    corpus, un seuil de cinq documents pour écarter les formes trop rares, et un
    motif de jeton qui ne retient que les suites d'au moins trois lettres, ce
    qui écarte les nombres et les débris d'océrisation.
    """
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=sorted(base.SOURCES), default="mistral")
    p.add_argument("--granularite", choices=["bloc", "article"], default="article")
    p.add_argument("--regroupement", choices=["kmeans", "hdbscan"], required=True)
    p.add_argument("--k", type=int, default=20, help="kmeans seulement")
    p.add_argument("--min-taille", type=int, default=30, help="hdbscan seulement")
    p.add_argument("--modele", choices=sorted(MODELES), default="minilm",
                   help="modèle de plongement ; sa longueur maximale conditionne "
                        "la part de l'article effectivement encodée")
    p.add_argument("--voisins", type=int, default=15,
                   help="UMAP n_neighbors : étendue du voisinage considéré")
    p.add_argument("--dimensions", type=int, default=5,
                   help="UMAP n_components : dimension de l'espace réduit")
    p.add_argument("--sans-umap", action="store_true",
                   help="regroupe directement les plongements, sans réduction")
    p.add_argument("--graine", type=int, default=1)
    p.add_argument("--longueur-min", type=int, default=3)
    p.add_argument("--min-mots", type=int, default=None)
    args = p.parse_args()

    min_mots = args.min_mots if args.min_mots is not None else (
        5 if args.granularite == "bloc" else 20)

    print(f"\n{'='*66}")
    print(f"BERTopic : {args.source} | {args.granularite} | {args.regroupement}"
          + (f" k={args.k}" if args.regroupement == "kmeans"
             else f" taille min={args.min_taille}"))
    print(f"{'='*66}\n1. Constitution du corpus…")
    docs = charger_documents(args.source, args.granularite, min_mots,
                             args.longueur_min)
    if not docs:
        sys.exit("Aucun document retenu.")

    print("\n2. Représentation vectorielle…")
    vecteurs = plonger(docs, args.source, args.granularite, args.modele)

    print("\n3. Regroupement…")
    from bertopic import BERTopic
    from bertopic.vectorizers import ClassTfidfTransformer
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP

    stopwords = sorted(base.charger_stopwords())
    vectoriseur = CountVectorizer(stop_words=stopwords, min_df=5,
                                  token_pattern=r"(?u)\b[^\W\d_]{3,}\b")
    if args.sans_umap:
        # BERTopic exige un objet de réduction ; celui-ci rend les vecteurs tels
        # quels, ce qui permet d'isoler l'effet de la réduction de dimension.
        class SansReduction:
            """Réduction neutre, qui rend les vecteurs tels quels.

            BERTopic exige un objet de réduction et appelle sur lui les trois
            méthodes de l'interface de scikit-learn. Les fournir sans rien faire
            permet de regrouper directement les 768 dimensions, et donc de
            mesurer ce que la réduction apporte ou retire.
            """

            def fit(self, X, y=None):
                """Ne fait rien et rend l'objet, comme l'exige l'interface."""
                return self

            def transform(self, X):
                """Rend les vecteurs sans les modifier."""
                return X

            def fit_transform(self, X, y=None):
                """Rend les vecteurs sans les modifier."""
                return X
        reduction = SansReduction()
    else:
        reduction = UMAP(n_neighbors=args.voisins, n_components=args.dimensions,
                         min_dist=0.0, metric="cosine", random_state=args.graine)

    if args.regroupement == "kmeans":
        from sklearn.cluster import KMeans
        regroupeur = KMeans(n_clusters=args.k, random_state=args.graine, n_init=10)
    else:
        from hdbscan import HDBSCAN
        regroupeur = HDBSCAN(min_cluster_size=args.min_taille,
                             metric="euclidean", cluster_selection_method="eom",
                             prediction_data=True)

    # language="english", valeur par défaut, applique aux documents un
    # re.sub(r"[^A-Za-z0-9 ]+", "") qui supprime tout caractère accentué :
    # « théâtre » devient « thtre » et « qu'il » devient « quil ».
    modele = BERTopic(language="multilingual",
                      embedding_model=None, umap_model=reduction,
                      hdbscan_model=regroupeur, vectorizer_model=vectoriseur,
                      ctfidf_model=ClassTfidfTransformer(),
                      calculate_probabilities=False, verbose=False)
    t0 = time.time()
    affectations, _ = modele.fit_transform([d["texte"] for d in docs],
                                           embeddings=vecteurs)
    duree = time.time() - t0
    print(f"  terminé en {duree:.1f}s")

    info = modele.get_topic_info()
    groupes = [int(t) for t in info["Topic"].tolist()]
    print(f"  {sum(1 for g in groupes if g >= 0)} groupes"
          + (f", {int(info.loc[info['Topic'] == -1, 'Count'].iloc[0])} documents "
             f"sans groupe" if -1 in groupes else ", aucun document écarté"))

    # ── Sorties, au format des runs LDA ───────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffixe = (f"k{args.k}" if args.regroupement == "kmeans"
               else f"mt{args.min_taille}")
    umap_tag = ("_sansumap" if args.sans_umap else
                (f"_v{args.voisins}d{args.dimensions}"
                 if (args.voisins, args.dimensions) != (15, 5) else ""))
    run_id = (f"bertopic_{args.regroupement}_{args.source}_{args.granularite}"
              f"_{suffixe}_{args.modele}{umap_tag}_g{args.graine}_{ts}")
    dossier = OUTPUT_DIR / run_id
    dossier.mkdir(parents=True, exist_ok=True)

    tops = []
    for g in sorted(t for t in groupes if t >= 0):
        mots = [m for m, _ in modele.get_topic(g)][:10]
        tops.append({"topic_id": g, "top_words": " ".join(mots),
                     "size": int(info.loc[info["Topic"] == g, "Count"].iloc[0])})
    (dossier / "topics.json").write_text(
        json.dumps(tops, ensure_ascii=False, indent=2), encoding="utf-8")

    k_reel = len(tops)
    span = []
    for d, a in zip(docs, affectations):
        a = int(a)
        dist = [0.0] * k_reel
        if a >= 0:
            dist[a] = 1.0
        span.append({"doc_id": d["doc_id"], "fascicule": d["fascicule"],
                     "unite": d["unite"], "page": d["page"],
                     "topic_id": a, "weight": 1.0 if a >= 0 else 0.0,
                     "dist": dist})
    (dossier / "span_topic.json").write_text(
        json.dumps(span, ensure_ascii=False, indent=2), encoding="utf-8")

    # Corpus de référence pour le NPMI, au format attendu par metrics_lda.py
    analyse = vectoriseur.build_analyzer()
    (dossier / "training_data.txt").write_text(
        "\n".join(f"{d['doc_id']}\tno_label\t{' '.join(analyse(d['texte']))}"
                  for d in docs), encoding="utf-8")

    (dossier / "meta.json").write_text(json.dumps(
        {"model": f"bertopic_{args.regroupement}", "run_id": run_id,
         "source": args.source, "granularite": args.granularite,
         "k": k_reel, "k_demande": args.k if args.regroupement == "kmeans" else None,
         "min_taille": args.min_taille if args.regroupement == "hdbscan" else None,
         "graine": args.graine, "n_docs": len(docs),
         "n_sans_groupe": sum(1 for a in affectations if int(a) < 0),
         "longueur_min": args.longueur_min, "min_mots": min_mots,
         "modele_plongement": MODELES[args.modele][0],
         "modele_court": args.modele,
         "umap_voisins": None if args.sans_umap else args.voisins,
         "umap_dimensions": None if args.sans_umap else args.dimensions,
         "longueur_max": MODELES[args.modele][1], "duree_s": round(duree, 1)},
        ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n  → {dossier.name}")
    for t in sorted(tops, key=lambda x: -x["size"])[:15]:
        print(f"    groupe {t['topic_id']:3d} ({t['size']:5d}) : {t['top_words']}")


if __name__ == "__main__":
    main()
