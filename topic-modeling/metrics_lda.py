"""
metrics_lda.py : métriques d'évaluation d'un modèle thématique

CE QUE FAIT CE SCRIPT

Lit les sorties d'un run et calcule quatre familles de mesures. Il sert aux deux
familles de méthodes, la modélisation thématique et le regroupement de
plongements écrivant leurs résultats dans le même format.

Aucune de ces mesures ne dit si un thème est juste. Aucune vérité de référence
n'existe pour ce corpus, et les mesures servent à comparer des configurations
entre elles plutôt qu'à établir une qualité dans l'absolu.

LES MESURES

  NPMI            Cohérence d'un thème, mesurée par la fréquence à laquelle ses
                  mots de tête paraissent ensemble dans un corpus de référence.
                  L'information mutuelle ponctuelle est normalisée par la
                  probabilité de la paire, ce qui borne le résultat entre -1 et
                  +1 : -1 quand deux mots ne paraissent jamais ensemble, 0
                  quand ils sont indépendants, +1 quand ils vont toujours de
                  pair. Jey Han Lau, David Newman et Timothy Baldwin ont montré
                  en 2014 que cette mesure est celle qui s'accorde le mieux au
                  jugement des annotateurs.

  Diversité       Part de mots distincts parmi tous les mots de tête réunis.
                  Une valeur de 1 signale que les thèmes ne partagent aucun mot.
                  Une valeur basse signale des thèmes redondants, défaut que le
                  NPMI seul ne détecte pas : des thèmes identiques seraient tous
                  très cohérents.

  Entropie        Incertitude de l'affectation d'un document à un thème. Elle
                  vaut 0 lorsqu'un document appartient entièrement à un thème,
                  et ln(K) lorsqu'il se répartit également entre tous. La
                  version normalisée divise par ln(K), ce qui rend les valeurs
                  comparables entre des runs de K différents.

  Jaccard         Stabilité entre deux runs conduits dans les mêmes conditions
                  avec des graines différentes. Chaque thème du premier est
                  apparié au thème du second qui lui ressemble le plus, sans
                  réemploi, et le recouvrement moyen mesure ce que la part
                  aléatoire de l'estimation laisse varier.

LE CORPUS DE RÉFÉRENCE

Le NPMI se calcule sur un corpus qui fournit les dénombrements de cooccurrence.
Un mot de tête absent de ce corpus n'y a pas de probabilité estimable, et reçoit
alors un NPMI de -1, valeur qui signifie "jamais co-occurrents" quand la
situation est "non mesurable". Employer une référence qui ne couvre pas le
vocabulaire d'un run écrase donc sa valeur par un effet de plancher.

Les valeurs rapportées dans le mémoire proviennent pour cette raison de la
référence non filtrée que construit reference_brute.py.

ENTRÉES, dans le dossier du run

  mallet.topic_distributions.txt ou span_topic.json   affectation des documents
  topics.json                                         mots de tête
  training_data.txt                                   corpus prétraité
  meta.json                                           paramètres du run

SORTIE

  metrics.json ou metrics_ref.json dans le dossier du run, et un rapport imprimé

PAQUETS EMPLOYÉS

  argparse, json, math, re, collections, itertools, pathlib   bibliothèque
  standard

Aucune dépendance extérieure n'est nécessaire. Les mesures sont calculées ici
plutôt qu'importées d'une bibliothèque, afin que leur définition exacte soit
lisible et que les deux familles de méthodes soient mesurées par le même code.

USAGE

    python3 metrics_lda.py --run resultats/{nom_du_run}
    python3 metrics_lda.py --run resultats/{nom_du_run} --n-top 10
    python3 metrics_lda.py --run resultats/{nom_du_run} --reference
"""

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path


# Chargement des fichiers MALLET

def load_distributions(path: Path) -> list[list[float]]:
    """Charge mallet.topic_distributions.txt → liste de vecteurs de proba."""
    distrib = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split("\t")
        if len(parts) < 3:
            continue
        # Format : index \t doc_id \t p0 \t p1 \t ... \t pK-1
        probs = [float(x) for x in parts[2:]]
        if probs:
            distrib.append(probs)
    return distrib


def load_topic_keys(path: Path, n_top: int) -> list[list[str]]:
    """Charge mallet.topic_keys.txt → liste de listes de top-mots par topic."""
    topics = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split("\t")
        if len(parts) < 3:
            continue
        words = parts[2].split()[:n_top]
        topics.append(words)
    return topics


def load_corpus(path: Path) -> list[list[str]]:
    """Charge training_data.txt → liste de docs tokenisés.

    Le format MALLET place l'identifiant du document et son étiquette en tête
    de ligne. Les compter comme des mots fausserait le NPMI, l'étiquette étant
    présente dans tous les documents."""
    docs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        champs = line.rstrip("\n").split("\t")
        texte = champs[2] if len(champs) >= 3 else line
        words = texte.strip().split()
        if words:
            docs.append(words)
    return docs


# Entropie

def entropy_per_doc(probs: list[float]) -> float:
    """Entropie de Shannon d'une distribution de topics pour un document."""
    return -sum(p * math.log(p) for p in probs if p > 0)


def mean_entropy(distrib: list[list[float]]) -> dict:
    """
    Entropie moyenne sur tous les documents.
    Retourne aussi l'entropie max théorique (= ln K) pour normalisation.
    """
    K = len(distrib[0])
    entropies = [entropy_per_doc(d) for d in distrib]
    mean  = sum(entropies) / len(entropies)
    max_e = math.log(K) if K > 1 else 1.0
    return {
        "entropy_mean":       round(mean, 4),
        "entropy_max":        round(max_e, 4),
        "entropy_normalized": round(mean / max_e, 4),   # 0=parfait, 1=uniforme
        "entropy_per_doc":    [round(e, 4) for e in entropies],
    }


# Diversité

def diversity(topics: list[list[str]], corpus: list[list[str]] | None = None) -> dict:
    """
    diversity      = |mots uniques dans top-words| / (K × N_top)
    diversity_norm = |mots uniques dans top-words| / |vocab distinct du corpus|
                     → pénalise les sources avec un grand vocab artificiel (bruit OCR)
    """
    all_words  = [w for t in topics for w in t]
    unique     = len(set(all_words))
    total      = len(all_words)
    counts     = Counter(all_words)
    shared     = [w for w, c in counts.items() if c > 1]

    tokens_distincts = len({w for doc in corpus for w in doc}) if corpus else None
    div_norm         = round(unique / tokens_distincts, 4) if tokens_distincts else None

    return {
        "diversity":          round(unique / total, 4),
        "diversity_norm":     div_norm,
        "tokens_distincts":   tokens_distincts,
        "unique_words":       unique,
        "total_words":        total,
        "shared_words":       sorted(shared),
        "n_shared":           len(shared),
    }


# NPMI (cohérence)

def build_cooc(corpus: list[list[str]], vocab: set[str]) -> tuple[Counter, Counter, int]:
    """
    Construit les compteurs de co-occurrence par document (fenêtre = document entier).
    Retourne (co_counts, word_counts, n_docs).
    """
    word_counts = Counter()
    co_counts   = Counter()
    n_docs      = len(corpus)

    for doc in corpus:
        words_in_doc = set(doc) & vocab
        for w in words_in_doc:
            word_counts[w] += 1
        for w1, w2 in combinations(sorted(words_in_doc), 2):
            co_counts[(w1, w2)] += 1

    return co_counts, word_counts, n_docs


def npmi_pair(w1: str, w2: str, co_counts: Counter,
              word_counts: Counter, n_docs: int, eps: float = 1e-10) -> float:
    """NPMI entre deux mots : log(P(w1,w2) / P(w1)P(w2)) / -log(P(w1,w2))"""
    p_w1   = word_counts[w1] / n_docs
    p_w2   = word_counts[w2] / n_docs
    pair   = tuple(sorted([w1, w2]))
    p_w1w2 = co_counts[pair] / n_docs

    if p_w1w2 < eps or p_w1 < eps or p_w2 < eps:
        return -1.0   # jamais co-occurrants = cohérence nulle

    pmi   = math.log(p_w1w2 / (p_w1 * p_w2))
    npmi  = pmi / (-math.log(p_w1w2))
    return max(-1.0, min(1.0, npmi))


def topic_npmi(topic_words: list[str], co_counts: Counter,
               word_counts: Counter, n_docs: int) -> float:
    """NPMI moyen sur toutes les paires de top-mots d'un topic."""
    pairs = list(combinations(topic_words, 2))
    if not pairs:
        return 0.0
    scores = [npmi_pair(w1, w2, co_counts, word_counts, n_docs) for w1, w2 in pairs]
    return sum(scores) / len(scores)


def mean_npmi(topics: list[list[str]], corpus: list[list[str]]) -> dict:
    """NPMI moyen sur tous les topics."""
    vocab     = set(w for t in topics for w in t)
    co_counts, word_counts, n_docs = build_cooc(corpus, vocab)

    per_topic = []
    for i, t in enumerate(topics):
        score = topic_npmi(t, co_counts, word_counts, n_docs)
        per_topic.append({"topic_id": i, "npmi": round(score, 4), "top_words": " ".join(t)})

    mean = sum(x["npmi"] for x in per_topic) / len(per_topic)
    return {
        "npmi_mean":      round(mean, 4),
        "npmi_per_topic": per_topic,
    }


# Affichage

def print_report(meta: dict, ent: dict, div: dict, npmi_res: dict):
    """Imprime le rapport de mesures d'un run.

    Le rapport donne les valeurs d'ensemble, puis le détail par thème, ce qui
    permet de repérer un thème isolément incohérent. Un thème dont le NPMI vaut
    -1 signale le plus souvent un vocabulaire absent du corpus de référence
    plutôt qu'une véritable incohérence.
    """
    K = meta.get("k", "?")
    print(f"\n{'═'*58}")
    print(f"  Métriques : {meta.get('fascicule','?')}  K={K}  ({meta.get('n_docs','?')} docs)")
    print(f"{'═'*58}")

    print("\nEntropie (incertitude d'affectation)")
    print(f"  Moyenne          : {ent['entropy_mean']:.4f}")
    print(f"  Max théorique    : {ent['entropy_max']:.4f}  (= ln {K})")
    print(f"  Normalisée [0,1] : {ent['entropy_normalized']:.4f}"
          f"  {'← proche 0 = topics distincts ✓' if ent['entropy_normalized'] < 0.5 else '← proche 1 = topics mélangés ✗'}")

    print("\nDiversité (unicité des mots-clés)")
    print(f"  Score [0,1]      : {div['diversity']:.4f}"
          f"  {'← bonne diversité ✓' if div['diversity'] > 0.7 else '← mots trop partagés ✗'}")
    if div['diversity_norm'] is not None:
        print(f"  Normalisée/vocab : {div['diversity_norm']:.4f}"
              f"  (mots uniques / {div['tokens_distincts']} tokens distincts du corpus)")
    print(f"  Mots uniques     : {div['unique_words']} / {div['total_words']}")
    if div['shared_words']:
        print(f"  Mots partagés    : {', '.join(div['shared_words'][:15])}"
              f"{'…' if len(div['shared_words']) > 15 else ''}")

    print("\nNPMI (cohérence des top-mots)")
    print(f"  Moyenne [-1,+1]  : {npmi_res['npmi_mean']:.4f}"
          f"  {'← cohérence correcte ✓' if npmi_res['npmi_mean'] > -0.1 else '← mots peu liés ✗'}")
    print()
    print(f"  Par topic :")
    for t in npmi_res["npmi_per_topic"]:
        bar = "▓" * max(0, int((t['npmi'] + 1) * 5))
        print(f"    Topic {t['topic_id']:2d}  NPMI={t['npmi']:+.4f}  {bar:<10}  {t['top_words'][:50]}")

    print()


# Main

def load_from_nmf(run_dir: Path, n_top: int):
    """Charge distributions et topics depuis une sortie NMF (span_topic + topics.json)."""
    topics_raw = json.loads((run_dir / "topics.json").read_text())
    span_topic = json.loads((run_dir / "span_topic.json").read_text())
    k = len(topics_raw)

    # Distributions : dist[] dans span_topic
    distrib = [st["dist"] for st in span_topic if "dist" in st]

    # Top-mots
    topics = [t["top_words"].split()[:n_top] for t in topics_raw]

    # Corpus tokenisé
    corpus_path = run_dir / "training_data.txt"
    corpus = load_corpus(corpus_path) if corpus_path.exists() else []

    return distrib, topics, corpus


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run",   required=True, help="Dossier de résultats LDA ou NMF")
    parser.add_argument("--n-top", type=int, default=10, help="Nb top-mots pour NPMI")
    parser.add_argument("--json",  action="store_true", help="Export metrics.json")
    parser.add_argument("--reference", action="store_true",
                        help="calcule le NPMI et la diversité sur le corpus de "
                             "référence commun plutôt que sur celui du run, ce "
                             "qui rend les méthodes comparables entre elles")
    args = parser.parse_args()

    run_dir = Path(args.run)
    meta    = json.loads((run_dir / "meta.json").read_text())

    # Détecter le modèle
    mallet_dir = run_dir / "mallet_training"
    is_mallet  = mallet_dir.exists()

    print(f"Chargement depuis {run_dir.name}  [{'MALLET' if is_mallet else 'NMF'}]…")
    if is_mallet:
        distrib = load_distributions(mallet_dir / "mallet.topic_distributions.txt")
        topics  = load_topic_keys(mallet_dir / "mallet.topic_keys.txt", args.n_top)
        corpus  = load_corpus(mallet_dir / "training_data.txt")
    else:
        distrib, topics, corpus = load_from_nmf(run_dir, args.n_top)

    if args.reference:
        import corpus_reference
        source = meta.get("source", "mistral")
        if source.endswith("_reocr") or source == "mistral_reocr":
            source = "mistral"
        granularite = meta.get("granularite", "bloc")
        corpus = corpus_reference.charger(source, granularite)
        print(f"  corpus de référence commun : {source}/{granularite}")

    print(f"  {len(distrib)} docs, {len(topics)} topics, {len(corpus)} docs corpus")

    # Calcul
    print("Calcul des métriques…")
    ent      = mean_entropy(distrib)
    div      = diversity(topics, corpus)
    npmi_res = mean_npmi(topics, corpus)

    # Affichage
    print_report(meta, ent, div, npmi_res)

    # Export optionnel
    if args.json:
        out = {
            "meta":      meta,
            "reference_commune": bool(args.reference),
            "entropy":   ent,
            "diversity": div,
            "npmi":      npmi_res,
        }
        out_path = run_dir / ("metrics_ref.json" if args.reference
                              else "metrics.json")
        out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
        print(f"\n  → {out_path}")
