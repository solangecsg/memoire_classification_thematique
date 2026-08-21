"""
analyze_corpus_tokens.py : statistiques de mots/tokens sur tout le corpus,
sans aucun appel API, pour dimensionner le pipeline de classification IPTC.

Réutilise l'extraction d'articles et le tokenizer Mistral réel (mistral-common)
déjà mis en place dans classify_iptc_mistral_batched.py.

Produit (dans ce même dossier analyse_couts/) :
  - stats_par_article.csv : 1 ligne par article retenu
    (fascicule, article_id, title, n_words, n_tokens)
  - stats_resume.csv : min/max/moyenne/médiane,
    à la fois par article et par fascicule
  - un résumé en console : tokens du prompt système, de la liste IPTC
    (567 étiquettes), et tokens d'input total pour tout le corpus en
    1 appel/article

LE DÉNOMBREMENT DES JETONS

Les comptes viennent du tokeniseur du modèle, mistral-common, épinglé dans le
requirements.txt à la version 1.11.6. En son absence, count_tokens se rabat sur
une estimation par caractères qui donne des valeurs sensiblement différentes,
sans erreur ni avertissement : la liste des étiquettes y pèse 4 821 jetons au
lieu de 8 775, et le coût annoncé tombe de moitié. Les valeurs rapportées dans le
mémoire supposent le tokeniseur installé.


Usage :
    python3 analyze_corpus_tokens.py
"""

import csv
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "classification"))

from classify_iptc_mistral_batched import (  # noqa: E402
    MIN_WORDS,
    MISTRAL_RESULTS_DIR,
    SYSTEM_PROMPT,
    TAXONOMY_PATH,
    build_leaves,
    count_tokens,
    extract_articles,
    leaves_prompt_str,
)

OUTPUT_DIR = Path(__file__).parent  # écrit dans ce même dossier analyse_couts/
PER_ARTICLE_CSV = OUTPUT_DIR / "stats_par_article.csv"
SUMMARY_CSV = OUTPUT_DIR / "stats_resume.csv"


def stats(values):
    """Statistiques min/max/moyenne/médiane d'une liste de valeurs : {} si la
    liste est vide (évite une exception sur une catégorie sans données).
    """
    if not values:
        return {"min": 0, "max": 0, "mean": 0, "median": 0}
    return {
        "min": min(values),
        "max": max(values),
        "mean": round(statistics.mean(values), 1),
        "median": statistics.median(values),
    }


def main():
    """Point d'entrée : charge la taxonomie, mesure les tokens du prompt système
    et de la liste IPTC, puis parcourt tous les fascicules pour extraire et
    tokeniser chaque article retenu. Écrit stats_par_article.csv (détail) et
    stats_resume.csv (min/max/moyenne/médiane par article et par fascicule),
    et affiche le total de tokens d'input pour tout le corpus en 1-article/appel.
    Aucun appel API : uniquement extraction + tokenisation locale.
    """
    leaves = build_leaves(TAXONOMY_PATH)
    leaves_str = leaves_prompt_str(leaves)

    system_tokens = count_tokens(SYSTEM_PROMPT)
    iptc_list_tokens = count_tokens(leaves_str)
    print(f"✓ {len(leaves)} étiquettes IPTC chargées")
    print(f"  tokens prompt système : {system_tokens:,}")
    print(f"  tokens liste IPTC (567 étiquettes) : {iptc_list_tokens:,}")

    fascicules = sorted(
        p.name.replace("_reocr", "")
        for p in MISTRAL_RESULTS_DIR.iterdir()
        if p.is_dir() and p.name.endswith("_reocr")
    )
    print(f"✓ {len(fascicules)} fascicule(s) trouvé(s) dans {MISTRAL_RESULTS_DIR.name}\n")

    rows = []
    per_fascicule = {}

    for i, fascicule in enumerate(fascicules):
        articles = extract_articles(fascicule)
        kept = [a for a in articles if len(a["text"].split()) >= MIN_WORDS]

        fasc_words, fasc_tokens = 0, 0
        for a in kept:
            n_words = len(a["text"].split())
            n_tokens = count_tokens(a["text"])
            rows.append(
                {
                    "fascicule": fascicule,
                    "article_id": a["id"],
                    "title": a["title"],
                    "n_words": n_words,
                    "n_tokens": n_tokens,
                }
            )
            fasc_words += n_words
            fasc_tokens += n_tokens

        per_fascicule[fascicule] = {
            "n_articles": len(kept),
            "n_words": fasc_words,
            "n_tokens": fasc_tokens,
        }
        print(f"  [{i+1}/{len(fascicules)}] {fascicule} : {len(kept)} article(s), {fasc_words:,} mots, {fasc_tokens:,} tokens")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(PER_ARTICLE_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["fascicule", "article_id", "title", "n_words", "n_tokens"])
        w.writeheader()
        w.writerows(rows)
    print(f"\n✓ {PER_ARTICLE_CSV}")

    article_words = [r["n_words"] for r in rows]
    article_tokens = [r["n_tokens"] for r in rows]
    fasc_n_articles = [v["n_articles"] for v in per_fascicule.values()]
    fasc_words = [v["n_words"] for v in per_fascicule.values()]
    fasc_tokens = [v["n_tokens"] for v in per_fascicule.values()]

    summary_rows = [
        {"niveau": "article", "mesure": "n_words", **stats(article_words)},
        {"niveau": "article", "mesure": "n_tokens", **stats(article_tokens)},
        {"niveau": "fascicule", "mesure": "n_articles", **stats(fasc_n_articles)},
        {"niveau": "fascicule", "mesure": "n_words_total", **stats(fasc_words)},
        {"niveau": "fascicule", "mesure": "n_tokens_total", **stats(fasc_tokens)},
    ]
    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["niveau", "mesure", "min", "max", "mean", "median"])
        w.writeheader()
        w.writerows(summary_rows)
    print(f"✓ {SUMMARY_CSV}")

    n_articles_total = len(rows)
    per_call_overhead = system_tokens + iptc_list_tokens + 50  # + gabarit texte (~50 tokens de marge)
    total_fixed = n_articles_total * per_call_overhead
    total_variable = sum(article_tokens)
    total_input = total_fixed + total_variable

    print(f"\n{'=' * 60}")
    print(f"Corpus complet : {len(fascicules)} fascicule(s), {n_articles_total} article(s) retenu(s)")
    print(f"  tokens fixes par appel (system + liste + gabarit) : {per_call_overhead:,}")
    print(f"  total tokens fixes (x{n_articles_total} appels, 1 article/appel) : {total_fixed:,}")
    print(f"  total tokens articles (variable) : {total_variable:,}")
    print(f"  Total des jetons d'entrée, un article par appel : {total_input:,}")


if __name__ == "__main__":
    main()
