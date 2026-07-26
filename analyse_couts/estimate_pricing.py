"""
estimate_pricing.py : Estime le coût de la classification IPTC en 1 article/appel,
par article, par fascicule, et pour le corpus entier, en décomposant chaque appel
en ses 3 composantes : prompt système (fixe), liste IPTC (fixe), article (variable).

Tokens de sortie (complétion) :
  - min/max sont calculés exactement à partir du schéma JSON strict (1 à 5
    thèmes garantis par `enum`) : pas une hypothèse, une borne réelle.
  - la moyenne (COMPLETION_PER_ARTICLE) reste calibrée sur un run réel de 30
    articles (930 tokens / 30), à prendre comme meilleure estimation "typique"
    entre les deux bornes.

Hypothèses (calibrées sur le même run réel de 30 articles) :
  - CACHE_HIT_RATE : part du préfixe fixe (system + liste IPTC) effectivement
    facturée au tarif réduit du cache — ne s'applique QUE au prompt système
    et à la liste IPTC (identiques à chaque appel), jamais au texte de
    l'article (différent à chaque appel, jamais en cache).

Entrée (dans ce même dossier analyse_couts/) :
  - stats_par_article.csv (généré par analyze_corpus_tokens.py)

Sorties (dans ce même dossier analyse_couts/) :
  - cout_par_article.csv     : détail des 3 composantes par article
  - cout_par_fascicule.csv   : idem agrégé par fascicule
  - cout_resume.csv          : min/max/moyenne/médiane + total corpus
  - cout_comparaison_modeles.csv

Usage :
    python estimate_pricing.py
"""

import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "classification"))

from classify_iptc_mistral_batched import MISTRAL_MODEL, PRICING, SYSTEM_PROMPT, TAXONOMY_PATH  # noqa: E402
from classify_iptc_mistral_batched import build_leaves, count_tokens, leaves_prompt_str  # noqa: E402

RESULTS_DIR = Path(__file__).parent  # lit/écrit dans ce même dossier analyse_couts/
PER_ARTICLE_STATS_CSV = RESULTS_DIR / "stats_par_article.csv"

COST_PER_ARTICLE_CSV = RESULTS_DIR / "cout_par_article.csv"
COST_PER_FASCICULE_CSV = RESULTS_DIR / "cout_par_fascicule.csv"
COST_SUMMARY_CSV = RESULTS_DIR / "cout_resume.csv"
COST_MODELS_CSV = RESULTS_DIR / "cout_comparaison_modeles.csv"

# Calibration empirique (run réel, fascicule 4109000, 30 articles) 
COMPLETION_PER_ARTICLE = 31    # 930 tokens de sortie / 30 articles
CACHE_HIT_RATE = 0.554         # 159 534 / 288 192 tokens de prompt en cache
CACHE_DISCOUNT = 0.10          # -90% annoncé sur les tokens en cache (mistral.ai/pricing/api/)


def stats(values):
    """Statistiques min/max/moyenne/médiane d'une liste de valeurs : {} si la
    liste est vide (évite une exception sur une catégorie sans données).
    """
    if not values:
        return {"min": 0, "max": 0, "mean": 0, "median": 0}
    return {
        "min": round(min(values), 5),
        "max": round(max(values), 5),
        "mean": round(statistics.mean(values), 5),
        "median": round(statistics.median(values), 5),
    }


def call_cost(tokens_prompt, tokens_iptc, tokens_article, pin, pout, output_min, output_max):
    """Décompose le coût d'un appel (1 article) en ses 3 composantes.

    Le cache (quand il s'applique) ne porte que sur (prompt + iptc) : le seul
    contenu strictement identique d'un appel à l'autre. `tokens_article`
    n'est jamais en cache : c'est un texte différent à chaque appel.

    La complétion est donnée sur 3 scénarios : min (1 thème, borne réelle du
    schéma), moyen (calibré sur un run réel), max (5 thèmes, borne réelle).
    """
    cost_prompt_nc = tokens_prompt / 1e6 * pin
    cost_iptc_nc = tokens_iptc / 1e6 * pin
    cost_prompt_c = cost_prompt_nc * ((1 - CACHE_HIT_RATE) + CACHE_HIT_RATE * CACHE_DISCOUNT)
    cost_iptc_c = cost_iptc_nc * ((1 - CACHE_HIT_RATE) + CACHE_HIT_RATE * CACHE_DISCOUNT)
    cost_article = tokens_article / 1e6 * pin

    cost_completion_min = output_min / 1e6 * pout
    cost_completion = COMPLETION_PER_ARTICLE / 1e6 * pout
    cost_completion_max = output_max / 1e6 * pout

    base_input_nc = cost_prompt_nc + cost_iptc_nc + cost_article
    base_input_c = cost_prompt_c + cost_iptc_c + cost_article

    return {
        "cout_prompt_sans_cache": cost_prompt_nc,
        "cout_prompt_avec_cache": cost_prompt_c,
        "cout_iptc_sans_cache": cost_iptc_nc,
        "cout_iptc_avec_cache": cost_iptc_c,
        "cout_article": cost_article,
        "cout_completion_min": cost_completion_min,
        "cout_completion": cost_completion,
        "cout_completion_max": cost_completion_max,
        "cout_total_sans_cache_min": base_input_nc + cost_completion_min,
        "cout_total_sans_cache": base_input_nc + cost_completion,
        "cout_total_sans_cache_max": base_input_nc + cost_completion_max,
        "cout_total_avec_cache_min": base_input_c + cost_completion_min,
        "cout_total_avec_cache": base_input_c + cost_completion,
        "cout_total_avec_cache_max": base_input_c + cost_completion_max,
    }


def main():
    """Point d'entrée : relit stats_par_article.csv (produit par
    analyze_corpus_tokens.py), calcule le coût de chaque article (3 composantes
    + bornes de sortie min/moyen/max, avec/sans cache), écrit le détail par
    article et par fascicule, le résumé statistique, la comparaison entre
    modèles, et affiche la répartition du coût total du corpus par composante.
    """
    if not PER_ARTICLE_STATS_CSV.exists():
        raise SystemExit(f"{PER_ARTICLE_STATS_CSV} introuvable — lance d'abord analyze_corpus_tokens.py")

    leaves = build_leaves(TAXONOMY_PATH)
    leaves_str = leaves_prompt_str(leaves)
    tokens_prompt = count_tokens(SYSTEM_PROMPT)
    tokens_iptc = count_tokens(leaves_str) + 50  # + gabarit texte ("Étiquettes disponibles :" etc., marge)
    print(f"✓ tokens prompt système : {tokens_prompt:,}")
    print(f"✓ tokens liste IPTC (567 étiquettes) : {tokens_iptc:,}")

    # Bornes réelles de la sortie JSON (schéma strict : 1 à 5 thèmes)
    codes = list(leaves.keys())
    output_min = count_tokens(json.dumps({"themes": [codes[0]]}))
    output_max = count_tokens(json.dumps({"themes": codes[:5]}))
    print(f"✓ tokens de sortie : min {output_min} (1 thème) / moyen {COMPLETION_PER_ARTICLE} (calibré) / max {output_max} (5 thèmes)")

    rows = list(csv.DictReader(open(PER_ARTICLE_STATS_CSV, encoding="utf-8")))
    for r in rows:
        r["n_tokens"] = int(r["n_tokens"])

    pin, pout = PRICING.get(MISTRAL_MODEL, (None, None))
    if pin is None:
        raise SystemExit(f"Tarif inconnu pour {MISTRAL_MODEL} — ajoute-le à PRICING dans classify_iptc_mistral_batched.py")
    print(f"✓ modèle : {MISTRAL_MODEL} (${pin}/M in, ${pout}/M out)\n")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # détail par article (3 composantes) 
    article_out = []
    for r in rows:
        c = call_cost(tokens_prompt, tokens_iptc, r["n_tokens"], pin, pout, output_min, output_max)
        article_out.append(
            {
                "fascicule": r["fascicule"],
                "article_id": r["article_id"],
                "title": r["title"],
                "tokens_prompt": tokens_prompt,
                "tokens_iptc": tokens_iptc,
                "tokens_article": r["n_tokens"],
                "tokens_total": tokens_prompt + tokens_iptc + r["n_tokens"],
                **{k: round(v, 6) for k, v in c.items()},
            }
        )
    with open(COST_PER_ARTICLE_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(article_out[0].keys()))
        w.writeheader()
        w.writerows(article_out)
    print(f"✓ {COST_PER_ARTICLE_CSV}")

    # agrégation par fascicule
    by_fasc = defaultdict(list)
    for a in article_out:
        by_fasc[a["fascicule"]].append(a)

    sum_cols = [
        "tokens_prompt", "tokens_iptc", "tokens_article", "tokens_total",
        "cout_prompt_sans_cache", "cout_prompt_avec_cache",
        "cout_iptc_sans_cache", "cout_iptc_avec_cache",
        "cout_article",
        "cout_completion_min", "cout_completion", "cout_completion_max",
        "cout_total_sans_cache_min", "cout_total_sans_cache", "cout_total_sans_cache_max",
        "cout_total_avec_cache_min", "cout_total_avec_cache", "cout_total_avec_cache_max",
    ]
    fascicule_out = []
    for fascicule, arts in by_fasc.items():
        row = {"fascicule": fascicule, "n_articles": len(arts)}
        for col in sum_cols:
            row[col] = round(sum(a[col] for a in arts), 6)
        fascicule_out.append(row)
    with open(COST_PER_FASCICULE_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fascicule_out[0].keys()))
        w.writeheader()
        w.writerows(fascicule_out)
    print(f"✓ {COST_PER_FASCICULE_CSV}")

    # résumé (min/max/moyenne/médiane, par article et par fascicule)
    summary_rows = []
    for col in sum_cols:
        summary_rows.append({"niveau": "article", "mesure": col, **stats([a[col] for a in article_out])})
    for col in sum_cols:
        summary_rows.append({"niveau": "fascicule", "mesure": col, **stats([f[col] for f in fascicule_out])})
    for scenario in ("sans_cache", "avec_cache"):
        summary_rows.append(
            {
                "niveau": "corpus_entier",
                "mesure": f"cout_total_{scenario}",
                "min": round(sum(a[f"cout_total_{scenario}_min"] for a in article_out), 2),
                "max": round(sum(a[f"cout_total_{scenario}_max"] for a in article_out), 2),
                "mean": "",
                "median": round(sum(a[f"cout_total_{scenario}"] for a in article_out), 2),
            }
        )
    with open(COST_SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["niveau", "mesure", "min", "max", "mean", "median"])
        w.writeheader()
        w.writerows(summary_rows)
    print(f"✓ {COST_SUMMARY_CSV}  (note : pour 'corpus_entier', le total est dans la colonne 'median')")

    #  comparaison entre modèles (total corpus)
    models_out = []
    for model, (m_pin, m_pout) in PRICING.items():
        total_nc = total_c = total_nc_min = total_nc_max = total_c_min = total_c_max = 0.0
        for r in rows:
            c = call_cost(tokens_prompt, tokens_iptc, r["n_tokens"], m_pin, m_pout, output_min, output_max)
            total_nc += c["cout_total_sans_cache"]
            total_c += c["cout_total_avec_cache"]
            total_nc_min += c["cout_total_sans_cache_min"]
            total_nc_max += c["cout_total_sans_cache_max"]
            total_c_min += c["cout_total_avec_cache_min"]
            total_c_max += c["cout_total_avec_cache_max"]
        models_out.append(
            {
                "modele": model,
                "prix_in_par_M": m_pin,
                "prix_out_par_M": m_pout,
                "cout_corpus_sans_cache_min": round(total_nc_min, 2),
                "cout_corpus_sans_cache": round(total_nc, 2),
                "cout_corpus_sans_cache_max": round(total_nc_max, 2),
                "cout_corpus_avec_cache_min": round(total_c_min, 2),
                "cout_corpus_avec_cache": round(total_c, 2),
                "cout_corpus_avec_cache_max": round(total_c_max, 2),
            }
        )
    with open(COST_MODELS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(models_out[0].keys()))
        w.writeheader()
        w.writerows(models_out)
    print(f"✓ {COST_MODELS_CSV}")

    total_nc = sum(a["cout_total_sans_cache"] for a in article_out)
    total_c = sum(a["cout_total_avec_cache"] for a in article_out)
    total_nc_min = sum(a["cout_total_sans_cache_min"] for a in article_out)
    total_nc_max = sum(a["cout_total_sans_cache_max"] for a in article_out)
    total_prompt = sum(a["cout_prompt_sans_cache"] for a in article_out)
    total_iptc = sum(a["cout_iptc_sans_cache"] for a in article_out)
    total_article = sum(a["cout_article"] for a in article_out)
    total_completion_min = sum(a["cout_completion_min"] for a in article_out)
    total_completion = sum(a["cout_completion"] for a in article_out)
    total_completion_max = sum(a["cout_completion_max"] for a in article_out)

    print(f"\n{'=' * 60}")
    print(f"CORPUS ENTIER ({MISTRAL_MODEL}) : répartition sans cache :")
    print(f"  prompt système  : ${total_prompt:,.2f}")
    print(f"  liste IPTC      : ${total_iptc:,.2f}")
    print(f"  articles        : ${total_article:,.2f}")
    print(f"  complétion      : ${total_completion_min:,.2f} (min, 1 thème) : ${total_completion:,.2f} (moyen) : ${total_completion_max:,.2f} (max, 5 thèmes)")
    print(f"  TOTAL sans cache : ${total_nc_min:,.2f} (min) : ${total_nc:,.2f} (moyen) : ${total_nc_max:,.2f} (max)")
    print(f"  TOTAL avec cache : ${total_c:,.2f} (moyen)")


if __name__ == "__main__":
    main()
