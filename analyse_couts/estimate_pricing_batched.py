"""
estimate_pricing_batched.py : Pré-détermine la composition des lots (le même
algorithme glouton que classify_iptc_mistral_batched.py utiliserait réellement)
et estime le coût correspondant : input ef output sans aucun appel API, pour
comparer au scénario 1-article/appel déjà calculé par estimate_pricing.py.

Tokens de sortie (complétion) :
  - min/max sont calculés exactement à partir du schéma JSON strict du lot
    (chaque article : 1 à 5 thèmes garantis par `enum`, `article_id` réel du
    lot) : pas une hypothèse, une borne réelle, propre à chaque lot (dépend
    du nombre d'articles et de la longueur de leurs identifiants).
  - "moyen" = point milieu min/max , une estimation, pas une calibration
    réelle (on n'a pas encore de run réel en mode groupé pour calibrer).

Sorties :
  - themes/results/lots_composition.csv : quel article va dans quel lot
  - themes/results/cout_par_lot.csv     : coût de chaque lot (min/moyen/max)
  - themes/results/cout_batched_resume.csv : comparaison 1-article/appel vs groupage

Usage :
    python estimate_pricing_batched.py
"""

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "classification"))

from classify_iptc_mistral_batched import (  # noqa: E402
    MAX_BATCH_SIZE,
    MAX_BATCH_TOKENS,
    MIN_WORDS,
    MISTRAL_MODEL,
    MISTRAL_RESULTS_DIR,
    PRICING,
    SYSTEM_PROMPT,
    TAXONOMY_PATH,
    build_leaves,
    count_tokens,
    extract_articles,
    leaves_prompt_str,
    make_batches,
)

RESULTS_DIR = Path(__file__).parent.parent / "results"  # results/ est un dossier frère de classification/ et analyse_couts/
LOTS_CSV = RESULTS_DIR / "lots_composition.csv"
COST_LOTS_CSV = RESULTS_DIR / "cout_par_lot.csv"
SUMMARY_CSV = RESULTS_DIR / "cout_batched_resume.csv"
BASELINE_CSV = RESULTS_DIR / "cout_par_article.csv"  # produit par estimate_pricing.py

# Même hypothèse de cache que estimate_pricing.py (même run réel de référence)
CACHE_HIT_RATE = 0.554
CACHE_DISCOUNT = 0.10


def batch_output_tokens(batch, codes):
    """Bornes réelles (min/max) des tokens de sortie pour ce lot précis,
    calculées sur le vrai JSON du schéma strict avec les vrais article_id."""
    min_json = json.dumps({"articles": [{"article_id": a["id"], "themes": [codes[0]]} for a in batch]})
    max_json = json.dumps({"articles": [{"article_id": a["id"], "themes": codes[:5]} for a in batch]})
    return count_tokens(min_json), count_tokens(max_json)


def main():
    """Point d'entrée : reproduit exactement l'algorithme de groupage de
    classify_iptc_mistral_batched.py (make_batches) sur tout le corpus, sans
    aucun appel API, pour connaître la composition réelle des lots à l'avance.
    Écrit la composition des lots (lots_composition.csv), leur coût
    (cout_par_lot.csv, bornes min/moyen/max de sortie), et compare le total au
    scénario 1-article/appel déjà calculé par estimate_pricing.py.
    """
    leaves = build_leaves(TAXONOMY_PATH)
    leaves_str = leaves_prompt_str(leaves)
    codes = list(leaves.keys())
    fixed_overhead = count_tokens(SYSTEM_PROMPT) + count_tokens(leaves_str) + 50
    print(f"✓ overhead fixe par appel : {fixed_overhead:,} tokens")
    print(f"✓ budget par lot : {MAX_BATCH_TOKENS:,} tokens, {MAX_BATCH_SIZE} articles max")

    pin, pout = PRICING.get(MISTRAL_MODEL, (None, None))
    if pin is None:
        raise SystemExit(f"Tarif inconnu pour {MISTRAL_MODEL}")
    print(f"✓ modèle : {MISTRAL_MODEL} (${pin}/M in, ${pout}/M out)\n")

    fascicules = sorted(
        p.name.replace("_reocr", "")
        for p in MISTRAL_RESULTS_DIR.iterdir()
        if p.is_dir() and p.name.endswith("_reocr")
    )

    lot_rows = []
    cost_rows = []
    lot_id = 0

    for i, fascicule in enumerate(fascicules):
        articles = extract_articles(fascicule)
        kept = [a for a in articles if len(a["text"].split()) >= MIN_WORDS]
        batches = make_batches(kept, fixed_overhead)

        for b_idx, batch in enumerate(batches):
            lot_id += 1
            article_tokens = [count_tokens(a["text"]) for a in batch]
            batch_tokens = sum(article_tokens)
            n_articles = len(batch)

            for a, tok in zip(batch, article_tokens):
                lot_rows.append(
                    {
                        "lot_id": lot_id,
                        "fascicule": fascicule,
                        "batch_index": b_idx,
                        "article_id": a["id"],
                        "title": a["title"],
                        "n_tokens_article": tok,
                    }
                )

            output_min, output_max = batch_output_tokens(batch, codes)
            output_moyen = (output_min + output_max) / 2

            input_tokens = fixed_overhead + batch_tokens
            cached = fixed_overhead * CACHE_HIT_RATE
            uncached = fixed_overhead * (1 - CACHE_HIT_RATE) + batch_tokens

            row = {
                "lot_id": lot_id,
                "fascicule": fascicule,
                "batch_index": b_idx,
                "n_articles": n_articles,
                "tokens_articles": batch_tokens,
                "tokens_fixe": fixed_overhead,
                "tokens_total": input_tokens,
                "tokens_output_min": output_min,
                "tokens_output_moyen": round(output_moyen),
                "tokens_output_max": output_max,
            }
            for label, out_tok in (("min", output_min), ("moyen", output_moyen), ("max", output_max)):
                row[f"cout_sans_cache_{label}"] = round(
                    input_tokens / 1e6 * pin + out_tok / 1e6 * pout, 5
                )
                row[f"cout_avec_cache_{label}"] = round(
                    uncached / 1e6 * pin + cached / 1e6 * pin * CACHE_DISCOUNT + out_tok / 1e6 * pout, 5
                )
            cost_rows.append(row)

        sizes = [len(b) for b in batches]
        print(f"  [{i+1}/{len(fascicules)}] {fascicule} : {len(kept)} article(s) → {len(batches)} lot(s) (tailles : {sizes})")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(LOTS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(lot_rows[0].keys()))
        w.writeheader()
        w.writerows(lot_rows)
    print(f"\n✓ {LOTS_CSV}")

    with open(COST_LOTS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(cost_rows[0].keys()))
        w.writeheader()
        w.writerows(cost_rows)
    print(f"✓ {COST_LOTS_CSV}")

    n_calls = len(cost_rows)
    n_articles_total = sum(r["n_articles"] for r in cost_rows)

    totals = {}
    for label in ("min", "moyen", "max"):
        totals[f"nc_{label}"] = sum(r[f"cout_sans_cache_{label}"] for r in cost_rows)
        totals[f"c_{label}"] = sum(r[f"cout_avec_cache_{label}"] for r in cost_rows)

    baseline = {}
    if BASELINE_CSV.exists():
        base_rows = list(csv.DictReader(open(BASELINE_CSV, encoding="utf-8")))
        for label in ("min", "", "max"):
            suffix = f"_{label}" if label else ""
            baseline[f"nc_{label or 'moyen'}"] = sum(float(r[f"cout_total_sans_cache{suffix}"]) for r in base_rows)
            baseline[f"c_{label or 'moyen'}"] = sum(float(r[f"cout_total_avec_cache{suffix}"]) for r in base_rows)

    summary = [
        {
            "scenario": "1_article_par_appel",
            "n_appels": n_articles_total,
            "cout_sans_cache_min": round(baseline.get("nc_min", 0), 2),
            "cout_sans_cache_moyen": round(baseline.get("nc_moyen", 0), 2),
            "cout_sans_cache_max": round(baseline.get("nc_max", 0), 2),
            "cout_avec_cache_min": round(baseline.get("c_min", 0), 2),
            "cout_avec_cache_moyen": round(baseline.get("c_moyen", 0), 2),
            "cout_avec_cache_max": round(baseline.get("c_max", 0), 2),
        },
        {
            "scenario": "groupage_par_lot",
            "n_appels": n_calls,
            "cout_sans_cache_min": round(totals["nc_min"], 2),
            "cout_sans_cache_moyen": round(totals["nc_moyen"], 2),
            "cout_sans_cache_max": round(totals["nc_max"], 2),
            "cout_avec_cache_min": round(totals["c_min"], 2),
            "cout_avec_cache_moyen": round(totals["c_moyen"], 2),
            "cout_avec_cache_max": round(totals["c_max"], 2),
        },
    ]
    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    print(f"✓ {SUMMARY_CSV}")

    print(f"\n{'=' * 60}")
    print(f"GROUPAGE PAR LOT  {n_calls} appel(s) pour {n_articles_total} articles ({n_articles_total / n_calls:.1f} articles/appel en moyenne)")
    print(f"  sans cache : ${totals['nc_min']:,.2f} (min)  ${totals['nc_moyen']:,.2f} (moyen)  ${totals['nc_max']:,.2f} (max)")
    print(f"  avec cache : ${totals['c_min']:,.2f} (min)  ${totals['c_moyen']:,.2f} (moyen)  ${totals['c_max']:,.2f} (max)")
    if baseline:
        print(f"\nvs 1 article/appel ({n_articles_total} appels) :")
        print(f"  sans cache : ${baseline['nc_min']:,.2f} (min)  ${baseline['nc_moyen']:,.2f} (moyen)  ${baseline['nc_max']:,.2f} (max)")
        print(f"  avec cache : ${baseline['c_min']:,.2f} (min)  ${baseline['c_moyen']:,.2f} (moyen)  ${baseline['c_max']:,.2f} (max)")
        print(f"\n  économie sans cache (moyen) : ${baseline['nc_moyen'] - totals['nc_moyen']:,.2f} ({100 * (1 - totals['nc_moyen'] / baseline['nc_moyen']):.1f}%)")
        print(f"  économie avec cache (moyen) : ${baseline['c_moyen'] - totals['c_moyen']:,.2f} ({100 * (1 - totals['c_moyen'] / baseline['c_moyen']):.1f}%)")


if __name__ == "__main__":
    main()
