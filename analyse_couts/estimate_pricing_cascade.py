"""
estimate_pricing_cascade.py : Estime le coût du pipeline en cascade (étage 1
niveau 2 → étage 2 niveau 3 par branche), sans appel API, pour comparer au
groupage simple (estimate_pricing_batched.py) et au 1-article/appel
(estimate_pricing.py).

Méthode :
  - Étage 1 : TOUS les articles du corpus échantillonné (stats_par_article.csv,
    100 fascicules, 6699 articles) passent par le stage 1, groupés en lots
    (MAX_BATCH_SIZE=25) avec la liste réduite à 120 candidats niveau 2.
  - Distribution des branches niveau 2 : pas une hypothèse uniforme : dérivée
    des classifications réelles déjà faites (feuilles_mistral_batched/,
    285 articles réels, cf. la répartition observée), puis extrapolée
    proportionnellement aux 6699 articles. 3,5% des articles s'arrêtent au
    stage 1 (branche terminale), 96,5% nécessitent le stage 2.
  - Étage 2 : pour chaque branche non-terminale, le nombre d'articles alloué
    (proportionnel à la distribution observée) est groupé en lots avec la
    liste réduite aux enfants de cette branche (taille réelle : 1 à 111
    selon la branche).
  - Approximation assumée : la longueur des articles (tokens) est prise à la
    moyenne du corpus (521,8 tokens) pour composer les lots de l'étage 2, faute
    de connaître la longueur réelle de chaque article de chaque branche à cette
    échelle (on ne peut pas la dériver du sous-échantillon de 285 articles avec
    la même confiance que la distribution des branches elle-même).

Sorties (dans ce même dossier analyse_couts/) :
  - cout_cascade_par_branche.csv
  - cout_cascade_resume.csv (comparaison avec les 2 autres scénarios)

Usage :
    python estimate_pricing_cascade.py
"""

import csv
import glob
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "classification"))

from classify_iptc_mistral_cascade import (  # noqa: E402
    MAX_BATCH_SIZE,
    MAX_BATCH_TOKENS,
    MISTRAL_MODEL,
    PRICING,
    STAGE1_SYSTEM_PROMPT,
    STAGE2_SYSTEM_PROMPT_TEMPLATE,
    TAXONOMY_PATH,
    branch_prompt_str,
    build_stage_indices,
    count_tokens,
    leaves_prompt_str,
)

OWN_DIR = Path(__file__).parent  # CSV de coût : lus/écrits dans ce même dossier analyse_couts/
RESULTS_DIR = Path(__file__).parent.parent / "results"  # results/ : sorties JSON de classification/, dossier frère

PER_ARTICLE_STATS_CSV = OWN_DIR / "stats_par_article.csv"
BATCHED_RESULTS_DIR = RESULTS_DIR / "feuilles_mistral_batched"

COST_BRANCH_CSV = OWN_DIR / "cout_cascade_par_branche.csv"
SUMMARY_CSV = OWN_DIR / "cout_cascade_resume.csv"
BASELINE_BATCHED_CSV = OWN_DIR / "cout_batched_resume.csv"

CACHE_HIT_RATE = 0.554
CACHE_DISCOUNT = 0.10
AVG_ARTICLE_TOKENS = 521.8  # moyenne mesurée sur les 6699 articles (stats_resume.csv)


def real_branch_distribution(l2_leaf_codes, l3_by_branch):
    """Dérive la distribution réelle (terminal vs par branche) à partir des
    classifications déjà faites (feuilles_mistral_batched/), pas une hypothèse."""
    terminal_n = 0
    branch_counter = Counter()
    total = 0
    child_to_branch = {c: l2c for l2c, children in l3_by_branch.items() for c in children}

    for f in glob.glob(str(BATCHED_RESULTS_DIR / "*_themes.json")):
        d = json.load(open(f, encoding="utf-8"))
        for art in d["articles"]:
            if not art["themes"]:
                continue
            top_code = art["themes"][0]["code"]
            total += 1
            if top_code in l2_leaf_codes:
                terminal_n += 1
            elif top_code in child_to_branch:
                branch_counter[child_to_branch[top_code]] += 1
    return terminal_n, branch_counter, total


def n_batches_for(n_articles, fixed_overhead, avg_tokens=AVG_ARTICLE_TOKENS, per_article_wrapper=15):
    """Approxime le nb de lots pour N articles de longueur moyenne connue,
    avec un overhead fixe donné (même logique que make_batches, sans avoir
    besoin de connaître le détail article par article)."""
    if n_articles == 0:
        return 0
    per_article = avg_tokens + per_article_wrapper
    room_per_batch = MAX_BATCH_TOKENS - fixed_overhead
    max_by_tokens = max(1, int(room_per_batch // per_article))
    batch_size = min(MAX_BATCH_SIZE, max_by_tokens)
    return -(-n_articles // batch_size)  # ceil


def call_cost(input_tokens, output_tokens, pin, pout, cache_hit_rate=0.0):
    """Coût d'un appel à partir de ses tokens input/output, avec prise en compte
    optionnelle du cache de prompt (cache_hit_rate s'applique uniquement à la
    partie input, jamais à l'output).
    """
    if cache_hit_rate:
        cached = input_tokens * cache_hit_rate
        uncached = input_tokens * (1 - cache_hit_rate)
        return uncached / 1e6 * pin + cached / 1e6 * pin * CACHE_DISCOUNT + output_tokens / 1e6 * pout
    return input_tokens / 1e6 * pin + output_tokens / 1e6 * pout


def main():
    """Point d'entrée : construit les index de taxonomie niveau 2/niveau 3,
    estime le coût de l'étage 1 (tous les articles, liste réduite à 120), puis
    de l'étage 2 (par branche, en extrapolant la distribution réelle observée
    sur les articles déjà classés : pas une hypothèse uniforme). Écrit le détail
    par branche (cout_cascade_par_branche.csv) et la comparaison avec le
    groupage simple et le 1-article/appel (cout_cascade_resume.csv).
    """
    l2_candidates, l3_by_branch, l2_leaf_codes = build_stage_indices(TAXONOMY_PATH)
    print(f"✓ {len(l2_candidates)} candidats niveau 2 ({len(l2_leaf_codes)} terminaux, {len(l3_by_branch)} branches avec enfants)")

    #  Étage 1 : overhead fixe (mesuré une fois) 
    l2_leaves_str = leaves_prompt_str(l2_candidates)
    fixed_overhead_l2 = count_tokens(STAGE1_SYSTEM_PROMPT) + count_tokens(l2_leaves_str) + 100
    print(f"✓ overhead étage 1 : {fixed_overhead_l2:,} tokens")

    #  Distribution réelle des branches (dérivée des classifications déjà faites) 
    terminal_n, branch_counter, total_observed = real_branch_distribution(l2_leaf_codes, l3_by_branch)
    print(f"✓ distribution dérivée de {total_observed} articles réellement classés "
          f"({terminal_n} terminaux, {sum(branch_counter.values())} répartis sur {len(branch_counter)} branches)")

    rows = list(csv.DictReader(open(PER_ARTICLE_STATS_CSV, encoding="utf-8")))
    n_total = len(rows)
    total_article_tokens = sum(int(r["n_tokens"]) for r in rows)
    avg_tokens = total_article_tokens / n_total
    print(f"✓ corpus complet : {n_total} articles, {avg_tokens:.1f} tokens/article en moyenne")

    pin, pout = PRICING.get(MISTRAL_MODEL, (None, None))

    # ÉTAGE 1 : tous les articles, groupés (comme le groupage simple)
    n_batches_stage1 = n_batches_for(n_total, fixed_overhead_l2, avg_tokens)
    stage1_input_tokens = n_batches_stage1 * fixed_overhead_l2 + total_article_tokens
    # sortie stage 1 : {"articles":[{"article_id":..,"level2_code":..}]} — 1 code par article, pas de tableau
    out_min = count_tokens(json.dumps({"articles": [{"article_id": "DIV.100", "level2_code": "20000002"}]}))
    out_max = out_min  # taille quasi fixe (un seul code, pas de variation 1-5)
    stage1_output_tokens = n_total * out_min

    stage1_cost_nc = call_cost(stage1_input_tokens, stage1_output_tokens, pin, pout)
    stage1_cost_c = call_cost(stage1_input_tokens, stage1_output_tokens, pin, pout, CACHE_HIT_RATE)
    print(f"\nÉTAGE 1 : {n_batches_stage1} lot(s), {stage1_input_tokens:,} tokens input, {stage1_output_tokens:,} tokens output")
    print(f"  coût : ${stage1_cost_nc:.2f} sans cache / ${stage1_cost_c:.2f} avec cache")

    #  ÉTAGE 2 : par branche, distribution extrapolée du réel 
    scale = n_total / total_observed
    terminal_n_scaled = round(terminal_n * scale)

    branch_rows = []
    stage2_input_total = 0
    stage2_output_total_min = 0
    stage2_output_total_max = 0
    stage2_batches_total = 0

    for l2_code, n_observed in branch_counter.items():
        n_scaled = max(1, round(n_observed * scale))
        children = l3_by_branch[l2_code]
        l2_label = l2_candidates[l2_code]["label_fr"]
        branch_str = branch_prompt_str(children)
        system_prompt = STAGE2_SYSTEM_PROMPT_TEMPLATE.format(l2_label=l2_label)
        fixed_overhead_branch = count_tokens(system_prompt) + count_tokens(branch_str) + 100

        n_batches = n_batches_for(n_scaled, fixed_overhead_branch, avg_tokens)
        input_tokens = n_batches * fixed_overhead_branch + n_scaled * avg_tokens

        codes_sample = list(children.keys())[:5] or list(children.keys())[:1]
        out_min_1 = count_tokens(json.dumps({"articles": [{"article_id": "DIV.100", "themes": [codes_sample[0]]}]}))
        out_max_1 = count_tokens(json.dumps({"articles": [{"article_id": "DIV.100", "themes": codes_sample}]}))

        output_min = n_scaled * out_min_1
        output_max = n_scaled * out_max_1

        cost_min_nc = call_cost(input_tokens, output_min, pin, pout)
        cost_max_nc = call_cost(input_tokens, output_max, pin, pout)
        cost_min_c = call_cost(input_tokens, output_min, pin, pout, CACHE_HIT_RATE)
        cost_max_c = call_cost(input_tokens, output_max, pin, pout, CACHE_HIT_RATE)

        branch_rows.append(
            {
                "l2_code": l2_code,
                "l2_label": l2_label,
                "n_enfants_l3": len(children),
                "n_articles_observes": n_observed,
                "n_articles_extrapoles": n_scaled,
                "n_lots": n_batches,
                "tokens_fixe_par_appel": fixed_overhead_branch,
                "tokens_input_total": round(input_tokens),
                "cout_sans_cache_min": round(cost_min_nc, 4),
                "cout_sans_cache_max": round(cost_max_nc, 4),
                "cout_avec_cache_min": round(cost_min_c, 4),
                "cout_avec_cache_max": round(cost_max_c, 4),
            }
        )
        stage2_input_total += input_tokens
        stage2_output_total_min += output_min
        stage2_output_total_max += output_max
        stage2_batches_total += n_batches

    OWN_DIR.mkdir(parents=True, exist_ok=True)
    with open(COST_BRANCH_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(branch_rows[0].keys()))
        w.writeheader()
        w.writerows(sorted(branch_rows, key=lambda r: -r["n_articles_extrapoles"]))
    print(f"\n✓ {COST_BRANCH_CSV}")

    stage2_cost_min_nc = sum(r["cout_sans_cache_min"] for r in branch_rows)
    stage2_cost_max_nc = sum(r["cout_sans_cache_max"] for r in branch_rows)
    stage2_cost_min_c = sum(r["cout_avec_cache_min"] for r in branch_rows)
    stage2_cost_max_c = sum(r["cout_avec_cache_max"] for r in branch_rows)

    print(f"\nÉTAGE 2 : {stage2_batches_total} lot(s) sur {len(branch_rows)} branches "
          f"(+ {terminal_n_scaled} article(s) terminé(s) dès l'étage 1, sans coût étage 2)")
    print(f"  coût sans cache : ${stage2_cost_min_nc:.2f} (min) — ${stage2_cost_max_nc:.2f} (max)")
    print(f"  coût avec cache : ${stage2_cost_min_c:.2f} (min) — ${stage2_cost_max_c:.2f} (max)")

    #  TOTAL CASCADE 
    total_nc_min = stage1_cost_nc + stage2_cost_min_nc
    total_nc_max = stage1_cost_nc + stage2_cost_max_nc
    total_c_min = stage1_cost_c + stage2_cost_min_c
    total_c_max = stage1_cost_c + stage2_cost_max_c
    total_calls = n_batches_stage1 + stage2_batches_total

    print(f"\n{'=' * 60}")
    print(f"CASCADE TOTALE — {total_calls} appel(s) ({n_batches_stage1} étage 1 + {stage2_batches_total} étage 2)")
    print(f"  sans cache : ${total_nc_min:.2f} (min) — ${total_nc_max:.2f} (max)")
    print(f"  avec cache : ${total_c_min:.2f} (min) — ${total_c_max:.2f} (max)")

    #  Comparaison avec les scénarios précédents 
    comparison = [
        {"scenario": "cascade", "n_appels": total_calls,
         "cout_sans_cache_min": round(total_nc_min, 2), "cout_sans_cache_max": round(total_nc_max, 2),
         "cout_avec_cache_min": round(total_c_min, 2), "cout_avec_cache_max": round(total_c_max, 2)},
    ]
    if BASELINE_BATCHED_CSV.exists():
        for row in csv.DictReader(open(BASELINE_BATCHED_CSV, encoding="utf-8")):
            comparison.append(
                {
                    "scenario": row["scenario"],
                    "n_appels": row["n_appels"],
                    "cout_sans_cache_min": row["cout_sans_cache_min"],
                    "cout_sans_cache_max": row["cout_sans_cache_max"],
                    "cout_avec_cache_min": row["cout_avec_cache_min"],
                    "cout_avec_cache_max": row["cout_avec_cache_max"],
                }
            )
    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(comparison[0].keys()))
        w.writeheader()
        w.writerows(comparison)
    print(f"✓ {SUMMARY_CSV}")

    print(f"\n{'=' * 60}")
    print("COMPARAISON (sans cache, coût moyen min-max) :")
    for c in comparison:
        mid = (float(c["cout_sans_cache_min"]) + float(c["cout_sans_cache_max"])) / 2
        print(f"  {c['scenario']:<24} {c['n_appels']:>6} appels   ${mid:.2f}")


if __name__ == "__main__":
    main()
