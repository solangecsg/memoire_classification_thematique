# Classification thématique IPTC — articles de presse ancienne (Mistral)

Pipeline de classification multi-label (1 à 5 thèmes IPTC Media Topics niveau 3) d'articles de presse française ancienne (BnF/Gallica, OCR corrigé), via l'API Mistral. Comprend aussi la chaîne de ré-OCRisation (Mistral vision + Pero) et de mesure de qualité OCR qui alimente le corpus utilisé par la classification.

## Installation

```bash
pip install -r requirements.txt
```

Créer ensuite `config/.env` à la racine du dépôt avec :

```
MISTRAL_API_KEY=votre-clé-ici
```

(fichier volontairement absent du dépôt — jamais commité)

## Arborescence

```
classification-iptc/
├── requirements.txt
├── config/.env                                  ← à créer (MISTRAL_API_KEY=...)
│
├── classification/         ← les 4 variantes de classification LLM
├── analyse_couts/           ← estimation de coûts, sans appel API
├── results/                  ← sorties + synthèse
└── re-ocr/
    ├── scripts/              ← ré-OCRisation (Mistral vision + Pero)
    ├── metriques/             ← mesure de qualité OCR
    └── corpus/                ← ALTO/METS, avant et après ré-OCRisation
```

---

## `classification/` — classer les articles par thème IPTC

Les 4 scripts font tous la MÊME tâche (lire les articles d'un fascicule, leur assigner 1 à 5 thèmes IPTC via Mistral, sauvegarder un JSON par fascicule) mais avec une stratégie différente pour réduire le coût. Ils sont indépendants (chacun peut tourner seul) et partagent `iptc_mediatopic_official.json` (taxonomie IPTC officielle, format SKOS/JSON-LD, source : `cv.iptc.org/newscodes/mediatopic/`).

Usage commun :
```bash
cd classification
python classify_iptc_mistral<variante>.py --fascicule 4109000 --dry-run   # extraction + aperçu, sans appel API
python classify_iptc_mistral<variante>.py --fascicule 4109000 4109676     # fascicules précis
python classify_iptc_mistral<variante>.py                                 # tout le corpus
python classify_iptc_mistral<variante>.py --force                        # retraiter même si déjà sorti
```

| Script | Stratégie | Verdict |
|---|---|---|
| `classify_iptc_mistral.py` | Référence : 1 article = 1 appel API, liste complète des 567 étiquettes à chaque fois. | Coût de référence, le plus simple à comprendre. |
| `classify_iptc_mistral_cached.py` | Comme la référence, mais le contenu fixe (instructions + liste) est regroupé dans le message `system` + `prompt_cache_key`, pour profiter du cache de prompt Mistral (-90% sur les tokens déjà vus). | Gain réel mais modeste (~-47% sur l'échantillon testé). |
| `classify_iptc_mistral_batched.py` | Regroupe plusieurs articles par appel (budget dynamique de tokens, jusqu'à 25 articles/appel) — le coût fixe de la liste est ainsi partagé entre tous les articles du lot. | **La plus économique (-89% vs la référence). Voir `results/SYNTHESE_COUTS.md`.** |
| `classify_iptc_mistral_cascade.py` | En 2 étages : d'abord une classification niveau 2 (120 catégories, liste courte) sur tous les articles, puis un 2ᵉ passage niveau 3 par branche assignée (liste réduite aux seuls enfants de cette branche). | **Essayée et rejetée : coûte PLUS cher que le groupage simple** — le texte de chaque article est envoyé 2 fois (une fois par étage), ce qui mange l'économie faite sur la liste. Voir section 8 de `SYNTHESE_COUTS.md` pour le détail du mécanisme. Conservée dans le dépôt à titre d'exemple documenté d'une piste d'optimisation qui semblait logique sur le papier mais ne tient pas à la mesure. |

Points communs aux 4 scripts :
- Comptage de tokens réel via `mistral-common` (tokenizer Mistral téléchargé depuis Hugging Face au premier lancement), pas une approximation en caractères.
- Schéma JSON strict (`response_format: json_schema`) : les codes IPTC renvoyés par Mistral sont contraints par `enum` à la liste fournie — impossible d'obtenir un code hors taxonomie.
- Retry différencié : 401/403 arrête tout de suite (clé invalide), 400 abandonne sans réessayer (requête invalide), 429/5xx retente avec backoff exponentiel.
- Suivi des tokens et du temps de réponse par appel, agrégés par fascicule puis pour tout le run (avec estimation de coût selon `MISTRAL_MODEL`).
- Reprise automatique : un fascicule déjà sorti est sauté (sauf `--force`).

## `analyse_couts/` — estimer le coût sans dépenser un centime

Ces scripts ne font AUCUN appel API — ils réutilisent les fonctions d'extraction et de tokenisation de `classification/classify_iptc_mistral_batched.py` pour mesurer/estimer les coûts avant de lancer un vrai run.

| Script | Rôle | Sortie |
|---|---|---|
| `analyze_corpus_tokens.py` | Statistiques mots/tokens sur tout le corpus (par article, par fascicule). À lancer en premier. | `results/stats_par_article.csv`, `results/stats_resume.csv` |
| `estimate_pricing.py` | Coût du scénario 1-article/appel : décompose chaque appel en 3 composantes (prompt système, liste IPTC, texte article), bornes de sortie min/moyen/max, comparaison entre modèles. | `results/cout_par_article.csv`, `cout_par_fascicule.csv`, `cout_resume.csv`, `cout_comparaison_modeles.csv` |
| `estimate_pricing_batched.py` | Coût du groupage par lot : reproduit l'algorithme de composition des lots pour prédire exactement quels articles seraient groupés ensemble, et leur coût. | `results/lots_composition.csv`, `cout_par_lot.csv`, `cout_batched_resume.csv` |
| `estimate_pricing_cascade.py` | Coût de la cascade : distribution des branches niveau 2 dérivée des articles RÉELLEMENT déjà classés (pas une hypothèse uniforme), extrapolée à tout le corpus. | `results/cout_cascade_par_branche.csv`, `cout_cascade_resume.csv` |

```bash
cd analyse_couts
python analyze_corpus_tokens.py        # à lancer en premier (produit stats_par_article.csv)
python estimate_pricing.py
python estimate_pricing_batched.py
python estimate_pricing_cascade.py
```

Le récit complet (scénarios, échelles jusqu'à 2,6M fascicules, hébergement local vs cloud, ce qui est mesuré vs supposé) est dans **`results/SYNTHESE_COUTS.md`**.

## `results/` — sorties

- `SYNTHESE_COUTS.md` — document de synthèse (à lire en premier pour comprendre les résultats)
- `stats_*.csv`, `cout_*.csv`, `lots_composition.csv` — données brutes derrière la synthèse
- `feuilles_mistral/`, `feuilles_mistral_cached/`, `feuilles_mistral_batched/`, `feuilles_mistral_cascade/` — sorties de classification par fascicule, une par variante de script
- `sample_review.json` — échantillon utilisé pour la relecture qualité manuelle

## `re-ocr/` — ré-OCRisation et mesure de qualité

### `re-ocr/scripts/` — corriger l'OCR d'origine

| Script | Rôle |
|---|---|
| `reocr_mistral.py` | Pour chaque bloc ALTO d'une page : crop l'image JP2/JPG correspondante, envoie le crop à Mistral (`pixtral-12b`) en vision, remplace le texte ALTO par la transcription Mistral. Usage : `python reocr_mistral.py --key VOTRE_CLE [--fascicule ID] [--page N] [--dry-run]`. |
| `reocr_mistral.ipynb` | Version notebook (Google Colab) du script ci-dessus. |
| `reocr_pero.ipynb` | Ré-OCRisation via le moteur **Pero** (notebook Colab, GPU requis) — alternative à Mistral pour comparer la qualité entre moteurs. |
| `reocr_pero_image.ipynb` | Variante de `reocr_pero.ipynb` travaillant directement sur les images plutôt que sur l'ALTO existant. |

### `re-ocr/metriques/` — mesurer la qualité de l'OCR

| Script | Rôle | Sortie |
|---|---|---|
| `stats_pages.py` | Par page et par source (BnF original / Pero / Mistral) : tokens distincts, nb de mots, nb de caractères, nb d'articles (depuis le TOC/METS) — pour comparer objectivement les 3 sources. Usage : `python stats_pages.py [--n 10]`. | `resultats_stats/stats_par_page.csv`, `stats_mistral_brut.csv`, `vocab_venn.html` |
| `ocrqa/impresso_ocrqa.py` | Score de qualité OCR (ratio de mots reconnus, méthode **impresso**) sur les 3 sources. Usage : `python3 impresso_ocrqa.py [--n 5]`. | `ocrqa/resultats_ocrqa/ocrqa_results.json`, `ocrqa_summary.csv`, `ocrqa_report.html` |

### `re-ocr/corpus/` — données (suivies via **Git LFS**, `.gitattributes`, ~790 Mo)

- `original/{fascicule}/{toc,ocr}` — ALTO/METS d'origine (100 fascicules, sans les images)
- `reocr_mistral/{fascicule}_reocr/{toc,ocr,logs}` — ALTO corrigé par Mistral (100 fascicules)

---

## Ce qui n'est PAS inclus (volontairement)

- **Les images** (JP2/JPG des pages scannées) : ~19 Go pour 100 fascicules, soumises aux conditions de réutilisation Gallica — seuls les fichiers texte (ALTO/METS) sont inclus.
- **La clé API** (`config/.env`) : à créer soi-même, jamais commitée.
- **L'interface de relecture/vérité terrain** (app Streamlit) : reste en local, hors de ce dépôt.

Cette arborescence est déjà en place dans ce dépôt (corpus texte inclus, sans les images) — les scripts tournent directement une fois la clé API renseignée dans `config/.env`.
