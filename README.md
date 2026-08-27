# Classification thématique IPTC : articles de presse patrimoniale numérisée

Pipeline de classification multi-label (1 à 5 thèmes IPTC Media Topics niveau 2 et 3) d'articles de presse française ancienne (BnF Gallica, OCR corrigé), via l'API Mistral. Comprend aussi la chaîne de ré-OCRisation (Mistral vision ou Pero) et de mesure de qualité OCR qui alimente le corpus utilisé par la classification.

[![tests](https://github.com/solangecsg/memoire_classification_thematique/actions/workflows/tests.yml/badge.svg)](https://github.com/solangecsg/memoire_classification_thematique/actions/workflows/tests.yml)

## Tests et intégration continue

Les fonctions dont dépendent les mesures rapportées sont couvertes par des tests
unitaires : la stratification par bande de fréquence, l'intervalle de Wilson, la
constitution des lots d'articles et la détection des thèmes de bruit. Ce sont
celles dont une erreur fausserait un résultat sans faire échouer aucune
exécution.

```bash
python -m pytest tests/ -v
```

Deux invariants y comptent plus que les autres, parce que toute la campagne en
dépend : **un article est envoyé entier ou pas du tout**, jamais tronqué, même
s'il dépasse à lui seul le budget de jetons d'un appel ; et **aucun lot n'est
vide**, ce qui ferait payer un appel sans contenu. Les deux ont été éprouvés par
mutation, en cassant volontairement la fonction pour vérifier que le test le
signale.

L'intégration continue (`.github/workflows/tests.yml`) lance les tests sur deux
versions de Python, vérifie que tous les scripts compilent, que chaque fonction
porte une docstring, et qu'aucune chaîne ressemblant à une clé d'API n'a été
versée. Aucune étape n'appelle un service distant : rien n'y dépense de jetons.

## Licence

Le code est sous licence MIT. Les fichiers ALTO et METS de `re-ocr/corpus/`
proviennent de la Bibliothèque nationale de France et restent soumis aux
conditions de leur producteur ; le référentiel IPTC Media Topics est publié sous
Creative Commons Attribution 4.0. Voir `LICENSE`.

## Prérequis

- **Python 3.11 ou supérieur** (testé avec Python 3.14).
- **Git LFS** installé avant de cloner le dépôt, sinon les fichiers du corpus (`re-ocr/corpus/`) sont récupérés sous forme de pointeurs LFS et non de vrais fichiers :
  ```bash
  git lfs install
  git clone git@github.com:solangecsg/memoire_classification_thematique.git
  ```
  Si le dépôt a déjà été cloné sans Git LFS installé, lancer `git lfs pull` depuis la racine du dépôt pour récupérer les vrais fichiers.
- **Une clé API Mistral valide**, à placer dans `config/.env` (voir Installation ci dessous), pour tout script de `classification/` ou `re-ocr/scripts/reocr_mistral.py` qui appelle l'API.

## Installation

```bash
pip install -r requirements.txt
```

Créer ensuite `config/.env` à la racine du dépôt avec :

```
MISTRAL_API_KEY=votre-clé-ici
```


## Arborescence

```
classification-iptc/
├── requirements.txt
├── config/.env                                  ← à créer (MISTRAL_API_KEY=...)
│
├── classification/         ← les variantes de classification, et leur vérification
│   └── verification/        ← épreuves en aveugle, réponses, sorties des contrôles
├── topic-modeling/          ← modélisation thématique, 111 exécutions et leurs mesures
├── analyse_couts/           ← estimation de coûts, sans appel API, et ses CSV
├── results/                  ← sorties de classification (JSON par fascicule)
└── re-ocr/
    ├── scripts/              ← ré-OCRisation (Mistral vision + Pero)
    ├── metriques/             ← mesure de qualité OCR
    └── corpus/                ← ALTO/METS, avant et après ré-OCRisation
```

---

## `classification/` : classer les articles par thème IPTC

Les 4 scripts `classify_iptc_mistral*.py` font tous la même tâche (lire les articles d'un fascicule, leur assigner 1 à 5 thèmes IPTC via Mistral, sauvegarder un JSON par fascicule) mais avec une stratégie différente pour réduire le coût. Les scripts `controle_*.py` et `verification_*.py`, décrits plus bas, éprouvent ces variantes et en mesurent la justesse. Ils sont indépendants (chacun peut tourner seul) et partagent `iptc_mediatopic_official.json` (taxonomie IPTC officielle, format SKOS/JSON-LD, source : `cv.iptc.org/newscodes/mediatopic/`).

Le diagramme d'activité de la variante retenue, groupage par lot, figure dans `classification/schema_pipeline_classification.png`, avec sa source vectorielle au format SVG à côté.

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
| `classify_iptc_mistral_batched.py` | Regroupe plusieurs articles par appel (budget dynamique de tokens, jusqu'à 25 articles/appel) le coût fixe de la liste est ainsi partagé entre tous les articles du lot. | **La plus économique, 89 % de moins que la référence. Les mesures détaillées sont dans `analyse_couts/`.** |
| `classify_iptc_mistral_cascade.py` | En 2 étages : d'abord une classification niveau 2 (120 catégories, liste courte) sur tous les articles, puis un 2ᵉ passage niveau 3 par branche assignée (liste réduite aux seuls enfants de cette branche). | **Plus chère que le groupage simple, et de qualité équivalente** : le texte de chaque article est envoyé 2 fois, une fois par étage, ce qui mange l'économie faite sur la liste. Le jugement humain conduit depuis la donne à 47,5 % contre 47,8 % pour le groupage, soit un écart nul pour un coût de 2,5 fois. Elle redevient en revanche la seule conception praticable dès que la facturation au jeton disparaît, c'est-à-dire en local : voir `classify_iptc_ollama_cascade.py`. |

Points communs aux 4 scripts `classify_iptc_mistral*.py` :
- Comptage de tokens réel via `mistral-common` (tokenizer Mistral téléchargé depuis Hugging Face au premier lancement), pas une approximation en caractères.
- Schéma JSON strict (`response_format: json_schema`) : les codes IPTC renvoyés par Mistral sont contraints par `enum` à la liste fournie . Impossible d'obtenir un code hors taxonomie.
- Retry différencié : 401/403 arrête tout de suite (clé invalide), 400 abandonne sans réessayer (requête invalide), 429/5xx retente avec backoff exponentiel.
- Suivi des tokens et du temps de réponse par appel, agrégés par fascicule puis pour tout le run (avec estimation de coût selon `MISTRAL_MODEL`).
- Reprise automatique : un fascicule déjà sorti est sauté (sauf `--force`).

---

## `classification/verification/` : la justesse des étiquettes

Les scripts ci-dessus mesurent un coût, un volume et une convergence entre
dispositifs. Aucun ne mesure une justesse, faute de vérité de référence pour ce
corpus. Quatre épreuves en aveugle ont été conduites pour y répondre dans les
limites du possible.

**Le principe est le même dans les quatre.** Un item présente un article et une
étiquette, sans dire d'où l'étiquette vient. La moitié environ des étiquettes
sont des leurres, tirés d'autres articles du même corpus et de la même bande de
fréquence. Le taux d'acceptation des leurres mesure la complaisance du jugement :
s'il est bas, le taux d'acceptation des vraies étiquettes devient lisible.

| Épreuve | Objet du jugement | Items | Vraies acceptées | Leurres acceptés |
|---|---|---|---|---|
| **A** | les 62 étiquettes que le corpus n'emploie qu'une fois | 62 | 31/62 = 50,0 % | (sans leurres) |
| **B** | des attributions de la campagne groupée | 129 | 33/69 = **47,8 %** | 1/60 = **1,7 %** |
| **C** | des attributions du régime un-article-par-appel | 116 | 23/58 = 39,7 % | 0/58 = 0,0 % |
| **D** | les deux cascades, mêlées et anonymes | 160 | commerciale 19/40 = **47,5 %**<br>locale 10/40 = **25,0 %** | 3/80 = 3,8 % |
| **E** | les deux régimes à justification préalable | 158 | cascade 28/40 = **70,0 %**<br>unitaire 23/39 = 59,0 % | 2/79 = 2,5 % |

625 items jugés au total, par un seul annotateur. Les taux de leurres, de 0,0 à
3,8 %, ne se distinguent pas les uns des autres : les cinq séances sont
comparables.

**Un second enseignement de l'épreuve A.** Sur les 25 étiquettes uniques rejetées,
15 le sont parce que la catégorie n'a pas de sens pour la période : six
disciplines sportives modernes, neuf notions postérieures au corpus. Le
référentiel IPTC n'attache aucune date à ses entrées, et le modèle les reçoit
donc comme des candidates légitimes pour un fascicule de 1880.

**Le consensus entre exécutions prédit la justesse.** Les étiquettes jugées en A
et B viennent toutes de la campagne groupée. Rapportées aux deux reprises
unitaires indépendantes, elles se répartissent ainsi :

| Reprises qui confirment l'étiquette | Acceptée par le jugement |
|---|---|
| aucune des deux | 25/65 = 38,5 % |
| une seule | 8/14 = 57,1 % |
| les deux | 31/33 = **93,9 %** |

L'accord porte donc sur trois exécutions réparties sur deux régimes. Le procédé
ne se réduit pas à exécuter deux fois la même chaîne : l'épreuve C, où une seule
confirmation indépendante est disponible, ne montre aucun effet (46,4 % avec
elle, 47,6 % sans).

**La justification préalable déplace le plafond.** Une phrase demandée au modèle
avant les étiquettes, plutôt qu'après, porte la cascade de 47,5 à **70,0 %**
d'étiquettes acceptées, pour un surcoût de 8 %. L'écart est significatif
(z = 2,04, p = 0,041). L'ordre des propriétés du schéma de sortie décide de tout :
produite avant le choix, la phrase le construit ; produite après, elle ne fait
que le rationaliser. C'est le seul des réglages éprouvés qui franchisse le
plafond de 47 à 55 % sur lequel les autres butaient.

**Les régimes éprouvés.** Tous portent sur les mêmes 169 articles, avec la même
liste de 567 étiquettes et une température nulle.

| Régime | Script | Étiq./art. | Justesse |
|---|---|---|---|
| groupage par lot de 25 | `classify_iptc_mistral_batched.py` | 2,18 | 47,8 % |
| un appel par article | `controle_appel_unitaire.py` | 3,21 | — |
| un appel par article, seconde reprise | `controle_appel_unitaire.py --temoin` | 3,15 | — |
| cascade commerciale | `controle_cascade_mistral.py` | 2,47 | 47,5 % |
| unitaire + justification | `controle_justification_mistral.py --mode unitaire` | 2,93 | 59,0 % |
| cascade + justification | `controle_justification_mistral.py --mode cascade` | 2,13 | **70,0 %** |
| local, liste entière | `classify_iptc_ollama.py` | 4,50 | non jugée |
| local, cascade | `classify_iptc_ollama_cascade.py` | 2,36 | 25,0 % |
| local, cascade + justification | `classify_iptc_ollama_cascade.py --justification` | 1,96 | non jugée |

Deux reprises du régime unitaire, à invite et à modèle identiques, ne rendent le
même jeu d'étiquettes que dans 34,9 % des cas, pour un recouvrement de Jaccard
moyen de 0,631. Toute mesure porte donc sur une exécution particulière.

Les régimes locaux tournent sur un MacBook Air M1 doté de 8 Go de mémoire vive.
Cette capacité borne ce qui peut y être exécuté : le dernier régime a perdu
10 articles sur 169 par expiration du serveur local.

### Refaire les épreuves

```bash
cd classification
python3 verification_etiquettes.py            # fabrique les épreuves A et B
python3 verification_unitaire.py              # fabrique l'épreuve C (graine 2)
python3 verification_cascades.py              # fabrique l'épreuve D (graine 3)
python3 verification_justification.py         # fabrique l'épreuve E (graine 4)
streamlit run app_verification.py             # annoter, une réponse enregistrée à la fois
python3 verification_etiquettes.py --depouiller
python3 verification_justification.py --depouiller
```

Les graines sont fixées : les épreuves se régénèrent à l'identique. Les réponses
sont enregistrées par identifiant d'item plutôt que par rang, de sorte qu'une
épreuve refabriquée ne les désaligne pas.

## `topic-modeling/` : les méthodes qui font émerger leurs catégories

Les deux premières méthodes comparées par le mémoire, la modélisation probabiliste
et le regroupement de plongements, avec les 111 exécutions de la campagne. Chaque
run conserve ses paramètres exacts et ses mesures, de sorte que les balayages que
les annexes du mémoire ne reprennent pas se relisent ici. Voir
`topic-modeling/README.md`, section « Les blocs renvoyés au dépôt ».

## `analyse_couts/` : estimer le coût sans dépenser un centime

Ces scripts ne font aucun appel API : ils réutilisent les fonctions d'extraction et de tokenisation de `classification/classify_iptc_mistral_batched.py` pour mesurer/estimer les coûts avant de lancer un vrai run.

| Script | Rôle | Sortie |
|---|---|---|
| `analyze_corpus_tokens.py` | Statistiques mots/tokens sur tout le corpus (par article, par fascicule). À lancer en premier. | `stats_par_article.csv`, `stats_resume.csv` |
| `estimate_pricing.py` | Coût du scénario 1-article/appel : décompose chaque appel en 3 composantes (prompt système, liste IPTC, texte article), bornes de sortie min/moyen/max, comparaison entre modèles. | `cout_par_article.csv`, `cout_par_fascicule.csv`, `cout_resume.csv`, `cout_comparaison_modeles.csv` |
| `estimate_pricing_batched.py` | Coût du groupage par lot : reproduit l'algorithme de composition des lots pour prédire exactement quels articles seraient groupés ensemble, et leur coût. | `lots_composition.csv`, `cout_par_lot.csv`, `cout_batched_resume.csv` |
| `estimate_pricing_cascade.py` | Coût de la cascade : distribution des branches niveau 2 dérivée des articles réellement déjà classés (pas une hypothèse uniforme), extrapolée à tout le corpus. | `cout_cascade_par_branche.csv`, `cout_cascade_resume.csv` |

Toutes ces sorties sont écrites dans ce même dossier `analyse_couts/`, à côté des scripts qui les produisent.

```bash
cd analyse_couts
python3 analyze_corpus_tokens.py       # à lancer en premier (produit stats_par_article.csv)
python3 estimate_pricing.py
python3 estimate_pricing_batched.py
python3 estimate_pricing_cascade.py
```

Les scripts de `analyse_couts/` produisent ces chiffres et les écrivent dans les CSV du même dossier. Chaque script porte en tête l'objet de sa mesure et ses hypothèses.

**Le tokeniseur conditionne les chiffres.** Les comptes de jetons viennent de `mistral-common`, épinglé dans le `requirements.txt`. En son absence, `count_tokens` se rabat sur une estimation par caractères sans avertir : la liste des étiquettes y pèse 4 821 jetons au lieu de 8 775 et le coût annoncé tombe de moitié. Comme les trois scripts d'estimation lisent le relevé produit par `analyze_corpus_tokens.py`, il faut relancer celui-ci après toute installation du tokeniseur, sans quoi les comptes fixes et les comptes par article viendraient de deux régimes différents.

## `results/` : sorties de classification

- `feuilles_mistral/`, `feuilles_mistral_cached/`, `feuilles_mistral_batched/`, `feuilles_mistral_cascade/` : sorties de classification par fascicule, une par variante de script
- `sample_review.json` : échantillon utilisé pour la relecture qualité manuelle

## `re-ocr/` : ré-OCRisation et mesure de qualité

### `re-ocr/scripts/` : corriger l'OCR d'origine

| Script | Rôle |
|---|---|
| `reocr_mistral.py` | Pour chaque bloc ALTO d'une page : crop l'image JP2/JPG correspondante, envoie le crop à Mistral (`pixtral-12b`) en vision, remplace le texte ALTO par la transcription Mistral. Usage : `python reocr_mistral.py --key VOTRE_CLE [--fascicule ID] [--page N] [--dry-run]`. |
| `reocr_mistral.ipynb` | Version notebook (Google Colab) du script ci-dessus. |
| `reocr_pero.ipynb` | Ré-OCRisation via le moteur **Pero** (notebook Colab, GPU requis) : alternative à Mistral pour comparer la qualité entre moteurs. |
| `reocr_pero_image.ipynb` | Variante de `reocr_pero.ipynb` travaillant directement sur les images plutôt que sur l'ALTO existant. |

### `re-ocr/metriques/` : mesurer la qualité de l'OCR

| Script | Rôle | Sortie |
|---|---|---|
| `stats_pages.py` | Par page et par source (BnF original / Pero / Mistral) : tokens distincts, nb de mots, nb de caractères, nb d'articles (depuis le TOC/METS) — pour comparer objectivement les 3 sources. Usage : `python3 stats_pages.py [--n 10]`. | `resultats_stats/stats_par_page.csv`, `stats_mistral_brut.csv`, `vocab_venn.html` |
| `ocrqa/impresso_ocrqa.py` | Score de qualité OCR (ratio de mots reconnus, méthode **impresso**) sur les sources présentes. Usage : `python3 impresso_ocrqa.py [--n 5]`. | `ocrqa/resultats_ocrqa/ocrqa_results.json`, `ocrqa_summary.csv`, `ocrqa_report.html` |

**Les ALTO produits par Pero ne sont pas versés.** Le moteur a servi d'étape vers le modèle multimodal, et la comparaison à trois sources est rapportée dans le mémoire pour la démarche qu'elle documente plutôt que comme une expérience à rejouer. `stats_pages.py` s'arrête sur un message qui le dit ; `impresso_ocrqa.py` signale la source absente et poursuit sur les deux autres. Les relevés obtenus à trois sources restent versés dans `resultats_stats/` et `ocrqa/resultats_ocrqa/`.

### `re-ocr/corpus/` : structure du corpus

100 fascicules (numéros de périodique) numérisés par la BnF et diffusés sur Gallica, identifiés par leur cote NUMM/ARK (ex. `4109000`). Suivi via **Git LFS** (`.gitattributes`, ~790 Mo), sans les images. Deux versions du même corpus : `original/` (OCR BnF d'origine) et `reocr_mistral/` (texte corrigé par Mistral vision). Chaque fascicule est un dossier contenant :

| Fichier / dossier | Format | Contenu |
|---|---|---|
| `toc/T<id>.xml` | METS | Table des matières structurelle du fascicule : structMap logique avec un div par article (`ARTICLE`), chacun référençant les blocs de texte ALTO qui le composent (`area BEGIN="PAG_x_TBxxxxxx"`). C'est ce découpage à l'article qui permet à `classification/` d'extraire le texte article par article. |
| `ocr/X000000N.xml` | ALTO | Une page, un fichier. Texte reconnu (TextBlock/TextLine/String) avec coordonnées. Dans `original/`, texte OCR BnF d'origine ; dans `reocr_mistral/`, texte remplacé bloc par bloc par la transcription Mistral vision. |
| `manifest.xml` | METS | Métadonnées techniques et administratives de la numérisation (issues du pipeline BnF), distinctes du contenu structurel de `toc/`. |
| `etatDocument<id>.xml` | XML interne BnF | Métadonnées de suivi : identifiant ARK, cote, format source (JP2), indicateurs qualité (présence de fichiers ALTO, DAISY, etc). |
| `logs/X000000N.json` (`reocr_mistral/` uniquement) | JSON | Un fichier par page : détail des appels à l'API Mistral vision effectués pour cette page, bloc par bloc. Utile pour le débogage et le suivi de coût de la ré-OCRisation. |

Lien avec `classification/` : les scripts de `classification/` lisent leurs articles directement depuis ce corpus, via `MISTRAL_RESULTS_DIR` (pointe vers `re-ocr/corpus/reocr_mistral/`) et `SAMPLE_DIR` (pointe vers `re-ocr/corpus/original/`), définis en tête de chaque script.

Point d'attention : le découpage à l'article présent dans `toc/T<id>.xml` est une caractéristique de ce corpus, constitué pour ce POC dans un contexte de reprise de données. Il n'est pas présent tel quel sur l'ensemble du corpus presse patrimoniale de la BnF. Réutiliser `classification/` sur un autre corpus suppose de disposer du même type de découpage à l'article, ou de le reconstruire au préalable.

---

## Ce qui n'est pas inclus volontairement

- **Les images** (JP2/JPG des pages scannées) : ~19 Go pour 100 fascicules, seuls les fichiers texte (ALTO/METS) sont inclus.
- **La clé API** (`config/.env`) : à créer soi-même.
- **Les kits d'évaluation d'un second annotateur** : non versés tant que la personne concernée n'a pas donné son accord sur ses réponses.

Cette arborescence est déjà en place dans ce dépôt (corpus texte inclus, sans les images) ; les scripts tournent directement une fois la clé API renseignée dans `config/.env`.
