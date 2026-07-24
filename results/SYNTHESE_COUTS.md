# Synthèse — coûts de tokenisation et de classification IPTC

**Projet** : classification thématique multi-label (1 à 5 étiquettes IPTC Media Topics niveau 3) d'articles de presse ancienne (BnF/Gallica, OCR corrigé par Mistral), via l'API Mistral.

**Échantillon mesuré** : 100 fascicules, 6699 articles retenus (≥ 10 mots), extraits et tokenisés avec le tokenizer Mistral réel (`mistral-common`, pas une approximation en caractères).

**Sources** : tous les chiffres de ce document sont lus directement dans les CSV du dossier `themes/results/` (générés par `analyze_corpus_tokens.py`, `estimate_pricing.py` et `estimate_pricing_batched.py`) — aucun n'est recalculé à la main.

**Objet de ce document** : ce document accompagne la fin d'un stage/mémoire et est destiné à guider un éventuel déploiement à plus grande échelle — par une équipe interne ou un prestataire technique qui reprendrait ce travail. Il documente les choix testés, ceux qui fonctionnent, ceux qui ont été essayés et rejetés (section 8), et ce qui reste à vérifier avant toute mise en production (section 9 et ci-dessous).

**⚠️ Portée réelle de ce POC — à ne pas généraliser tel quel** : cette fonctionnalité de classification thématique a été pensée dans le contexte d'un projet de **reprise de données**, et suppose en entrée des fichiers **déjà découpés à l'article** (structure logique METS/TOC avec des divs `ARTICLE` — voir section 1). **Ce découpage à l'article n'existe pas sur l'ensemble de la presse patrimoniale** : il est spécifique aux corpus qui l'ont produit lors d'un traitement particulier, pas une caractéristique générale des fascicules numérisés par la BnF.

Les estimations de coût de ce document (notamment l'extrapolation à 2,6M fascicules, section 6) ont été calculées **à l'échelle de la presse patrimoniale prise comme ordre de grandeur de référence** — ce n'est pas le corpus cible réel. **Pour une application au projet Réforme (ou tout autre corpus cible réel disposant du découpage à l'article), il faut refaire les estimations** à partir de deux chiffres propres à ce corpus, pas ceux de ce document :
1. le nombre réel de fascicules du corpus cible ;
2. le nombre total de tokens représenté par le texte des articles à classer dans ce corpus (mesurable avec `analyze_corpus_tokens.py` une fois le corpus disponible, comme cela a été fait ici pour l'échantillon de 100 fascicules — section 2).

---

## Résumé exécutif

- **Quoi** : classifier chaque article d'un fascicule de presse ancienne avec 1 à 5 thèmes IPTC Media Topics (niveau 3), via l'API Mistral, à partir du texte OCR déjà corrigé.
- **Approche recommandée** : **groupage par lot** (`classify_iptc_mistral_batched.py`) — regrouper plusieurs articles par appel plutôt qu'un seul, pour partager le coût fixe de la liste d'étiquettes. C'est de loin le levier le plus rentable (-89 % vs 1 article/appel), largement testé et documenté (sections 4 à 6).
- **Ce qui a été essayé et écarté** : découper la classification en 2 étages (niveau 2 puis niveau 3 par branche) pour réduire encore la taille de la liste — ça coûte en réalité plus cher, pas moins (section 8). Gardé dans le dépôt à titre d'exemple documenté.
- **Budget** : ~$3,63 pour l'échantillon de 100 fascicules (6699 articles, mistral-large-latest, groupage, sans cache) ; extrapolé linéairement à ~$94 000 pour 2,6M fascicules (section 6) — extrapolation à prendre comme ordre de grandeur, pas une prévision ferme (l'échantillon vient d'un seul titre/période).
- **Hébergement local (GPU)** : potentiellement moins cher à grande échelle, mais basé sur des benchmarks publiés, pas une mesure réelle sur ce prompt — protocole de vérification en section 11, pas encore exécuté.

## ⚠️ Risque principal non résolu : la qualité de classification n'a jamais été validée

**Aucune vérité terrain n'existe pour ce projet.** Tous les chiffres de ce document mesurent le **coût** et le **volume** (tokens, appels, temps) — pas la **justesse** des thèmes assignés par le modèle. Une interface de relecture humaine existe (`app_annotation_batched.py`, Streamlit, en local — voir le dépôt de travail, pas inclus dans le GitHub public) mais n'a été utilisée que pour explorer manuellement quelques articles, jamais pour produire un échantillon de vérité terrain statistiquement exploitable.

Concrètement, avant tout déploiement à l'échelle, il manque :
- Un échantillon annoté à la main (même quelques centaines d'articles) pour mesurer un vrai taux de précision/rappel.
- Une vérification que le groupage par lot (plusieurs articles dans un même appel) ne dégrade pas la qualité par rapport à 1 article/appel — risque théorique connu ("lost in the middle"), jamais mesuré ici (voir l'échange sur ce sujet et les sources académiques trouvées, aucune ne teste directement notre cas).
- Une comparaison de qualité entre modèles (mistral-large vs les modèles moins chers recommandés section 5) — rien ne garantit qu'un modèle 5x moins cher classe aussi bien.

**Ne pas interpréter les recommandations de coût de ce document comme une garantie de qualité.** Elles répondent uniquement à la question « combien ça coûte », pas « est-ce que c'est fiable ».

## Ressources associées

- **Code** : `[À COMPLÉTER — URL du dépôt GitHub une fois publié]`
- **Mémoire** : `[À COMPLÉTER — référence/chapitre du mémoire une fois soutenu]`
- **Contact** : `[À COMPLÉTER — personne à contacter en cas de question après la fin du stage]`

## Conformité et licences

`[À COMPLÉTER]` — à documenter avant tout déploiement en production : conditions de réutilisation des données Gallica/BnF pour le corpus source, licence de la taxonomie IPTC Media Topics (CC BY 4.0, déjà vérifiée lors de sa récupération — voir section sur `iptc_mediatopic_official.json`), conditions d'usage de l'API Mistral (et de tout modèle tiers utilisé en remplacement, ex. Qwen — licence Apache 2.0 déjà vérifiée section 11.1).

---

## 1. Composition d'un appel

Chaque appel de classification est composé de 3 blocs :

| Composante | Tokens | Nature |
|---|---:|---|
| Prompt système (instructions) | 137 | fixe, identique à chaque appel |
| Liste des 567 étiquettes IPTC (niveau 3 + niveau 2 terminales) | 8 775 | fixe, identique à chaque appel |
| **Overhead fixe total** | **8 912** | payé à chaque appel en mode 1-article/appel |
| Texte de l'article | 13 → 23 119 (moy. 522, médiane 200) | variable, propre à chaque article |

La liste IPTC pèse **94,5 %** du volume d'entrée total du corpus (63,2M tokens sur 100 fascicules, 1 article/appel) — c'est le facteur dominant de tous les calculs qui suivent.

*Source : `stats_par_article.csv`, `stats_resume.csv`.*

## 2. Statistiques du corpus échantillonné (100 fascicules)

| Niveau | Mesure | Min | Max | Moyenne | Médiane |
|---|---|---:|---:|---:|---:|
| Article | mots | 10 | 9 867 | 315,2 | 114 |
| Article | tokens | 13 | 23 119 | 521,8 | 200 |
| Fascicule | nb d'articles | 4 | 202 | 67,0 | 54,5 |
| Fascicule | tokens (total articles) | 6 335 | 100 686 | 34 954,7 | 32 123,5 |

*Source : `stats_resume.csv`.*

## 3. Tokens de sortie (complétion) — bornes réelles

Le schéma JSON strict garantit 1 à 5 thèmes par article ⇒ les bornes sont calculées exactement (pas une hypothèse) :

| Mode | Min (1 thème) | Moyen* | Max (5 thèmes) |
|---|---:|---:|---:|
| 1 article/appel | 15 tokens | 31 tokens | 55 tokens |
| Groupage par lot (par article, avec `article_id` répété) | ~27 tokens | ~47 tokens | ~67 tokens |

\* Le "moyen" en 1-article/appel est **calibré sur un run réel** (930 tokens de sortie / 30 articles). Le "moyen" en groupage est un point milieu min/max — pas encore calibré sur un run réel groupé.

*Source : `cout_par_article.csv`, `cout_par_lot.csv` (colonnes `tokens_output_*`).*

## 4. Coût total — échantillon (100 fascicules, mistral-large-latest)

| Scénario | Nb d'appels | Sans cache (min/moyen/max) | Avec cache (min/moyen/max) |
|---|---:|---|---|
| **1 article/appel** | 6 699 | $31,75 / $31,91 / $32,15 | $16,87 / $17,03 / $17,27 |
| **Groupage par lot** (budget 40k tokens, 25 articles max/lot) | 316 | $3,43 / $3,63 / $3,83 | $2,73 / $2,93 / $3,13 |

**Le groupage réduit le coût de ~89 %** (moyenne, sans cache : $31,91 → $3,63) — c'est le levier de loin le plus puissant, bien plus que le cache seul (-47 %). Combiner les deux (groupage + cache) donne $2,93, un gain marginal supplémentaire modeste par rapport au groupage seul.

*Source : `cout_batched_resume.csv`.*

### Répartition du coût par composante (1 article/appel, sans cache)

| Composante | Coût | Part |
|---|---:|---:|
| Liste IPTC | $29,39 | 92,1 % |
| Texte des articles | $1,75 | 5,5 % |
| Prompt système | $0,46 | 1,4 % |
| Complétion (sortie) | $0,31 | 1,0 % |

*Source : `cout_resume.csv`, sortie console de `estimate_pricing.py`.*

## 5. Comparaison entre modèles (échantillon, 100 fascicules)

| Modèle | $/M in | $/M out | 1 article/appel (sans cache) | 1 article/appel (avec cache) | Groupage par lot (sans cache) | Groupage par lot (avec cache) |
|---|---:|---:|---:|---:|---:|---:|
| mistral-large-latest | 0,50 | 1,50 | $31,91 | $17,03 | $3,63 | $2,93 |
| mistral-medium-latest | 1,50 | 7,50 | $96,35 | $51,70 | $11,85 | $9,74 |
| mistral-small-latest | 0,15 | 0,60 | $9,60 | $5,14 | $1,14 | $0,93 |
| ministral-8b-latest | 0,15 | 0,15 | $9,51 | $5,05 | $0,99 | $0,78 |
| ministral-3b-latest | 0,10 | 0,10 | $6,34 | $3,36 | $0,66 | $0,52 |

*Source : `cout_comparaison_modeles.csv` (1-article/appel) ; calcul équivalent appliqué à `cout_par_lot.csv` pour le groupage.*

## 6. Extrapolation à 2,6 millions de fascicules (×26 000)

⚠️ **Extrapolation linéaire** à partir de 100 fascicules d'un seul titre/période — ordre de grandeur, pas une prévision budgétaire fiable pour un corpus réellement hétérogène (titres, époques, densités d'articles très différentes).

⚠️ **Ce chiffre de 2,6M fascicules est celui de la presse patrimoniale prise comme échelle de référence, pas le corpus cible réel** (voir l'avertissement en introduction) — ce POC suppose des fichiers déjà découpés à l'article, une structure absente de la majorité de la presse patrimoniale. Pour un corpus cible réel (ex. projet Réforme), refaire ce calcul avec son nombre de fascicules et son volume de tokens propres, pas ceux-ci.

| Scénario | Sans cache | Avec cache |
|---|---:|---:|
| 1 article/appel (mistral-large) | $829 660 (min $825 500 / max $835 900) | $442 780 |
| Groupage par lot (mistral-large) | $94 380 (min $89 180 / max $99 580) | $76 180 |
| Groupage par lot (ministral-8b) | $25 740 | $20 280 |
| Groupage par lot (ministral-3b) | $17 160 | $13 520 |
| Cascade (mistral-large, voir section 8) | $119 210 (min $114 660 / max $123 760) | $70 330 (min $65 780 / max $74 880) |

~174 millions d'articles, ~8,2 millions de lots (vs 174 millions d'appels en 1-article/appel) ; ~14 millions d'appels pour la cascade (538 × 26 000).

La cascade reste plus chère que le groupage simple à cette échelle aussi (~$119 210 contre ~$94 380 sans cache) — le mécanisme identifié en section 8 (texte d'article dupliqué) ne s'améliore pas avec le volume, il est proportionnel au nombre d'articles quelle que soit l'échelle.

*Source : `cout_batched_resume.csv` et `cout_cascade_resume.csv` × 26 000 ; méthode identique appliquée par modèle.*

## 7. Hébergement local (GPU cloud, modèle open source)

Structure de coût différente : **$/heure de GPU loué**, pas $/token. Avec mise en cache de préfixe automatique (gratuite, pas à -90 % comme en cloud), le volume de calcul réel se limite à : préfixe fixe (payé une seule fois) + texte des articles + sortie.

**Débit GPU — sources réelles (pas une estimation à l'aveugle)** :
- RTX 4090, vLLM, DeepSeek-R1-Distill-Qwen-7B, mode offline/batch : **4 173 tokens/s** agrégé — [DatabaseMart](https://www.databasemart.com/blog/vllm-gpu-benchmark-rtx4090)
- RTX 4090, vLLM, DeepSeek-R1-Distill-Llama-8B, mode offline/batch : **2 769 tokens/s** agrégé — [DatabaseMart](https://www.databasemart.com/blog/vllm-gpu-benchmark-rtx4090)
- RTX 4090, vLLM, Qwen3-Coder-30B (modèle plus gros que notre cible), serving haute concurrence : **2 259 tokens/s** agrégé — [CloudRift](https://www.cloudrift.ai/gpu-benchmarks)
- ⚠️ Ne pas confondre avec le débit *par utilisateur unique* (single-request, non pertinent pour du traitement par lot) : seulement 78-140 tokens/s pour Llama 3 8B — bien plus bas, mais mesure une toute autre chose (latence conversationnelle, pas débit de traitement en masse).

| Scénario (100 fascicules) | Tokens réels à traiter | Coût (débit bas 2 259 tok/s → haut 4 173 tok/s) |
|---|---:|---|
| 1 article/appel | 3 712 052 (moyen) | $0,10 → $0,18 |
| Groupage par lot | 3 822 090 (moyen) | $0,10 → $0,19 |

**Résultat contre-intuitif : le groupage n'apporte quasi rien en local** (+3 % de volume à cause de la répétition d'`article_id`, sans rien économiser puisque le préfixe est déjà gratuit après le 1er appel). Le levier du groupage est spécifique à la facturation cloud par token.

À l'échelle de 2,6M fascicules : **$2 570 à $4 890** selon le débit réel (au lieu de $442 780-829 660 en cloud avec mistral-large).

\* Hypothèse restante non mesurée : le prix GPU (~$0,40/h, RTX 4090 — ordre de grandeur observé $0,29-0,59/h). Le débit, lui, s'appuie maintenant sur des benchmarks réels ci-dessus, pas une estimation générique. Ne comprend pas le temps d'ingénierie, le risque qualité d'un modèle self-hosted, ni les frais de démarrage/téléchargement des poids.

## 8. Piste explorée et rejetée : pipeline en cascade (niveau 2 → niveau 3)

**Idée** : puisque la liste IPTC pèse 92 % du coût (section 4), la découper en 2 étages pour réduire sa taille à chaque appel :
- **Étage 1** : tous les articles classés parmi les 120 catégories **niveau 2** seulement (liste ~4,3x plus petite que les 567 complètes) — par lots, comme le groupage simple.
- **Regroupement** : les articles sont réunis par branche niveau 2 assignée (mélangés entre fascicules, pour maximiser la taille des lots par branche).
- **Étage 2** : pour les 68 branches qui ont des enfants niveau 3 (515 répartis très inégalement, 1 à 111 par branche, médiane 4), un 2ᵉ passage classe en niveau 3 avec une liste réduite aux seuls enfants de CETTE branche. Les 52 branches sans enfant (3,5 % des articles, mesuré sur 285 articles déjà classés) s'arrêtent à l'étage 1.

Script : `classify_iptc_mistral_cascade.py` (fonctionnel, testé réellement sur 30 articles). Estimation : `estimate_pricing_cascade.py`, basée sur la **distribution réelle des branches** (dérivée des 285 articles déjà classés, pas une hypothèse uniforme).

**Résultat : la cascade coûte PLUS cher que le groupage simple, pas moins.**

| Scénario | Appels | Coût (sans cache) |
|---|---:|---:|
| Groupage simple | 316 | **$3,63** |
| Cascade (`MAX_BATCH_SIZE=25`, comme le groupage) | 538 | $4,60 |
| Cascade (plafond de lot relevé au maximum permis par le budget de tokens) | 198 | $4,38 (plancher) |

**Cause identifiée** : réduire la liste ne réduit pas le texte des articles à envoyer — et la cascade **envoie ce texte deux fois** (étage 1 puis étage 2, pour 96,5 % des articles), contre une seule fois en groupage simple :

| | Groupage simple | Cascade |
|---|---:|---:|
| Tokens de texte d'article facturés (corpus entier) | 3,50M | 6,87M (**×1,96**) |

Le texte d'article ne peut jamais être mis en cache (il change à chaque article) — doubler son volume mange une bonne partie de l'économie faite sur la liste. Même constat en local (GPU) : la cascade demande **1,94x plus de calcul réel** (7,42M contre 3,82M tokens), donc ~2x plus cher et plus long, peu importe le modèle de facturation.

**Conclusion** : l'idée était logique sur le papier (la liste dominait le coût), mais elle ignorait le coût de la duplication du texte. Elle ne vaudrait le coup que pour des articles très courts par rapport à la taille de la liste — l'inverse de notre corpus (522 tokens/article en moyenne, contre une liste déjà réduite à ~2000 tokens à l'étage 1).

**À l'échelle de 2,6M fascicules** (voir section 6) : ~$119 210 sans cache / ~$70 330 avec cache pour la cascade, contre ~$94 380 / $76 180 pour le groupage simple — l'écart ne se résorbe pas avec le volume, il est structurel (proportionnel au nombre d'articles, pas à l'échelle du corpus).

*Source : `cout_cascade_par_branche.csv`, `cout_cascade_resume.csv`.*

## 9. Ce qui est mesuré vs ce qui est une hypothèse

| Élément | Statut |
|---|---|
| Tokens du prompt système, de la liste IPTC, des articles | **Mesuré** (tokenizer Mistral réel) |
| Bornes min/max de la sortie JSON | **Calculé** (schéma strict, structure exacte) |
| Tarifs par modèle | **Officiels** (mistral.ai/pricing/api/ pour Mistral ; sources tierces recoupées pour les autres) |
| Tokens de sortie "moyen" (1-article/appel) | Calibré sur **1 run réel** (30 articles) |
| Taux de cache (55,4 %) | Calibré sur **1 run réel** (30 articles) — peut varier selon le rythme réel des appels |
| Tokens de sortie "moyen" en groupage | **Estimation** (point milieu min/max, pas de run réel groupé) |
| Débit GPU local (2259-4173 tokens/s) | **Sourcé** (benchmarks vLLM réels RTX 4090, modèles 7-30B — DatabaseMart, CloudRift) — mais pas mesuré sur notre prompt exact |
| Prix de location GPU (~$0,40/h) | Ordre de grandeur observé ($0,29-0,59/h RTX 4090) — pas un contrat/devis réel |
| Extrapolation à 2,6M fascicules | **Hypothèse forte** : suppose la composition de l'échantillon (longueur/densité d'articles) représentative de tout le corpus visé |

## 10. Recommandations

1. **Le groupage par lot est le levier le plus rentable en cloud** (-89 % à lui seul) — largement devant le cache de prompt (-47 %), le choix d'un modèle moins cher, ou la cascade niveau 2 → niveau 3 (qui coûte plus cher, voir section 8) — et sans compromis connu sur la fenêtre de contexte (marge x5 même dans le pire cas).
2. **Un modèle plus petit (Ministral 8B/3B) reste pertinent** pour une tâche de classification fermée (567 étiquettes, schéma strict) — mais sans vérité terrain, la qualité relative n'est pas vérifiée (voir échange sur la relecture d'échantillon).
3. **Ne pas découper la classification en étages pour économiser des tokens** : ça semble intuitif (la liste domine le coût) mais le texte des articles, dupliqué à chaque étage supplémentaire, mange l'économie — vérifié en cloud ET en local (section 8).
4. **L'hébergement local ne devient intéressant que si le motif est le contrôle/la confidentialité**, pas le prix à ce volume — à ce volume, le cloud groupé (mistral-large, ~$94 380 pour 2,6M fascicules) et le local (~$2 570-4 890 hors ingénierie) ont un écart réel, mais le vrai enjeu est le temps d'ingénierie et le risque qualité, pas le $/token.
5. **Avant tout passage à l'échelle**, fiabiliser : (a) le taux de cache réel sur un vrai run groupé, (b) la qualité du groupage (risque "lost in the middle" non vérifié), (c) la représentativité de l'échantillon de 100 fascicules pour l'extrapolation à 2,6M.
6. **Avant de se fier aux chiffres d'hébergement local (section 7)**, les vérifier par un vrai test sur un petit échantillon (GPU cloud + vLLM, protocole en section 11) — les débits utilisés (2 259-4 173 tokens/s) viennent de benchmarks publiés sur d'autres prompts, pas d'une mesure sur le nôtre (préfixe énorme + texte court + sortie minuscule, un profil différent de la plupart des benchmarks génériques).

## 11. Passer à la pratique : tester réellement l'hébergement local

Toute la section 7 (coût local) repose sur des **benchmarks publiés**, pas une mesure sur notre prompt exact. Voici ce qu'il faut pour transformer l'estimation en vraie mesure — ce qui peut être préparé à l'avance, et ce qui demande une action directe (compte, paiement, déploiement).

### 11.1 Choix du modèle et du tokenizer

| | Ministral 8B (`mistralai/Ministral-8B-Instruct-2410`) | Qwen3-8B |
|---|---|---|
| Licence | Mistral Research License — **usage non-commercial uniquement** | **Apache 2.0** — totalement libre, y compris commercial |
| Tokenizer utilisé jusqu'ici | Déjà en place dans le pipeline (`mistral-common`, téléchargé depuis HF) — c'est lui qui sert à `count_tokens()` dans tous nos scripts | Différent (vocabulaire BPE propre à Qwen) — nécessite `transformers` : `AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")` |
| Débit RTX 4090 déjà sourcé | Comparable à `DeepSeek-R1-Distill-Llama-8B` : **2 769 tokens/s** agrégé ([DatabaseMart](https://www.databasemart.com/blog/vllm-gpu-benchmark-rtx4090)) | Les 2 benchmarks sourcés en section 7 sont déjà des modèles à architecture Qwen : `DeepSeek-R1-Distill-Qwen-7B` (**4 173 tokens/s**) et `Qwen3-Coder-30B` (**2 259 tokens/s**, plus gros que notre cible) — un Qwen3-8B natif tomberait vraisemblablement dans cette fourchette |
| Structured output (schéma JSON strict + `enum`) | Supporté par vLLM, indépendamment du modèle | Idem — c'est une fonctionnalité de vLLM, pas du modèle |

**Recommandation** : Qwen3-8B est le choix le plus simple pour un test — licence sans ambiguïté, et débit déjà documenté par nos propres sources. Ministral 8B reste cohérent avec le tokenizer déjà utilisé partout ailleurs dans le pipeline, au prix d'un léger doute sur la licence pour un usage hors recherche stricte.

*Sources vérifiées le jour de la rédaction : licences Hugging Face des modèles concernés.*

### 11.2 vLLM — le serveur d'inférence

**Ce que c'est** : un serveur d'inférence LLM qui expose une API compatible OpenAI (`/v1/chat/completions`), avec 2 fonctionnalités essentielles pour notre cas :
- **Cache de préfixe automatique** (`--enable-prefix-caching`) — sans ce flag, le serveur ne réutilise pas le calcul du préfixe fixe répété (system + liste IPTC) d'un appel à l'autre, ce qui invaliderait la comparaison avec le cloud.
- **Sortie structurée avec schéma JSON strict + `enum`** (`response_format: {"type": "json_schema", ...}`) — confirmé supporté nativement, le même format qu'on envoie déjà à l'API Mistral cloud ([doc officielle vLLM](https://docs.vllm.ai/en/latest/features/structured_outputs/)). Nos scripts n'auraient donc quasiment rien à changer côté schéma.

**Commande de lancement type** (Docker, GPU unique) :
```bash
docker run --gpus all -p 8000:8000 \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen3-8B \
  --enable-prefix-caching \
  --max-model-len 32768
```
Le serveur écoute ensuite sur `http://<ip-du-pod>:8000/v1/chat/completions` — il suffit de remplacer `MISTRAL_URL` par cette adresse dans une variante du script (sans clé API Mistral).

### 11.3 GPU cloud — Vast.ai et RunPod.io

Les deux fournisseurs déjà identifiés (section 7) pour louer une RTX 4090 à l'heure (~$0,29-0,59/h). Ce que ça demande concrètement, **à faire soi-même** (création de compte et paiement ne sont pas des actions que je peux effectuer) :

1. **Créer un compte** sur [vast.ai](https://vast.ai) ou [runpod.io](https://runpod.io), ajouter un moyen de paiement.
2. **Choisir une offre GPU** : filtrer sur RTX 4090, comparer les prix/hôtes (Vast.ai) ou choisir un GPU Cloud/Community Cloud (RunPod).
3. **Choisir un template** : les deux plateformes proposent des images prêtes à l'emploi (dont vLLM) — sinon utiliser l'image Docker `vllm/vllm-openai` directement avec la commande ci-dessus.
4. **Lancer le pod**, attendre le téléchargement du modèle (quelques minutes selon la taille, ~16 Go pour un 8B).
5. **Récupérer l'URL/IP exposée** (port 8000 ou celui configuré) — c'est ce qu'il faut ensuite fournir pour brancher le script de classification dessus.
6. **Ne pas oublier d'arrêter le pod** à la fin du test — la facturation est à l'heure, y compris à l'arrêt si le pod n'est pas explicitement terminé (selon la plateforme).

Une fois le pod lancé et l'URL obtenue, le test lui-même consiste à relancer le pipeline (par ex. sur les 5-6 fascicules déjà utilisés en cloud pour comparaison directe) contre ce endpoint, et à comparer le temps réel mesuré (le script mesure déjà `temps_reponse_s` par appel) au coût GPU-heure effectivement facturé — donnant un chiffre mesuré à la place de l'estimation par benchmark de la section 7.
