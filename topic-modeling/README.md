# Modélisation thématique et regroupement de plongements

Ce dossier contient le code et les mesures de la troisième partie du mémoire
*La classification thématique automatique de la presse patrimoniale numérisée*.
Trois familles de méthodes y sont comparées sur le même corpus de cent
fascicules : la modélisation thématique probabiliste, le regroupement de
plongements de phrases, et la classification contre le vocabulaire contrôlé
IPTC, dont le code se trouve dans `../classification/`.

Le corpus employé est celui du dépôt, `../re-ocr/corpus/`, dans ses deux états :
le texte hérité de l'océrisation d'origine et le texte corrigé par
ré-océrisation.

## Prérequis

Python 3.11 ou plus récent. Les dépendances figurent dans le `requirements.txt`
à la racine du dépôt, section `topic-modeling/`.

```bash
pip install -r ../requirements.txt
```

MALLET n'est pas un paquet Python et n'est pas redistribué ici. Télécharger la
distribution 2021-08 depuis <https://mimno.github.io/Mallet/> et la déplier dans
`topic-modeling/Mallet-202108/`. Java 8 ou plus récent est nécessaire pour
l'exécuter.

## Arborescence

```
topic-modeling/
├── lda_mallet_corpus.py        constitution du corpus et runs MALLET
├── metrics_lda.py              NPMI, diversité, entropie, Jaccard
├── corpus_reference.py         corpus de référence filtré (voir plus bas)
├── reference_brute.py          corpus de référence non filtré, employé
├── recap_runs.py               tableau récapitulatif de tous les runs
├── bertopic_corpus.py          plongements, UMAP, k-means ou HDBSCAN
├── jugement_humain.py          fabrication et dépouillement des kits
├── app_jugement.py             interface Streamlit d'annotation
├── classla_iptc.py             classifieur affiné, niveau 1 de l'IPTC
├── comparer_classla_mistral.py croisement des deux classifications
├── matrices_confusion.py       cartes de chaleur des deux confrontations
├── projections_clusters.py     nuages de points des regroupements
├── figures_chapitre3.py        coûts, couverture et distributions
├── figures_comparaison.py      balayage de K et dépense à l'échelle
├── pages_thematiques.py        pages de presse coloriées par thème
├── composition_periode.py      composition thématique par période
├── stopwords_fr_presse.txt     listes d'arrêt constituées pour ce corpus
├── stopwords_extra.txt
├── stoplocs_fr_presse.txt
├── RESULTATS-CAMPAGNE.md       relevé de tous les runs
├── jugement/GRILLE.md          protocole d'évaluation humaine
└── resultats/                  paramètres, thèmes et métriques de 111 runs
```

## Le corpus et ses unités

`lda_mallet_corpus.py` constitue le corpus pour toutes les méthodes. Les autres
scripts l'importent, de sorte que les trois familles reçoivent les mêmes
documents.

Deux sources sont disponibles, `bnf` pour le texte hérité et `mistral` pour le
texte ré-océrisé. Deux granularités sont disponibles, `article` et `bloc`.
L'article est reconstitué depuis la carte logique du fichier METS, qui donne
l'ordre des blocs. Le bloc vient de la reconnaissance de mise en page.

La carte logique type également 62 808 divisions comme paragraphes. Le contrôle
des renvois montre que chacune pointe vers un bloc et un seul. Le paragraphe ne
constitue donc pas une troisième granularité.

Le seuil de longueur minimale vaut 20 mots pour l'article et 5 pour le bloc. Les
mots de moins de trois caractères et ceux dont la fréquence est inférieure à 5
sont retirés dans la configuration retenue.

## Modélisation thématique probabiliste

```bash
python3 lda_mallet_corpus.py --source mistral --granularite article \
        --k 20 --iterations 2000 --graine 1 --freq-min 5 --longueur-min 3
```

Les options couvrent l'ensemble de l'expérience : source, granularité, nombre de
thèmes, itérations, graine, répétitions, optimisation des hyperparamètres,
filtrage par fréquence, par longueur et par catégorie grammaticale.

Chaque run écrit dans `resultats/` un dossier contenant `meta.json` avec les
paramètres exacts, `topics.json` avec les thèmes et leurs mots de tête,
`span_topic.json` avec l'affectation des documents et `training_data.txt` avec
le corpus prétraité. Les deux derniers ne sont pas versés dans ce dépôt, étant
volumineux et régénérables.

Les métriques se calculent ensuite :

```bash
python3 metrics_lda.py resultats/<nom_du_run>
```

## Regroupement de plongements de phrases

```bash
python3 bertopic_corpus.py --regroupement kmeans  --k 20 --modele e5
python3 bertopic_corpus.py --regroupement hdbscan --min-taille 20 --modele e5
```

Trois modèles de plongement sont éprouvés : `minilm` et `camembert`, bornés à
128 jetons, et `e5`, borné à 512. Les vecteurs sont mis en cache dans
`embeddings/` avec un manifeste d'identifiants, ce qui permet de faire varier
l'algorithme de regroupement sans recalculer la représentation. Le cache n'est
pas versé ici.

Les options `--voisins`, `--dimensions` et `--sans-umap` règlent la réduction de
dimension.

## Le corpus de référence des mesures

Le NPMI et la diversité se calculent sur un corpus de référence qui fournit les
dénombrements de cooccurrence. Deux constructions figurent dans ce dossier.

`corpus_reference.py` produit une référence normalisée comme le corpus
d'entraînement, listes d'arrêt appliquées et seuil de fréquence de 5. Un mot de
tête absent de cette référence reçoit un NPMI de −1 contre tous les autres mots
de son thème, faute de probabilité estimable. Les configurations entraînées sous
d'autres filtres s'en trouvent pénalisées par effet de plancher.

`reference_brute.py` produit une référence qui ne retire rien : ni mots vides,
ni formes rares, ni formes courtes. Elle compte 83 682 formes contre 21 355 pour
la précédente, et couvre le vocabulaire de tous les runs. **C'est elle qui
fournit les valeurs rapportées dans le mémoire.** Les mots vides qu'elle contient
n'affectent pas les mesures, le dénombrement des cooccurrences ne portant que
sur le vocabulaire des thèmes évalués.

```bash
python3 reference_brute.py              # construit les quatre références
python3 reference_brute.py --mesurer    # recalcule le NPMI de tous les runs
```

Le fichier `metrics_brut.json` de chaque run contient la valeur retenue, ainsi
que le nombre de mots de tête qui manquaient à la référence filtrée.

## Évaluation humaine

Le classement des méthodes repose sur des mesures automatiques dont la
littérature signale qu'elles s'accordent mal au jugement des lecteurs. Une
évaluation à la main a été conduite selon le protocole d'intrusion de mot de
Chang et de ses coauteurs, complété d'une question de dénomination.

```bash
python3 jugement_humain.py               # fabrique les kits
streamlit run app_jugement.py            # interface d'annotation
python3 jugement_humain.py --depouiller  # calcule les scores
```

Les kits se régénèrent à l'identique, la graine étant fixée dans le script. Les
réponses recueillies ne sont pas versées ici. Le protocole complet figure dans
`jugement/GRILLE.md`.

## Classification par un modèle affiné

`classla_iptc.py` applique au corpus le classifieur multilingue diffusé par
CLASSLA, obtenu par affinage de `xlm-roberta-large`. Il prédit les dix-sept
catégories du premier niveau de l'IPTC.

```bash
python3 classla_iptc.py --limite 20    # essai
python3 classla_iptc.py                # tout le sous-corpus ré-océrisé
```

`comparer_classla_mistral.py` croise ces prédictions avec celles obtenues au
troisième niveau par les scripts de `../classification/`. La confrontation
demande une remontée dans l'arbre IPTC par la relation `broader` du schéma
officiel.

```bash
python3 comparer_classla_mistral.py resultats/<dossier_classla>
```

Le script rapporte l'accord entre les deux classifications, un témoin de
permutation qui donne la valeur de cet accord sans appariement, et le nombre
d'étiquettes de troisième niveau que chaque catégorie de premier niveau absorbe.

## Les figures

Deux scripts produisent les figures reprises dans le mémoire.

`matrices_confusion.py` écrit deux cartes de chaleur en code TikZ. La première
croise les catégories du classifieur affiné avec les ancêtres de premier niveau
des étiquettes du modèle génératif. La seconde croise les thèmes de la
modélisation probabiliste avec les groupes du regroupement de plongements, et
donne l'information mutuelle normalisée entre les deux partitions.

`projections_clusters.py` réduit les plongements à deux dimensions et trace quatre
nuages de points, répartis en trois figures : les groupes du partitionnement en
k moyennes, ceux du
regroupement par densité avec les articles écartés en gris, le corpus à la
granularité du bloc colorié par la cohérence des groupes, et les catégories du
vocabulaire contrôlé sur la projection des articles. Les blocs reçoivent leur
propre projection, qui ne se superpose pas à celle des articles. Le script rapporte aussi trois
statistiques calculées dans l'espace des plongements complet, qui vérifient ce
que les figures donnent à voir.

`figures_chapitre3.py` produit les quatre figures de la classification contre le
vocabulaire contrôlé : le coût par article selon le nombre d'articles portés par
un appel, la couverture du référentiel branche par branche, les deux
distributions sur base comparable et la distribution des étiquettes par rang de
fréquence. Le script reconstruit la liste des 567 étiquettes en appelant la
fonction du script de classification qui la fabrique, de sorte que les deux
définitions restent solidaires.

`pages_thematiques.py` dessine des pages du corpus à leurs coordonnées réelles,
un rectangle par bloc de texte. Une première figure remplit chaque bloc selon le
poids d'un thème, une seconde donne à chaque article la couleur de sa catégorie
IPTC et son intitulé. L'option `--verifier` confronte l'étendue d'un thème au
découpage en articles du METS.

`figures_comparaison.py` produit deux figures dont l'objet traverse les
chapitres : les balayages du nombre de groupes des deux familles sur un même axe,
et la dépense des trois méthodes à l'échelle de la plateforme, en deux panneaux
puisque les unités diffèrent.

`composition_periode.py` agrège les affectations par tranche de vingt ans. Les
dates de publication sont lues dans le manifeste de chaque fascicule et
recoupées, avec `--controle`, avec le relevé des adresses Gallica. Deux
tracés sont disponibles, sur les mêmes valeurs. `--figure` donne la composition
en aires empilées, qui montre un tout dont les parts se répartissent. `--petits`
donne un graphique par thème, à échelle commune et rangés par période de
maximum, qui répond à l'autre question, celle de l'apparition et de la
disparition d'une matière : une aire empilée ne laisse lire commodément que la
bande du bas, seule à reposer sur une ligne de base droite. Les deux portent en
panneau supérieur l'effectif de chaque tranche, que la normalisation à cent
effacerait, alors que la première repose sur 37 articles et l'avant-dernière
sur 1 299.

```bash
python3 matrices_confusion.py
python3 projections_clusters.py             # cache la projection au premier appel
python3 projections_clusters.py --recalculer
python3 figures_chapitre3.py
python3 figures_comparaison.py
python3 pages_thematiques.py
python3 composition_periode.py --controle
python3 composition_periode.py --figure
python3 composition_periode.py --petits
```

La projection d'affichage va vers deux dimensions, quand celle du regroupement
va vers cinq. Les groupes dessinés sont ceux qu'a formés la chaîne, et leurs
positions sur le plan sont propres à la figure.

Deux de ces figures demandent des fichiers qui ne sont pas versés ici, le cache
de plongements pour les nuages de points et l'affectation des documents pour la
matrice des deux familles. Les scripts le signalent et donnent la commande qui
les reconstitue. Les autres figures se produisent depuis le dépôt seul.

Les sorties elles-mêmes ne sont pas versées. Elles se régénèrent par les
commandes ci-dessus, et les figures définitives figurent dans le mémoire.

## Les mesures versées

`resultats/` contient 111 runs. Quatre fichiers par run sont conservés :

| Fichier | Contenu |
|---|---|
| `meta.json` | paramètres exacts, date, durée |
| `topics.json` | thèmes et mots de tête |
| `metrics_brut.json` | métriques sur la référence non filtrée |
| `metrics_ref.json` | métriques sur la référence filtrée, pour mémoire |

Le tableau récapitulatif se régénère par `recap_runs.py`, et son état figure
dans `RESULTATS-CAMPAGNE.md`.

## Les blocs renvoyés au dépôt

Neuf blocs ont été retirés des annexes du mémoire parce que ce dépôt les tient
sous une forme plus complète, et que le corps du texte n'en cite aucun. Ils se
retrouvent comme suit.

| Retiré des annexes | Où le trouver ici |
|---|---|
| Quatre entraînements sans graine fixée | `resultats/*_g[1-4]_*/metrics_ref.json` |
| Effet de la réduction de dimension | runs `bertopic_*` dont `meta.json` fait varier `n_neighbors` et `n_components` |
| Balayage du nombre de groupes | runs `bertopic_kmeans_*_mt*`, un par valeur de K |
| Comparaison des trois modèles de plongement | runs `*_e5_*`, `*_camembert_*`, `*_multilingual_*` |
| Regroupement par densité, relevé | runs `bertopic_hdbscan_*`, champ `min_cluster_size` |
| Effet de la borne des modèles de plongement | `meta.json`, champ `max_seq_length` |
| Tableau des modèles de plongement éprouvés | `meta.json`, champ `embedding_model` |
| Relevé de chronométrage du calcul local | `meta.json`, champ `duree_s`, récapitulé par `recap_runs.py` |
| Schéma de la chaîne de traitement | `../classification/schema_pipeline_classification.svg` |

Le dixième bloc retiré, la dérivation formelle de l'allocation de Dirichlet
latente, ne figure pas ici : elle relève d'un manuel, et le lecteur qui la
cherche la trouvera chez Blei, Ng et Jordan.

Le mémoire conserve en annexe, à l'inverse, ce dont ce dépôt ne saurait tenir
lieu : les deux incidents de méthode, les pièces citées par le corps du
texte, et les résultats qualitatifs — les dix thèmes de la configuration
retenue, les thèmes occupés par des débris d'océrisation, les matrices de
confusion.

## Ce qui n'est pas versé

Les caches de plongements, les corpus de référence, l'affectation des documents
et les corpus prétraités sont régénérables par les scripts ci-dessus. Ils
représentent environ 1,3 Go et ne figurent pas ici.

Les runs de juillet, conduits sur un fascicule unique et sans graine fixée, ne
figurent pas davantage. Le mémoire établit que la variance entre exécutions y
dépasse les écarts mesurés, ce qui leur retire toute valeur conclusive.
