# Évaluation humaine des thèmes — protocole

Généré le 15 août 2026 par `jugement_humain.py`. Soixante items, vingt thèmes
tirés de chacun des trois modèles retenus, mélangés et anonymisés.

## Pourquoi

Le classement des méthodes repose jusqu'ici sur le NPMI, mesure automatique de
cohérence dont Hoyle et ses coauteurs ont montré qu'elle s'accorde mal au
jugement des lecteurs. Cette évaluation donne un point d'ancrage humain. Elle
suit le protocole d'intrusion de mot de Chang et ses coauteurs, qui est la
référence du domaine, et y ajoute une question de dénomination qui répond
directement à la critique de l'étiquetage développée au chapitre 2.

## Ce qu'il y a dans le dossier

| Fichier | Contenu |
|---|---|
| `items.csv` | les 60 items à annoter, trois colonnes à remplir |
| `themes_complets.csv` | les dix mots de tête de chaque item, pour la question 2 |
| `cle.csv` | **à n'ouvrir qu'après annotation** : modèle d'origine et intrus |

## Comment annoter

Ouvrez `items.csv` dans un tableur. Chaque ligne présente six mots. Trois
colonnes sont à remplir.

**`intrus_designe`** — Cinq de ces six mots viennent d'un même thème, le sixième
d'un autre. Écrivez le mot qui vous paraît étranger aux cinq autres, ou son
numéro de colonne. Répondez même quand vous hésitez : une réponse au hasard fait
partie de la mesure. Ne laissez vide que si vous sautez l'item.

**`nommable`** — Ouvrez `themes_complets.csv` à la même ligne, qui donne les dix
mots du thème. Sauriez-vous nommer ce thème pour un lecteur de la plateforme ?
Répondez `oui`, `approx` ou `non`.

**`nom_propose`** — Si vous avez répondu `oui` ou `approx`, écrivez l'intitulé.
Deux ou trois mots suffisent.

## Règles à tenir

N'ouvrez pas `cle.csv` avant d'avoir terminé : la connaissance du modèle
d'origine invaliderait la comparaison.

Annotez d'une traite si possible, ou notez où vous vous êtes arrêtée. La fatigue
se répartit sur les trois modèles puisque les items sont mélangés, mais une
interruption longue change la manière de juger.

Ne cherchez pas les mots dans le corpus. La question est de savoir si le thème
se lit tel qu'il est présenté.

## Deux annotateurs valent mieux qu'un

Si quelqu'un accepte de faire la même chose, copiez `items.csv` sous
`items_annotateur2.csv` avant de commencer. L'accord entre les deux jugements
donnera une mesure de la difficulté de la tâche elle-même, indépendante des
modèles, et c'est le protocole que le chapitre 3 emploiera pour l'évaluation
contre le vocabulaire contrôlé.

## Dépouillement

```bash
python3 jugement_humain.py --depouiller
```

Le script rapporte, par modèle, la part d'intrus correctement désignés et la
part de thèmes jugés nommables. Le hasard donne 16,7 %, soit une chance sur six.
Un modèle dont les thèmes sont cohérents doit dépasser nettement ce seuil.

## Ce que les résultats permettront de dire

Si l'ordre des trois modèles reproduit celui du NPMI, les mesures automatiques
se trouvent validées sur ce corpus et la réserve de Hoyle peut être citée sans
qu'elle entame les conclusions.

Si l'ordre diffère, c'est le classement automatique qui devient suspect, et le
chapitre devra le dire. Les deux issues sont utiles, ce qui est la marque d'une
expérience bien posée.

La question de dénomination répond pour sa part à une affirmation du texte : le
thème résiduel de l'allocation de Dirichlet latente devrait recueillir une
majorité de `non`, quand les rubriques nettes du regroupement par densité
devraient recueillir des `oui`. Si tel n'est pas le cas, c'est la critique de
l'étiquetage qui demande à être nuancée.
