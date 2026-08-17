"""
comparer_classla_mistral.py : deux classifications, deux niveaux du référentiel

CE QUE FAIT CE SCRIPT

Confronte les prédictions de l'encodeur affiné, qui décide entre les dix-sept
catégories de premier niveau de l'IPTC, à celles obtenues par modèle de langue
contre les 567 entrées de troisième niveau.

Les deux dispositifs opèrent sur les mêmes articles à deux niveaux du même
arbre. Leur confrontation demande une remontée : chaque étiquette de troisième
niveau est ramenée à son ancêtre de premier niveau par la relation broader du
schéma officiel, jusqu'à atteindre l'une des dix-sept racines.

Aucune vérité de référence n'existe pour ce corpus. L'accord mesuré ici ne
prouve donc pas la justesse : deux méthodes peuvent se tromper ensemble. Il
mesure la convergence de deux dispositifs sans rapport de principe, ce qui reste
le seul contrôle disponible.

LES QUATRE QUANTITÉS RAPPORTÉES

  Accord large     part des articles pour lesquels au moins une des étiquettes du
                 modèle de langue tombe sous la catégorie choisie par
                 l'encodeur. Le modèle de langue en rendant de une à cinq, le
                 critère est généreux par construction.

  Accord strict    même chose pour la seule première étiquette rendue, que le
                 modèle place en tête.

  Témoin           les étiquettes de l'encodeur sont permutées entre articles,
                 puis l'accord est recalculé. La permutation conserve les deux
                 distributions et détruit l'appariement. Le résultat dit ce que
                 vaudrait l'accord entre deux classifications sans rapport, ce
                 qui rend le chiffre principal interprétable.

  Repliement       nombre d'étiquettes de troisième niveau que chaque catégorie de
                 premier niveau absorbe. Il chiffre ce que le grain de
                 l'encodeur rend indistinct.

L'accord est également ventilé selon le nombre d'étiquettes rendues, qui le fait
croître mécaniquement, et selon la probabilité de l'encodeur, qui dit si les
désaccords se concentrent sur ses décisions hésitantes.

ENTRÉES

  resultats/{dossier_classla}/predictions.json   sorties de classla_iptc.py
  ../results/feuilles_mistral_batched/*.json     sorties de la classification
                                                 au troisième niveau
  ../classification/iptc_mediatopic_official.json  taxonomie officielle, format
                                                 SKOS, employée pour la remontée

SORTIE

  Un rapport imprimé sur la sortie standard. Le rediriger vers un fichier pour
  le conserver auprès du run.

PAQUETS EMPLOYÉS

  argparse, json, random, collections, pathlib   bibliothèque standard

Aucune dépendance extérieure n'est nécessaire, le script ne faisant que croiser
des fichiers déjà produits.

USAGE

    python3 comparer_classla_mistral.py resultats/classla_mistral_article_5361_...
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ICI = Path(__file__).resolve().parent


def _trouver_depot() -> Path:
    """Racine du dépôt de classification, cherchée en remontant l'arborescence.
    Deux dispositions sont admises, comme dans lda_mallet_corpus : celle du
    dossier de travail et celle du dépôt, où ces scripts vivent dans
    topic-modeling/ à côté de classification/."""
    for base in [ICI, *ICI.parents]:
        for relatif in (Path("github") / "classification-iptc", Path(".")):
            if (base / relatif / "classification" /
                    "iptc_mediatopic_official.json").is_file():
                return (base / relatif).resolve()
    raise SystemExit("dépôt de classification introuvable depuis " + str(ICI))


GITHUB = _trouver_depot()
SCHEMA = GITHUB / "classification" / "iptc_mediatopic_official.json"
FEUILLES = GITHUB / "results" / "feuilles_mistral_batched"
# Graine du tirage de permutation. Fixée pour que le témoin soit reproductible :
# deux exécutions du script rendent la même valeur.
GRAINE = 20260816

# Nombre de permutations du témoin. Cinquante suffisent à stabiliser la moyenne
# à moins d'un point de pourcentage, le tirage portant sur plusieurs milliers
# d'articles.
N_PERMUTATIONS = 50


# Le modèle de CLASSLA rend les intitulés anglais de premier niveau, à une
# exception près : « politics » là où le schéma actuel écrit « politics and
# government ». Sans cette correspondance, les articles politiques, catégorie la
# plus fournie du corpus, sont tous comptés en désaccord.
ALIAS = {"politics": "politics and government"}


def schema() -> tuple[dict[str, str | None], dict[str, str], dict[str, str]]:
    """Lit la taxonomie officielle et rend trois tables de correspondance.

    Les trois tables sont, dans l'ordre :
      code d'un concept vers le code de son ancêtre de premier niveau,
      code d'un concept vers son intitulé français,
      intitulé anglais d'une racine vers son code.

    La troisième est construite sur les seules dix-sept racines. Quatre
    intitulés anglais servent en effet à deux concepts dans les profondeurs du
    schéma, et les prendre tous exposerait à ce qu'un concept fin écrase une
    racine dans le dictionnaire.

    La fonction interne remonter parcourt la relation broader jusqu'à une
    racine. L'ensemble vu protège d'une boucle : le schéma est un arbre, mais
    rien dans le format ne l'impose, et une référence circulaire ferait tourner
    la boucle indéfiniment.

    Un concept dépourvu d'intitulé français reçoit son intitulé anglais. Cinq
    concepts sont dans ce cas sur les 390 employés par le corpus.
    """
    d = json.loads(SCHEMA.read_text(encoding="utf-8"))
    concepts = {c["qcode"].removeprefix("medtop:"): c for c in d["conceptSet"]}
    racines = {u.rsplit("/", 1)[-1] for u in d["hasTopConcept"]}
    label_fr, par_en = {}, {}
    for code, c in concepts.items():
        p = c["prefLabel"]
        label_fr[code] = p.get("fr") or p.get("en-GB", code)
        if code in racines and p.get("en-GB"):
            par_en[p["en-GB"]] = code
    for rendu, officiel in ALIAS.items():
        if officiel in par_en:
            par_en[rendu] = par_en[officiel]

    def remonter(code: str) -> str | None:
        """Remonte de proche en proche jusqu'à la racine dont le code dépend.

        Rend None lorsque la chaîne s'interrompt, ce qui arrive si un concept
        n'a pas de parent déclaré ou si le schéma contient une boucle.
        """
        vu = set()
        while code and code not in racines and code not in vu:
            vu.add(code)
            b = concepts.get(code, {}).get("broader")
            if not b:
                return None
            code = b[0].rsplit("/", 1)[-1]
        return code if code in racines else None

    return {k: remonter(k) for k in concepts}, label_fr, par_en


def charger_mistral() -> dict[str, list[dict]]:
    """Lit les classifications de troisième niveau, un fichier par fascicule.

    La clé du dictionnaire rendu est l'identifiant complet de l'article, formé
    du numéro de fascicule et de l'identifiant de division séparés par deux
    points. Ce format est celui qu'emploie lda_mallet_corpus, ce qui permet de
    croiser les deux jeux sans conversion.

    Les articles sans thème sont écartés : ils proviennent d'un appel qui a
    échoué, et les compter fausserait les dénombrements.
    """
    out: dict[str, list[dict]] = {}
    for f in sorted(FEUILLES.glob("*_themes.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        for a in d["articles"]:
            if a.get("themes"):
                out[f"{d['fascicule']}:{a['article_id']}"] = a["themes"]
    return out


def main() -> None:
    """Croise les deux classifications et imprime le rapport.

    Le traitement procède ainsi.

    1. Lire les deux jeux de prédictions et la taxonomie.
    2. Vérifier que tous les intitulés rendus par l'encodeur trouvent leur
       racine. Un intitulé sans racine serait compté en désaccord sans que rien
       ne le signale, et l'anomalie arrête donc le script.
    3. Pour chaque article commun, retenir la catégorie de l'encodeur et
       l'ensemble des ancêtres des étiquettes du modèle de langue.
    4. Calculer les deux taux d'accord, puis le témoin de permutation.
    5. Ventiler l'accord selon le nombre d'étiquettes et selon la probabilité.
    6. Dresser la table des désaccords, le repliement et les deux
       distributions.
    """
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dossier", type=Path, help="sortie de classla_iptc.py")
    args = p.parse_args()

    preds = {x["doc_id"]: x for x in json.loads(
        (args.dossier / "predictions.json").read_text(encoding="utf-8"))}
    mistral = charger_mistral()
    ancetre, label_fr, par_en = schema()

    communs = sorted(set(preds) & set(mistral))
    print(f"{len(preds)} articles classés par l'encodeur au premier niveau,")
    print(f"{len(mistral)} par le modèle de langue au troisième,")
    print(f"{len(communs)} en commun.\n")
    if not communs:
        return

    # Un intitulé rendu par l'encodeur qui ne trouve pas sa racine serait compté
    # en désaccord sans que rien ne le signale. L'anomalie arrête le script.
    inconnus = sorted({preds[k]["etiquette"] for k in communs
                       if preds[k]["etiquette"] not in par_en})
    if inconnus:
        raise SystemExit("intitulés de l'encodeur sans racine dans le schéma, "
                         "compléter ALIAS : " + ", ".join(inconnus))

    # Pour chaque article : catégorie de premier niveau choisie par l'encodeur,
    # et ancêtres de premier niveau des étiquettes du modèle de langue.
    enc: dict[str, str | None] = {}
    llm_tous: dict[str, set[str]] = {}
    llm_premier: dict[str, str | None] = {}
    sans_ancetre = 0
    for k in communs:
        enc[k] = par_en.get(preds[k]["etiquette"])
        racines = {ancetre.get(t["code"]) for t in mistral[k]} - {None}
        llm_tous[k] = racines
        llm_premier[k] = ancetre.get(mistral[k][0]["code"])
        if not racines:
            sans_ancetre += 1
    retenus = [k for k in communs if llm_tous[k]]
    n = len(retenus)

    large = sum(1 for k in retenus if enc[k] in llm_tous[k])
    strict = sum(1 for k in retenus if enc[k] is not None
                 and enc[k] == llm_premier[k])

    # Témoin : les étiquettes de l'encodeur sont permutées entre articles, ce qui
    # conserve les deux distributions et détruit l'appariement.
    rng = random.Random(GRAINE)
    valeurs = [enc[k] for k in retenus]
    tirages = []
    for _ in range(N_PERMUTATIONS):
        melange = valeurs[:]
        rng.shuffle(melange)
        tirages.append(sum(1 for k, v in zip(retenus, melange) if v in llm_tous[k]) / n)
    hasard = sum(tirages) / len(tirages)

    print(f"Accord sur {n} articles")
    print(f"  large  (une étiquette du modèle de langue au moins sous la "
          f"catégorie de l'encodeur) : {large:5} / {n}  = {large/n:.1%}")
    print(f"  strict (la première étiquette du modèle de langue)              "
          f"  : {strict:5} / {n}  = {strict/n:.1%}")
    print(f"  témoin de permutation, {N_PERMUTATIONS} tirages                 "
          f"  : {hasard:.1%}")
    if sans_ancetre:
        print(f"  {sans_ancetre} articles écartés, ancêtre introuvable")

    # L'accord large croît mécaniquement avec le nombre d'étiquettes rendues.
    print(f"\n  selon le nombre d'étiquettes rendues par le modèle de langue")
    par_n = defaultdict(lambda: [0, 0])
    for k in retenus:
        c = par_n[len(mistral[k])]
        c[1] += 1
        if enc[k] in llm_tous[k]:
            c[0] += 1
    for m in sorted(par_n):
        j, t = par_n[m]
        print(f"    {m} étiquette(s) : {j:5}/{t:5} = {j/t:6.1%}")

    # Et selon l'assurance de l'encodeur.
    print(f"\n  selon la probabilité rendue par l'encodeur")
    seuils = [(0.0, 0.5), (0.5, 0.9), (0.9, 0.99), (0.99, 1.01)]
    for lo, hi in seuils:
        sel = [k for k in retenus if lo <= preds[k]["score"] < hi]
        if sel:
            j = sum(1 for k in sel if enc[k] in llm_tous[k])
            print(f"    [{lo:.2f} ; {hi:.2f}[ : {j:5}/{len(sel):5} = "
                  f"{j/len(sel):6.1%}")

    # ── Désaccords ────────────────────────────────────────────────────────────
    des = Counter()
    for k in retenus:
        if enc[k] not in llm_tous[k]:
            des[(label_fr.get(enc[k], preds[k]["etiquette"]),
                 " / ".join(sorted(label_fr.get(r, r) for r in llm_tous[k])))] += 1
    print(f"\nDésaccords les plus fréquents ({sum(des.values())} au total)")
    print(f"  {'encodeur, niveau 1':38} {'modèle de langue, remonté':46} {'n':>4}")
    for (a, b), v in des.most_common(15):
        print(f"  {a[:38]:38} {b[:46]:46} {v:4}")

    # ── Repliement ────────────────────────────────────────────────────────────
    absorbe: dict[str, set[str]] = defaultdict(set)
    for k in mistral:
        for t in mistral[k]:
            r = ancetre.get(t["code"])
            if r:
                absorbe[r].add(t["label_fr"])
    total = sum(len(s) for s in absorbe.values())
    print(f"\nRepliement : étiquettes de troisième niveau absorbées par chaque "
          f"catégorie de premier,\nsur les {len(mistral)} articles classés")
    print(f"  {'catégorie de premier niveau':46} {'niveau 3':>9}")
    for r, s in sorted(absorbe.items(), key=lambda x: -len(x[1])):
        print(f"  {label_fr.get(r, r)[:46]:46} {len(s):9}")
    print(f"  {'total':46} {total:9}")
    print(f"\n  {len(absorbe)} des 17 catégories de premier niveau employées, "
          f"{total} étiquettes de troisième niveau distinctes,")
    print(f"  soit {total/len(absorbe):.1f} par catégorie en moyenne, "
          f"{max(len(s) for s in absorbe.values())} au maximum.")

    # ── Distributions comparées ───────────────────────────────────────────────
    c_enc = Counter(label_fr.get(enc[k], "?") for k in retenus)
    c_llm = Counter()
    for k in retenus:
        for r in llm_tous[k]:
            c_llm[label_fr.get(r, r)] += 1
    print(f"\nDistribution sur les {n} articles retenus "
          f"(l'encodeur rend une catégorie, le modèle de langue peut en toucher "
          f"plusieurs)")
    print(f"  {'catégorie':46} {'encodeur':>9} {'modèle de langue':>18}")
    for lab, v in c_enc.most_common():
        print(f"  {lab[:46]:46} {v:9} {c_llm.get(lab, 0):18}")
    absentes = [lab for lab in c_llm if lab not in c_enc]
    for lab in sorted(absentes, key=lambda x: -c_llm[x]):
        print(f"  {lab[:46]:46} {0:9} {c_llm[lab]:18}")


if __name__ == "__main__":
    main()
