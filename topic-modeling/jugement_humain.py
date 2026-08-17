"""
jugement_humain.py : prépare et dépouille l'évaluation humaine des thèmes

POURQUOI CE SCRIPT

Le classement des méthodes repose sur le NPMI, mesure automatique de cohérence.
Alexander Hoyle et ses coauteurs ont montré en 2021 que ces mesures s'accordent
mal au jugement des lecteurs dès qu'on les compare dans des conditions
contrôlées. Une évaluation à la main est donc nécessaire, faute de quoi le
chapitre reposerait sur un instrument que son propre exposé déclare suspect.

LES DEUX TÂCHES

  1. INTRUSION. Six mots sont présentés, dont cinq viennent d'un même thème et un
     d'un autre. L'annotateur désigne l'intrus. Un thème cohérent le rend
     visible ; un thème incohérent ramène la réponse au hasard, soit une chance
     sur six. Le protocole est celui de Jonathan Chang et de ses coauteurs, qui
     est la référence du domaine.

  2. DÉNOMINATION. Les dix mots de tête étant cette fois donnés, l'annotateur dit
     s'il saurait nommer le thème pour un lecteur de la plateforme : oui,
     approximativement, ou non. Cette tâche ne figure pas dans le protocole
     d'origine. Elle a été ajoutée parce qu'elle répond directement à la critique
     de l'étiquetage développée au chapitre 2.

L'ORDRE DES DEUX TÂCHES IMPORTE

L'intrus est par construction absent des dix mots de tête du thème visé.
Présenter les dix mots à côté des six donnerait donc la réponse de la première
tâche. Les deux tâches sont écrites dans deux fichiers distincts, numérotés,
et les consignes demandent de terminer le premier avant d'ouvrir le second.

ANONYMAT DES ITEMS

Les soixante items sont mélangés et ne portent aucune indication de leur modèle
d'origine. La clé reste dans un fichier séparé, à n'ouvrir qu'après annotation :
savoir qu'un item vient de telle méthode influencerait le jugement.

ENTRÉES

  resultats/{run}/topics.json    pour chacun des trois modèles retenus, dont les
                                 chemins sont déclarés dans MODELES ci-dessous

SORTIES

  jugement/cle.csv                     modèle d'origine et intrus de chaque item
  jugement/kit_annotateur_{n}/1_intrusion.csv
  jugement/kit_annotateur_{n}/2_denomination.csv
  jugement/reponses/                   écrit par app_jugement.py

PAQUETS EMPLOYÉS

  argparse, csv, json, random, pathlib   bibliothèque standard

Aucune dépendance extérieure n'est nécessaire. L'interface d'annotation, qui
emploie Streamlit, se trouve dans app_jugement.py.

USAGE

    python3 jugement_humain.py               # produit le matériel
    python3 jugement_humain.py --depouiller  # calcule les scores
"""

import argparse
import csv
import json
import random
from pathlib import Path

ICI = Path(__file__).resolve().parent
SORTIE = ICI / "jugement"

# Graine du tirage. Fixée pour que les kits se régénèrent à l'identique : le
# dépôt n'a pas besoin de les contenir, et une contestation sur un item peut
# toujours être vérifiée en rejouant le script.
GRAINE = 20260815

# Les trois configurations retenues, à leur meilleur réglage.
MODELES = {
    "A": ("resultats/lda_corpus_mistral_article_k20_g1_f5_l3_20260814_215342",
          "LDA, article, K=20"),
    "B": ("resultats/bertopic_kmeans_mistral_article_k20_e5_g1_20260815_143200",
          "BERTopic k-means, article, K=20, e5"),
    "C": ("resultats/bertopic_hdbscan_mistral_article_mt20_e5_g1_20260815_161345",
          "BERTopic HDBSCAN, article, taille min. 20, e5"),
}
# Nombre de mots du thème présentés à côté de l'intrus. Cinq est la valeur du
# protocole de Chang, qui fixe le hasard à une chance sur six.
N_MOTS = 5

# Profondeur des mots de tête retenus pour décrire un thème. Dix est la valeur
# qu'emploient les travaux du domaine, ce qui rend les résultats comparables à
# la littérature.
N_TOP = 10

# Nombre de thèmes tirés par modèle. Vingt par modèle donnent soixante items,
# soit une séance d'une vingtaine de minutes. Au-delà, la fatigue de
# l'annotateur devient elle-même une variable.
MAX_THEMES = 20


def charger(dossier: Path) -> list[dict]:
    """Lit les thèmes d'un run et rend leurs dix mots de tête.

    Les thèmes vides sont écartés. Le regroupement par densité en produit
    parfois, lorsqu'un groupe trop petit ne reçoit aucun mot descriptif.
    """
    tp = json.loads((dossier / "topics.json").read_text(encoding="utf-8"))
    return [{"topic_id": t["topic_id"], "size": t.get("size", 0),
             "mots": t["top_words"].split()[:N_TOP]} for t in tp if t["top_words"].strip()]


def fabriquer(rng: random.Random) -> tuple[list[dict], list[dict]]:
    """Compose les soixante items et la clé qui leur correspond.

    Pour chaque thème retenu, la fonction procède ainsi.

    1. Tirer cinq mots parmi les dix de tête.
    2. Choisir un intrus dans un autre thème du même modèle. Le candidat doit
       être absent du vocabulaire de tête du thème visé, faute de quoi sa
       présence tiendrait à un recouvrement réel entre les deux thèmes plutôt
       qu'au hasard du tirage, et la tâche perdrait son sens.
    3. Mélanger les six mots, afin que la position de l'intrus ne le trahisse
       pas.

    Les items de tous les modèles sont ensuite mélangés entre eux et numérotés.
    Le mélange répartit la fatigue de l'annotateur sur les trois modèles, de
    sorte qu'elle n'en avantage aucun.

    Rend deux listes de même longueur et de même ordre : les items à annoter et
    la clé, qui porte le modèle d'origine, l'intrus et sa position.
    """
    items, cle = [], []
    for lettre, (chemin, libelle) in MODELES.items():
        d = ICI / chemin
        if not d.is_dir():
            print(f"  dossier absent, modèle ignoré : {chemin}")
            continue
        themes = charger(d)
        if len(themes) > MAX_THEMES:
            themes = rng.sample(themes, MAX_THEMES)
        for t in themes:
            # Cinq mots du thème, pris parmi ses dix premiers.
            retenus = rng.sample(t["mots"], min(N_MOTS, len(t["mots"])))
            # L'intrus vient d'un autre thème du même modèle, et n'appartient
            # pas au vocabulaire de tête du thème visé : sa présence doit tenir
            # au hasard du tirage plutôt qu'à un recouvrement réel.
            autres = [u for u in themes if u["topic_id"] != t["topic_id"]]
            candidats = [m for u in autres for m in u["mots"][:5]
                         if m not in t["mots"]]
            if not candidats:
                continue
            intrus = rng.choice(candidats)
            propose = retenus + [intrus]
            rng.shuffle(propose)
            items.append({"item": None, "mots": propose,
                          "dix_mots": " ".join(t["mots"])})
            cle.append({"item": None, "modele": lettre, "libelle": libelle,
                        "topic_id": t["topic_id"], "taille": t["size"],
                        "intrus": intrus, "position": propose.index(intrus) + 1})
    ordre = list(range(len(items)))
    rng.shuffle(ordre)
    items = [items[i] for i in ordre]
    cle = [cle[i] for i in ordre]
    for n, (it, c) in enumerate(zip(items, cle), 1):
        it["item"] = c["item"] = n
    return items, cle


def ecrire(items: list[dict], cle: list[dict], nb_annotateurs: int = 2) -> None:
    """Deux tâches, deux fichiers, dans cet ordre.

    L'intrus est par construction absent des dix mots de tête du thème visé.
    Présenter les dix mots à côté des six donnerait donc la réponse : la
    dénomination doit être faite après l'intrusion, dans un fichier distinct."""
    SORTIE.mkdir(exist_ok=True)
    (SORTIE / "cle.csv").write_text("", encoding="utf-8")
    with (SORTIE / "cle.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cle[0].keys()))
        w.writeheader()
        w.writerows(cle)

    for n in range(1, nb_annotateurs + 1):
        kit = SORTIE / f"kit_annotateur_{n}"
        kit.mkdir(exist_ok=True)
        with (kit / "1_intrusion.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["item", "mot_1", "mot_2", "mot_3", "mot_4", "mot_5",
                        "mot_6", "intrus_designe"])
            for it in items:
                w.writerow([it["item"], *it["mots"], ""])
        with (kit / "2_denomination.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["item", "dix_mots", "nommable", "nom_propose"])
            for it in items:
                w.writerow([it["item"], it["dix_mots"], "", ""])
        print(f"  kit {n} : {kit.name}/ (1_intrusion.csv, 2_denomination.csv)")
    print(f"  {len(items)} items par kit")
    print(f"  clé → {SORTIE/'cle.csv'}, à ne pas transmettre aux annotateurs")


def depouiller() -> None:
    """Lit les réponses, calcule les scores et les imprime.

    Deux emplacements sont acceptés pour les réponses. L'application Streamlit
    écrit dans jugement/reponses/, avec le temps passé sur chaque item. Un
    remplissage à la main se fait dans le kit lui-même, sans les temps. Le
    premier trouvé est employé.

    Une réponse est tenue pour juste lorsqu'elle donne le mot intrus ou son
    numéro de colonne, les consignes autorisant les deux formes.

    Le hasard donne 16,7 pour cent, soit une chance sur six. Un modèle dont les
    thèmes sont cohérents doit dépasser nettement ce seuil.

    L'accord entre annotateurs est calculé sur les items que deux personnes ont
    tous deux annotés. Il mesure la difficulté de la tâche elle-même,
    indépendamment des modèles.
    """
    cle = {int(c["item"]): c for c in
           csv.DictReader((SORTIE / "cle.csv").open(encoding="utf-8"))}
    kits = sorted(SORTIE.glob("kit_annotateur_*"))
    if not kits:
        print("  aucun kit trouvé")
        return
    reponses: dict[str, dict[int, str]] = {}
    temps: dict[str, list[float]] = {}
    for kit in kits:
        # L'application écrit dans reponses/ ; le remplissage à la main se fait
        # dans le kit lui-même. Les deux sources sont acceptées.
        candidats = [SORTIE / "reponses" / f"{kit.name}_intrusion.csv",
                     kit / "1_intrusion.csv"]
        f = next((c for c in candidats if c.exists()), None)
        if f is None:
            continue
        d, t = {}, []
        for l in csv.DictReader(f.open(encoding="utf-8")):
            r = (l.get("intrus_designe") or "").strip().lower()
            if r:
                d[int(l["item"])] = r
                if l.get("secondes"):
                    t.append(float(l["secondes"]))
        if d:
            reponses[kit.name] = d
            if t:
                temps[kit.name] = t

    for nom, rep in reponses.items():
        par: dict[str, dict] = {}
        for n, r in rep.items():
            c = cle[n]
            m = par.setdefault(c["modele"], {"libelle": c["libelle"], "n": 0, "justes": 0})
            m["n"] += 1
            if r == c["intrus"].lower() or r == str(c["position"]):
                m["justes"] += 1
        print(f"\n=== {nom} : {len(rep)} items annotés ===")
        print(f"{'modèle':46} {'intrus trouvé':>18}")
        for lettre in sorted(par):
            m = par[lettre]
            print(f"{m['libelle'][:46]:46} {m['justes']/m['n']:12.1%}  ({m['justes']}/{m['n']})")
        print("  hasard attendu : 16,7 %")
        if nom in temps:
            import statistics as stt
            v = temps[nom]
            print(f"  temps par item : médiane {stt.median(v):.0f} s, "
                  f"total {sum(v)/60:.0f} min")

    if len(reponses) > 1:
        noms = list(reponses)
        communs = set(reponses[noms[0]]) & set(reponses[noms[1]])
        if communs:
            acc = sum(1 for n in communs
                      if reponses[noms[0]][n] == reponses[noms[1]][n]) / len(communs)
            print(f"\naccord entre annotateurs sur {len(communs)} items communs : {acc:.1%}")

    # dénomination
    for kit in kits:
        f = kit / "2_denomination.csv"
        if not f.exists():
            continue
        par: dict[str, list[str]] = {}
        f2 = SORTIE / "reponses" / f"{kit.name}_denomination.csv"
        if f2.exists():
            f = f2
        for l in csv.DictReader(f.open(encoding="utf-8")):
            v = (l.get("nommable") or "").strip().lower()
            if v:
                par.setdefault(cle[int(l["item"])]["modele"], []).append(v)
        if not par:
            continue
        print(f"\n=== {kit.name} : dénomination ===")
        for lettre in sorted(par):
            v = par[lettre]
            oui = sum(1 for x in v if x.startswith("o"))
            approx = sum(1 for x in v if x.startswith("a"))
            print(f"{cle[1]['libelle'][:20] if False else lettre:4} "
                  f"oui {oui:3}  approx {approx:3}  non {len(v)-oui-approx:3}  "
                  f"sur {len(v)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--depouiller", action="store_true")
    args = ap.parse_args()
    if args.depouiller:
        depouiller()
    else:
        items, cle = fabriquer(random.Random(GRAINE))
        ecrire(items, cle)
