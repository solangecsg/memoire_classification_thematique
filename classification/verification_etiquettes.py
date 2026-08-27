"""
verification_etiquettes.py : fabrique et dépouille la vérification humaine des
étiquettes IPTC

POURQUOI CE SCRIPT

La partie du mémoire consacrée au vocabulaire contrôlé mesure un accord entre
deux dispositifs, 78,2 pour cent avec un classifieur affiné, et non une
exactitude. Aucune vérité de référence n'existe pour ce corpus. La première
question qu'un lecteur pose est donc restée sans réponse : une étiquette
attribuée est-elle juste ?

Ce script prépare deux épreuves qui y répondent dans les limites du possible,
sans prétendre produire la vérité de référence qui manque.

LES DEUX ÉPREUVES

  A. AUDIT DES HAPAX. Les 62 étiquettes qu'un seul article porte sont celles sur
     lesquelles repose l'argument le plus exposé du chapitre : le vocabulaire
     contrôlé atteint le rare là où les méthodes émergentes le diluent. Une
     étiquette employée une seule fois peut désigner une matière que le corpus
     ne traite qu'une fois, ou une erreur de classement, et rien dans la chaîne
     ne les sépare. L'épreuve les présente toutes, une par une.

  B. ACCEPTABILITÉ EN AVEUGLE. Un échantillon d'articles est présenté avec une
     seule étiquette. Une fois sur deux c'est celle que le modèle a réellement
     attribuée, une fois sur deux un leurre tiré des étiquettes d'un autre
     article. Le juge ignore laquelle lui est montrée.

POURQUOI DES LEURRES

Le juge est ici l'auteure du mémoire, ce qui expose l'épreuve au reproche
d'être juge et partie. Les leurres y répondent en rendant la complaisance
mesurable. L'épreuve produit deux taux plutôt qu'un : l'acceptation des vraies
étiquettes, qui estime la précision, et l'acceptation des leurres, qui dit ce
que vaut la première. Sans écart entre les deux, le premier chiffre ne vaut
rien, et le savoir vaut mieux que l'ignorer.

Le procédé est celui de l'intrusion de mot employée au chapitre précédent, dont
il reprend le principe : mêler à l'objet évalué un élément dont on sait qu'il
est faux, et mesurer la capacité du juge à l'écarter.

LA STRATIFICATION

Les étiquettes ne se valent pas. Une étiquette portée par cinq cents articles et
une étiquette portée par un seul ne posent pas le même problème au modèle.
L'échantillon est donc tiré par bandes de fréquence, et le dépouillement rend un
taux par bande. Les leurres sont tirés dans la bande de leur item, faute de quoi
leur détection tiendrait à leur rareté plutôt qu'à leur inadéquation.

ENTRÉES

  ../results/feuilles_mistral_batched/*_themes.json   attributions du modèle
  ../re-ocr/corpus/original/{fascicule}/toc/          cartes logiques
  ../re-ocr/corpus/reocr_mistral/{fascicule}_reocr/   texte ré-océrisé

SORTIES

  verification/A_hapax.md              les 62 items de la première épreuve
  verification/B_acceptabilite.md      les items de la seconde
  verification/reponses.csv            grille de réponse à remplir
  verification/cle.json                clé, à ne pas ouvrir avant de répondre

PAQUETS EMPLOYÉS

  json, csv, random, argparse, pathlib, collections   bibliothèque standard
  math                                                intervalles de Wilson

La graine est fixée : les mêmes kits se régénèrent à l'identique.

USAGE

    python3 verification_etiquettes.py              # fabrique les deux kits
    python3 verification_etiquettes.py --depouiller # calcule les taux
"""

import argparse
import csv
import glob
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ICI = Path(__file__).resolve().parent
SORTIE = ICI / "verification"
GRAINE = 1

# Bandes de fréquence des étiquettes. Une étiquette portée par cinq cents
# articles et une portée par un seul ne posent pas le même problème.
BANDES = [("fréquente", 100, 10**9), ("moyenne", 10, 100), ("rare", 2, 10), ("hapax", 1, 2)]

# Nombre d'items de la seconde épreuve, moitié vrais et moitié leurres.
N_ITEMS_B = 150

# Longueur du texte présenté au juge. Un thème se juge sur l'entrée en matière ;
# au-delà, la lecture ralentit sans rien ajouter.
MOTS_MONTRES = 180


def bande(n):
    """La bande de fréquence où tombe une étiquette employée n fois. La
    stratification par bande garantit que le rare soit jugé, alors qu'un
    tirage uniforme le manquerait presque toujours."""
    for nom, bas, haut in BANDES:
        if bas <= n < haut:
            return nom
    return "hapax"


def attributions():
    """Rend les attributions du modèle, article par article."""
    fs = sorted(glob.glob(str(ICI.parent / "results" / "feuilles_mistral_batched" / "*_themes.json")))
    if not fs:
        raise SystemExit(
            "résultats de classification introuvables sous "
            "results/feuilles_mistral_batched/. Ils sont produits par "
            "classify_iptc_mistral_batched.py.")
    out = []
    for f in fs:
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        fasc = d.get("fascicule") or Path(f).stem.split("_")[0]
        for a in d["articles"]:
            if a.get("themes"):
                out.append({"fascicule": fasc, "id": a["article_id"],
                            "titre": a.get("title") or "", "themes": a["themes"]})
    return out


def textes(fascicules):
    """Rend le texte de chaque article, reconstitué depuis la carte logique.

    La fonction du script de classification est réemployée telle quelle, de
    sorte que le juge lise exactement le texte que le modèle a reçu.
    """
    import importlib.util
    chemin = ICI / "classify_iptc_mistral_batched.py"
    spec = importlib.util.spec_from_file_location("classif", chemin)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    out = {}
    for i, f in enumerate(sorted(fascicules), 1):
        print(f"  texte {i}/{len(fascicules)} : {f}", end="\r", file=sys.stderr)
        for a in m.extract_articles(f):
            out[(f, a["id"])] = a["text"]
    print(" " * 60, end="\r", file=sys.stderr)
    return out


def extrait(texte, mots=MOTS_MONTRES):
    """Les premiers mots de l'article, en une seule ligne. L'annotateur juge sur
    un extrait de longueur constante, de sorte qu'un article long ne reçoive
    pas plus d'attention qu'un bref."""
    m = " ".join(texte.split())
    d = m.split(" ")
    return " ".join(d[:mots]) + (" […]" if len(d) > mots else "")


def fabriquer():
    """Écrit les deux kits et la clé."""
    rng = random.Random(GRAINE)
    arts = attributions()
    freq = Counter((t["code"], t["label_fr"]) for a in arts for t in a["themes"])
    par_bande = defaultdict(list)
    for (code, lib), n in freq.items():
        par_bande[bande(n)].append((code, lib))

    # A. les hapax
    hapax = {c for (c, _), n in freq.items() if n == 1}
    items_a = []
    for a in arts:
        for t in a["themes"]:
            if t["code"] in hapax:
                items_a.append({"epreuve": "A", "article": a, "etiquette": t, "vrai": True,
                                "bande": "hapax"})
    items_a.sort(key=lambda x: x["etiquette"]["label_fr"])

    # B. l'échantillon stratifié, moitié leurres
    par_art_bande = defaultdict(list)
    for a in arts:
        b = bande(freq[(a["themes"][0]["code"], a["themes"][0]["label_fr"])])
        par_art_bande[b].append(a)
    items_b, quota = [], N_ITEMS_B // len([b for b in par_art_bande if par_art_bande[b]])
    for b, pool in par_art_bande.items():
        for a in rng.sample(pool, min(quota, len(pool))):
            vrai = rng.random() < 0.5
            porte = {t["code"] for t in a["themes"]}
            if vrai:
                et = rng.choice(a["themes"])
            else:
                # Le leurre est tiré dans la même bande, faute de quoi sa
                # détection tiendrait à sa rareté plutôt qu'à son inadéquation.
                cands = [x for x in par_bande[b] if x[0] not in porte]
                if not cands:
                    continue
                c, lib = rng.choice(cands)
                et = {"code": c, "label_fr": lib}
            items_b.append({"epreuve": "B", "article": a, "etiquette": et,
                            "vrai": vrai, "bande": b})
    rng.shuffle(items_b)

    tous = items_a + items_b
    txt = textes({i["article"]["fascicule"] for i in tous})

    SORTIE.mkdir(exist_ok=True)
    entete = {
        "A": ("Épreuve A — les étiquettes employées une seule fois",
              "Les 62 étiquettes que le modèle n'a attribuées qu'à un seul article. "
              "Pour chacune, dites si elle convient à l'article. Aucun leurre ici."),
        "B": ("Épreuve B — acceptabilité",
              "Un article, une étiquette. Dites si elle convient. "
              "Une partie des étiquettes présentées n'a pas été attribuée à cet "
              "article : c'est voulu, et leur part n'est pas connue de vous."),
    }
    lignes_reponse = []
    for ep, liste in (("A", items_a), ("B", items_b)):
        titre, consigne = entete[ep]
        L = [f"# {titre}\n", consigne,
             "\nRépondez dans `reponses.csv` : `o` si l'étiquette convient, "
             "`n` sinon, `?` si vous hésitez. N'ouvrez pas `cle.json` avant "
             "d'avoir terminé.\n", "---\n"]
        for k, it in enumerate(liste, 1):
            a = it["article"]
            t = txt.get((a["fascicule"], a["id"]), "")
            ident = f"{ep}{k:03d}"
            L.append(f"## {ident}\n")
            L.append(f"**Étiquette proposée : {it['etiquette']['label_fr']}**\n")
            if a["titre"] and not re.fullmatch(r"[A-Z]+\.\d+", a["titre"]):
                L.append(f"*Titre relevé dans la carte logique :* {a['titre']}\n")
            L.append(f"> {extrait(t) if t else '(texte indisponible)'}\n")
            lignes_reponse.append((ident, ""))
        (SORTIE / f"{ep}_{'hapax' if ep=='A' else 'acceptabilite'}.md").write_text(
            "\n".join(L), encoding="utf-8")

    with open(SORTIE / "reponses.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item", "reponse"])
        w.writerows(lignes_reponse)

    cle = [{"item": f"{it['epreuve']}{k:03d}", "epreuve": it["epreuve"],
            "vrai": it["vrai"], "bande": it["bande"],
            "code": it["etiquette"]["code"], "libelle": it["etiquette"]["label_fr"],
            "fascicule": it["article"]["fascicule"], "article": it["article"]["id"]}
           for ep, liste in (("A", items_a), ("B", items_b))
           for k, it in enumerate(liste, 1)]
    (SORTIE / "cle.json").write_text(json.dumps(cle, ensure_ascii=False, indent=1),
                                     encoding="utf-8")
    print(f"épreuve A : {len(items_a)} items (hapax)")
    print(f"épreuve B : {len(items_b)} items, dont {sum(1 for i in items_b if not i['vrai'])} leurres")
    print(f"→ {SORTIE}")


def wilson(k, n, z=1.96):
    """Intervalle de Wilson, celui qu'emploie déjà l'évaluation du chapitre 2."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    e = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - e), min(1.0, c + e))


def depouiller():
    """Croise les réponses et la clé, et rend les taux avec leurs intervalles."""
    for f in ("cle.json", "reponses.csv"):
        if not (SORTIE / f).is_file():
            raise SystemExit(f"{f} absent : fabriquer les kits puis répondre.")
    cle = {c["item"]: c for c in json.loads((SORTIE / "cle.json").read_text(encoding="utf-8"))}
    rep = {}
    with open(SORTIE / "reponses.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            v = (r["reponse"] or "").strip().lower()
            if v:
                rep[r["item"]] = v
    if not rep:
        raise SystemExit("aucune réponse dans reponses.csv.")

    def taux(items, nom):
        """Affiche le taux d'acceptation d'un lot d'items, son intervalle de Wilson
        et le nombre de réponses indécises."""
        n = len(items)
        if not n:
            return
        oui = sum(1 for i in items if rep.get(i) == "o")
        inc = sum(1 for i in items if rep.get(i) == "?")
        bas, haut = wilson(oui, n)
        print(f"  {nom:<34} {oui:>3}/{n:<3} = {oui/n*100:5.1f} %  "
              f"[{bas*100:4.1f} ; {haut*100:4.1f}]" + (f"   {inc} hésitation(s)" if inc else ""))

    repondus = [i for i in cle if i in rep]
    print(f"{len(repondus)} items répondus sur {len(cle)}\n")
    print("ÉPREUVE A — étiquettes employées une seule fois")
    taux([i for i in repondus if cle[i]["epreuve"] == "A"], "convenables")
    b = [i for i in repondus if cle[i]["epreuve"] == "B"]
    vrais = [i for i in b if cle[i]["vrai"]]
    leurres = [i for i in b if not cle[i]["vrai"]]
    print("\nÉPREUVE B — acceptabilité en aveugle")
    taux(vrais, "vraies étiquettes acceptées")
    taux(leurres, "leurres acceptés")
    if vrais and leurres:
        pv = sum(1 for i in vrais if rep.get(i) == "o") / len(vrais)
        pl = sum(1 for i in leurres if rep.get(i) == "o") / len(leurres)
        print(f"\n  écart : {(pv-pl)*100:.1f} points")
        if pv - pl < 0.20:
            print("  L'écart est faible : le jugement ne discrimine guère, et le")
            print("  taux d'acceptation des vraies étiquettes ne dit alors pas grand-chose.")
    print("\n  par bande de fréquence, vraies étiquettes seules")
    for nom, _, _ in BANDES:
        taux([i for i in vrais if cle[i]["bande"] == nom], f"  {nom}")


def main():
    """Point d'entrée : fabrique les deux premières épreuves, ou dépouille les
    réponses déjà données."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--depouiller", action="store_true",
                   help="calculer les taux à partir de reponses.csv")
    a = p.parse_args()
    depouiller() if a.depouiller else fabriquer()


if __name__ == "__main__":
    main()
