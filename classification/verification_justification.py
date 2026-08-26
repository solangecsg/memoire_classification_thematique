"""
verification_justification.py : la quatrième épreuve, sur la justification préalable

POURQUOI CETTE ÉPREUVE

Un essai conduit sur le modèle local a montré qu'une phrase de justification
demandée AVANT les étiquettes transforme ses sorties : une cote de bourse qui
recevait « Homicide » reçoit « Médicaments » dès lors que le modèle écrit
d'abord de quoi l'article traite. Le procédé a été porté sur le service
commercial dans les deux régimes, pour un surcoût de un pour cent en appel
unitaire et de huit pour cent en cascade.

Ce que la mesure en dit s'arrête là. Le nombre d'étiquettes rendues baisse, de
3,21 à 2,93 en appel unitaire et de 2,47 à 2,13 en cascade, ce qui suggère un
choix plus sélectif sans l'établir : moins d'étiquettes peut signifier mieux
choisies comme cela peut signifier une description appauvrie. Les quatre épreuves
précédentes ont buté sur un plafond de 47 à 55 pour cent, et la justification est
le seul réglage éprouvé qui pourrait le déplacer. Seul un jugement humain le dira.

CE QUI REND LES SÉANCES COMPARABLES

Le taux de leurres acceptés, qui calibre chaque séance indépendamment. Il valait
1,7 pour cent à la première, 0,0 à la deuxième et 3,8 à la troisième, écarts que
le test ne distingue pas. Si la quatrième donne un taux voisin, les taux de
vraies étiquettes de tous les régimes se lisent les uns contre les autres.

CE QUE LE JUGE NE DOIT PAS POUVOIR DEVINER

Quel régime a produit quelle étiquette, ni que la justification est en cause. Les
items des deux régimes sont mêlés et tirés dans un ordre aléatoire. La phrase de
justification produite par le modèle n'est PAS montrée : la lire reviendrait à
juger un plaidoyer plutôt qu'une étiquette, et une justification bien tournée
emporterait l'adhésion sur une étiquette fausse. Les étiquettes déjà jugées lors
des épreuves précédentes sont écartées, une seconde soumission ne mesurant plus
qu'une mémoire.

ENTRÉES

  verification/controle_justification_unitaire.json   un appel par article
  verification/controle_justification_cascade.json    cascade à deux étages
  verification/cle.json, cle_unitaire.json,
  verification/cle_cascades.json                      pour écarter le déjà jugé

SORTIES

  verification/E_justification.md
  verification/cle_justification.json

PAQUETS EMPLOYÉS

  json, csv, random, argparse, pathlib, collections   bibliothèque standard
  les fonctions de verification_etiquettes, importées

USAGE

    python3 verification_justification.py              # fabrique le kit
    python3 verification_justification.py --depouiller # calcule les taux
"""

import argparse
import csv
import importlib.util
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ICI = Path(__file__).resolve().parent
SORTIE = ICI / "verification"
GRAINE = 4
N_VRAIS_PAR_REGIME = 40

SOURCES = (("unitaire", "controle_justification_unitaire.json"),
           ("cascade", "controle_justification_cascade.json"))


def _module(nom):
    """Charge un script voisin comme module, pour en réemployer le tokeniseur,
    l'extraction des articles et la construction de la liste d'étiquettes plutôt
    que de les redéfinir ici."""
    spec = importlib.util.spec_from_file_location(nom.replace(".py", ""), ICI / nom)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def deja_juge():
    """Les couples (article, code) déjà soumis à un jugement, à écarter. Les
    trois épreuves précédentes sont dépouillées ensemble : une étiquette déjà vue
    ne mesurerait plus qu'une mémoire."""
    vus = set()
    for f in ("cle.json", "cle_unitaire.json", "cle_cascades.json"):
        p = SORTIE / f
        if p.exists():
            for c in json.loads(p.read_text(encoding="utf-8")):
                vus.add((f'{c["fascicule"]}|{c["article"]}', c["code"]))
    return vus


def fabriquer():
    """Fabrique la quatrième épreuve : les deux régimes à justification préalable,
    mêlés et anonymes, moitié vraies étiquettes moitié leurres. Les juger dans une
    séance unique plutôt que deux écarte l'effet de séance, qui pèserait sur la
    comparaison."""
    v = _module("verification_etiquettes.py")
    cl = _module("classify_iptc_mistral_batched.py")
    rng = random.Random(GRAINE)
    leaves = cl.build_leaves(cl.TAXONOMY_PATH)
    vus = deja_juge()
    print(f"  {len(vus)} couples article-étiquette déjà jugés, écartés")

    sources = {}
    for nom, fic in SOURCES:
        p = SORTIE / fic
        if not p.exists():
            raise SystemExit(f"{fic} absent : le régime correspondant n'a pas fini.")
        sources[nom] = {k: d["codes"] for k, d in
                        json.loads(p.read_text(encoding="utf-8")).items() if "codes" in d}
        print(f"  régime {nom:<10} {len(sources[nom])} articles")

    # Fréquences sur la campagne groupée, pour reprendre les mêmes bandes.
    freq = Counter()
    for f in sorted(cl.OUTPUT_DIR.glob("*_themes.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        for a in d["articles"]:
            freq.update(t["code"] for t in (a.get("themes") or []))

    items = []
    for nom, sortie in sources.items():
        cand = [(k, c) for k, cs in sortie.items() for c in cs if (k, c) not in vus]
        par_bande = defaultdict(list)
        for k, c in cand:
            par_bande[v.bande(freq.get(c, 1))].append((k, c))
        print(f"     {nom} : {len(cand)} étiquettes jamais jugées, "
              f"réparties en {dict((b, len(x)) for b, x in par_bande.items())}")
        quota = max(1, N_VRAIS_PAR_REGIME // max(1, len(par_bande)))
        vrais = []
        for b, lot in par_bande.items():
            rng.shuffle(lot)
            vrais += [(k, c, b) for k, c in lot[:quota]]
        rng.shuffle(vrais)
        vrais = vrais[:N_VRAIS_PAR_REGIME]
        # Un leurre par vrai, tiré dans la même bande et absent de l'article.
        tous = defaultdict(list)
        for c, n in freq.items():
            tous[v.bande(n)].append(c)
        for k, _, b in vrais:
            interdits = set(sortie.get(k, [])) | {c for kk, c in vus if kk == k}
            possibles = [c for c in tous[b] if c not in interdits]
            if possibles:
                items.append((k, rng.choice(possibles), b, False, nom))
        items += [(k, c, b, True, nom) for k, c, b in vrais]
    rng.shuffle(items)

    fascs = sorted({k.split("|")[0] for k, _, _, _, _ in items})
    textes = v.textes(fascs)
    titres = {}
    for f in fascs:
        for a in cl.extract_articles(f):
            titres[(f, a["id"])] = a.get("title") or ""

    lignes = ["# Épreuve E — la justification demandée avant l'étiquette", "",
              "Un article, une étiquette. Dites si elle convient. Une partie des "
              "étiquettes présentées n'a pas été attribuée à cet article~: c'est voulu, "
              "et leur part n'est pas connue de vous.", "",
              "Répondez dans `reponses.csv`, à la suite des épreuves précédentes.", "",
              "---", ""]
    cle = []
    for n, (k, code, b, vrai, source) in enumerate(items, 1):
        ident = f"E{n:03d}"
        fasc, aid = k.split("|")
        lab = leaves[code]["label_fr"]
        lignes += [f"## {ident}", "", f"**Étiquette proposée : {lab}**", ""]
        if titres.get((fasc, aid)):
            lignes += [f"*Titre relevé dans la carte logique :* {titres[(fasc, aid)]}", ""]
        lignes += ["> " + v.extrait(textes.get((fasc, aid), "")), ""]
        cle.append({"item": ident, "epreuve": "E", "vrai": vrai, "bande": b,
                    "source": source, "code": code, "libelle": lab,
                    "fascicule": fasc, "article": aid})
    (SORTIE / "E_justification.md").write_text("\n".join(lignes), encoding="utf-8")
    (SORTIE / "cle_justification.json").write_text(
        json.dumps(cle, ensure_ascii=False, indent=1), encoding="utf-8")
    n_l = sum(1 for x in items if not x[3])
    print(f"\n  épreuve E : {len(items)} items, dont {n_l} leurres")
    for nom in sources:
        print(f"     régime {nom:<10} {sum(1 for x in items if x[4]==nom and x[3])} vrais, "
              f"{sum(1 for x in items if x[4]==nom and not x[3])} leurres")
    print(f"  → {SORTIE/'E_justification.md'}")


def depouiller():
    """Dépouille les réponses en séparant les deux régimes, que l'épreuve avait
    confondus à dessein, et les rapporte aux mêmes régimes sans justification."""
    v = _module("verification_etiquettes.py")
    cle = {c["item"]: c for c in
           json.loads((SORTIE / "cle_justification.json").read_text(encoding="utf-8"))}
    rep = {}
    with (SORTIE / "reponses.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            s = (r.get("reponse") or "").strip().lower()
            if s and r["item"] in cle:
                rep[r["item"]] = s

    def taux(items, nom):
        """Affiche le taux d'acceptation d'un lot d'items, son intervalle de Wilson
        et le nombre de réponses indécises."""
        n = len(items)
        if not n:
            return None
        oui = sum(1 for i in items if rep.get(i) == "o")
        inc = sum(1 for i in items if rep.get(i) == "?")
        bas, haut = v.wilson(oui, n)
        print(f"  {nom:<40} {oui:>3}/{n:<3} = {oui/n*100:5.1f} %  "
              f"[{bas*100:4.1f} ; {haut*100:4.1f}]" + (f"   {inc} hésitation(s)" if inc else ""))
        return oui / n

    faits = [i for i in cle if i in rep]
    if not faits:
        raise SystemExit("aucune réponse pour l'épreuve E.")
    print(f"{len(faits)} items répondus sur {len(cle)}\n")
    for source in ("unitaire", "cascade"):
        print(f"RÉGIME {source.upper()}, JUSTIFICATION PRÉALABLE")
        taux([i for i in faits if cle[i]["source"] == source and cle[i]["vrai"]],
             "vraies étiquettes acceptées")
        taux([i for i in faits if cle[i]["source"] == source and not cle[i]["vrai"]],
             "leurres acceptés")
        print()
    print("Tous leurres confondus, pour calibrer la séance")
    taux([i for i in faits if not cle[i]["vrai"]], "leurres acceptés")
    print("\n  Pour mémoire, sans justification~: régime groupé 47,8 %, cascade")
    print("  commerciale 47,5 %, étiquettes propres à l'appel unitaire 39,7 %.")
    print("  Leurres des trois séances précédentes~: 1,7 %, 0,0 %, 3,8 %.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--depouiller", action="store_true")
    a = ap.parse_args()
    depouiller() if a.depouiller else fabriquer()
