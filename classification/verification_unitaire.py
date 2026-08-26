"""
verification_unitaire.py : la seconde épreuve, sur les étiquettes du régime unitaire

POURQUOI CETTE ÉPREUVE

La première vérification, conduite le 23 août 2026, portait sur les sorties de la
campagne groupée : 47,8 pour cent des vraies étiquettes acceptées contre 1,7 pour
cent des leurres.

Le contrôle du 24 août a repris les mêmes articles un par appel, et l'analyse des
réponses déjà données montre que ce régime discrimine : il reprend 54,7 pour cent
des étiquettes acceptées et 12,5 pour cent des rejetées, soit un rapport de 4,4
pour 1. Une réserve pèse pourtant sur ce constat. Sur les 543 étiquettes que le
régime unitaire produit, 355 sont absentes du régime groupé et n'ont donc jamais
été jugées. On peut établir que l'appel unitaire écarte le mauvais ; on ne peut
pas établir que ce qu'il ajoute est bon.

Cette épreuve lève la réserve. Elle soumet un échantillon de ces étiquettes
nouvelles au même protocole que la première, de sorte que les deux taux
deviennent comparables.

CE QUI REND LES DEUX SÉANCES COMPARABLES

Le taux de leurres acceptés. Il calibre chaque séance indépendamment : si les
deux donnent un taux de leurres voisin, le juge y a appliqué la même sévérité, et
les taux de vraies étiquettes se comparent. Un écart entre les deux taux de
leurres invaliderait la comparaison, et vaudrait d'être rapporté comme tel.

La graine diffère de celle de la première épreuve, de sorte que les leurres ne
soient pas les mêmes.

ENTRÉES

  verification/controle_unitaire.json   les sorties du régime unitaire
  verification/cle.json                 la première épreuve, pour écarter ses items
  ../results/feuilles_mistral_batched/  les sorties du régime groupé

SORTIES

  verification/C_unitaire.md            les items de la seconde épreuve
  verification/cle_unitaire.json        la clé, à ne pas ouvrir avant d'avoir répondu

Les réponses se donnent dans la même interface et le même fichier que la première
épreuve, les identifiants ne se recouvrant pas.

PAQUETS EMPLOYÉS

  json, random, argparse, pathlib, collections   bibliothèque standard
  les fonctions de verification_etiquettes, importées

USAGE

    python3 verification_unitaire.py              # fabrique le kit
    python3 verification_unitaire.py --depouiller # calcule les taux
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
GRAINE = 2
N_ITEMS = 120


def _module(nom):
    """Charge `classify_iptc_mistral_batched.py` comme module, pour en réemployer
    le tokeniseur, l'extraction des articles et la construction de la liste
    d'étiquettes plutôt que de les redéfinir ici."""
    spec = importlib.util.spec_from_file_location(nom.replace(".py", ""), ICI / nom)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def fabriquer():
    """Fabrique la seconde épreuve, sur les étiquettes que le régime unitaire
    produit et que le régime groupé ne produit pas. Ces étiquettes n'ont
    jamais été jugées, et rien ne permettait jusqu'ici de dire si le régime
    unitaire décrit mieux ou décrit seulement davantage."""
    v = _module("verification_etiquettes.py")
    cl = _module("classify_iptc_mistral_batched.py")
    rng = random.Random(GRAINE)

    leaves = cl.build_leaves(cl.TAXONOMY_PATH)
    uni = {k: v2["codes"] for k, v2 in
           json.loads((SORTIE / "controle_unitaire.json").read_text(encoding="utf-8")).items()
           if "codes" in v2}
    grp = {}
    for f in sorted(cl.OUTPUT_DIR.glob("*_themes.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        fasc = str(d.get("fascicule") or f.stem.split("_")[0])
        for a in d["articles"]:
            grp[f"{fasc}|{a['article_id']}"] = {t["code"] for t in (a.get("themes") or [])}

    # Les étiquettes propres au régime unitaire, jamais soumises au jugement.
    neuves = [(k, c) for k, cs in uni.items() for c in cs
              if c not in grp.get(k, set())]
    print(f"  {len(neuves)} étiquettes propres au régime unitaire")

    # Fréquences sur la campagne groupée, pour reprendre les mêmes bandes.
    freq = Counter(c for cs in grp.values() for c in cs)
    par_bande = defaultdict(list)
    for k, c in neuves:
        par_bande[v.bande(freq.get(c, 1))].append((k, c))
    for b in par_bande:
        print(f"     {b:<12} {len(par_bande[b])}")

    # Tirage stratifié, moitié vrais et moitié leurres.
    n_vrais = N_ITEMS // 2
    quota = max(1, n_vrais // len(par_bande))
    vrais = []
    for b, lot in par_bande.items():
        rng.shuffle(lot)
        vrais += [(k, c, b) for k, c in lot[:quota]]
    rng.shuffle(vrais)
    vrais = vrais[:n_vrais]

    # Un leurre est une étiquette de la même bande, tirée d'un autre article.
    tous = defaultdict(list)
    for c, n in freq.items():
        tous[v.bande(n)].append(c)
    leurres = []
    for k, _, b in vrais[:N_ITEMS - n_vrais]:
        candidats = [c for c in tous[b] if c not in uni.get(k, []) and c not in grp.get(k, set())]
        if candidats:
            leurres.append((k, rng.choice(candidats), b))

    items = [(k, c, b, True) for k, c, b in vrais] + [(k, c, b, False) for k, c, b in leurres]
    rng.shuffle(items)

    textes = v.textes(sorted({k.split("|")[0] for k, _, _, _ in items}))
    titres = {}
    for f in sorted({k.split("|")[0] for k, _, _, _ in items}):
        for a in cl.extract_articles(f):
            titres[(f, a["id"])] = a.get("title") or ""

    lignes = ["# Épreuve C — les étiquettes du régime un article par appel", "",
              "Un article, une étiquette. Dites si elle convient. Une partie des "
              "étiquettes présentées n'a pas été attribuée à cet article~: c'est voulu, "
              "et leur part n'est pas connue de vous.", "",
              "Répondez dans `reponses.csv`, à la suite des épreuves précédentes~: "
              "`o` si l'étiquette convient, `n` sinon, `?` si vous hésitez. "
              "N'ouvrez pas `cle_unitaire.json` avant d'avoir terminé.", "", "---", ""]
    cle = []
    for n, (k, code, b, vrai) in enumerate(items, 1):
        ident = f"C{n:03d}"
        fasc, aid = k.split("|")
        lab = leaves[code]["label_fr"]
        lignes += [f"## {ident}", "", f"**Étiquette proposée : {lab}**", ""]
        if titres.get((fasc, aid)):
            lignes += [f"*Titre relevé dans la carte logique :* {titres[(fasc, aid)]}", ""]
        lignes += ["> " + v.extrait(textes.get((fasc, aid), "")), ""]
        cle.append({"item": ident, "epreuve": "C", "vrai": vrai, "bande": b,
                    "code": code, "libelle": lab, "fascicule": fasc, "article": aid})
    (SORTIE / "C_unitaire.md").write_text("\n".join(lignes), encoding="utf-8")
    (SORTIE / "cle_unitaire.json").write_text(
        json.dumps(cle, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n  épreuve C : {len(items)} items, dont {sum(1 for x in items if not x[3])} leurres")
    print(f"  → {SORTIE/'C_unitaire.md'}")


def depouiller():
    """Dépouille les réponses de la seconde épreuve et les rapporte à celles de
    la première."""
    v = _module("verification_etiquettes.py")
    cle = {c["item"]: c for c in
           json.loads((SORTIE / "cle_unitaire.json").read_text(encoding="utf-8"))}
    rep = {}
    with (SORTIE / "reponses.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            s = (r.get("reponse") or "").strip().lower()
            if s and r["item"] in cle:
                rep[r["item"]] = s
    if not rep:
        raise SystemExit("aucune réponse pour l'épreuve C.")

    def taux(items, nom):
        """Affiche le taux d'acceptation d'un lot d'items, son intervalle de Wilson
        et le nombre de réponses indécises."""
        n = len(items)
        if not n:
            return None
        oui = sum(1 for i in items if rep.get(i) == "o")
        inc = sum(1 for i in items if rep.get(i) == "?")
        bas, haut = v.wilson(oui, n)
        print(f"  {nom:<34} {oui:>3}/{n:<3} = {oui/n*100:5.1f} %  "
              f"[{bas*100:4.1f} ; {haut*100:4.1f}]" + (f"   {inc} hésitation(s)" if inc else ""))
        return oui / n

    faits = [i for i in cle if i in rep]
    print(f"{len(faits)} items répondus sur {len(cle)}\n")
    print("ÉPREUVE C — étiquettes propres au régime unitaire")
    tv = taux([i for i in faits if cle[i]["vrai"]], "vraies étiquettes acceptées")
    tl = taux([i for i in faits if not cle[i]["vrai"]], "leurres acceptés")
    if tv is not None and tl is not None:
        print(f"\n  écart : {100*(tv-tl):.1f} points")
        print("\n  Pour mémoire, épreuve B, régime groupé~: 47,8 % de vraies, 1,7 % de leurres.")
        print("  Les deux séances ne se comparent que si les taux de leurres sont voisins.")
    print("\n  par bande de fréquence, vraies étiquettes seules")
    for b in ("fréquente", "moyenne", "rare", "hapax"):
        it = [i for i in faits if cle[i]["vrai"] and cle[i]["bande"] == b]
        if it:
            taux(it, f"    {b}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--depouiller", action="store_true")
    a = ap.parse_args()
    depouiller() if a.depouiller else fabriquer()
