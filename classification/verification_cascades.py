"""
verification_cascades.py : la troisième épreuve, sur les deux cascades

POURQUOI CETTE ÉPREUVE

Les cascades à deux étages ont été mesurées par leur ressemblance avec les autres
régimes : la cascade locale triple son recouvrement avec le modèle commercial
lorsqu'elle raccourcit la liste, et la cascade commerciale coûte quatre fois
moins qu'un appel unitaire à liste entière. Aucune de ces mesures ne dit la
justesse. Deux dispositifs peuvent se ressembler et se tromper ensemble, ce que
la première épreuve avait déjà établi pour l'accord de 78,2 pour cent.

Cette épreuve soumet les deux cascades au même jugement humain que les régimes
précédents, dans une séance unique. Les juger séparément coûterait deux séances
et rendrait la comparaison plus fragile, la sévérité du juge pouvant dériver
d'un jour à l'autre.

CE QUI REND LES TROIS SÉANCES COMPARABLES

Le taux de leurres acceptés, qui calibre chaque séance indépendamment. Il valait
1,7 pour cent à la première et zéro à la deuxième, écart que le test ne
distingue pas. Si la troisième donne un taux voisin, les taux de vraies
étiquettes des cinq régimes se lisent les uns contre les autres.

CE QUE LE JUGE NE DOIT PAS POUVOIR DEVINER

Quel dispositif a produit quelle étiquette. Les items des deux cascades sont
mêlés et tirés dans un ordre aléatoire, et rien dans leur présentation ne les
distingue. Les étiquettes déjà jugées lors des épreuves précédentes sont
écartées, une seconde soumission ne mesurant plus qu'une mémoire.

ENTRÉES

  verification/controle_ollama_cascade.json      cascade locale
  verification/controle_cascade_mistral.json     cascade commerciale
  verification/cle.json, cle_unitaire.json       pour écarter le déjà jugé

SORTIES

  verification/D_cascades.md
  verification/cle_cascades.json

PAQUETS EMPLOYÉS

  json, csv, random, argparse, pathlib, collections   bibliothèque standard
  les fonctions de verification_etiquettes, importées

USAGE

    python3 verification_cascades.py              # fabrique le kit
    python3 verification_cascades.py --depouiller # calcule les taux
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
GRAINE = 3
N_VRAIS_PAR_CASCADE = 40


def _module(nom):
    """Charge `classify_iptc_mistral_batched.py` comme module, pour en réemployer
    le tokeniseur, l'extraction des articles et la construction de la liste
    d'étiquettes plutôt que de les redéfinir ici."""
    spec = importlib.util.spec_from_file_location(nom.replace(".py", ""), ICI / nom)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def deja_juge():
    """Les couples (article, code) déjà soumis à un jugement, à écarter."""
    vus = set()
    for f in ("cle.json", "cle_unitaire.json"):
        p = SORTIE / f
        if p.exists():
            for c in json.loads(p.read_text(encoding="utf-8")):
                vus.add((f'{c["fascicule"]}|{c["article"]}', c["code"]))
    return vus


def fabriquer():
    """Fabrique la troisième épreuve : les deux cascades mêlées et anonymes,
    moitié vraies étiquettes moitié leurres. Les juger dans une séance unique
    plutôt que deux écarte l'effet de séance, qui pèserait sur la
    comparaison."""
    v = _module("verification_etiquettes.py")
    cl = _module("classify_iptc_mistral_batched.py")
    rng = random.Random(GRAINE)
    leaves = cl.build_leaves(cl.TAXONOMY_PATH)
    vus = deja_juge()

    sources = {}
    for nom, fic in (("locale", "controle_ollama_cascade.json"),
                     ("commerciale", "controle_cascade_mistral.json")):
        p = SORTIE / fic
        if not p.exists():
            raise SystemExit(f"{fic} absent : la cascade correspondante n'a pas fini.")
        sources[nom] = {k: v2["codes"] for k, v2 in
                        json.loads(p.read_text(encoding="utf-8")).items() if "codes" in v2}
        print(f"  cascade {nom:<12} {len(sources[nom])} articles")

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
        quota = max(1, N_VRAIS_PAR_CASCADE // max(1, len(par_bande)))
        vrais = []
        for b, lot in par_bande.items():
            rng.shuffle(lot)
            vrais += [(k, c, b) for k, c in lot[:quota]]
        rng.shuffle(vrais)
        vrais = vrais[:N_VRAIS_PAR_CASCADE]
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

    lignes = ["# Épreuve D — les deux cascades à deux étages", "",
              "Un article, une étiquette. Dites si elle convient. Une partie des "
              "étiquettes présentées n'a pas été attribuée à cet article~: c'est voulu, "
              "et leur part n'est pas connue de vous.", "",
              "Répondez dans `reponses.csv`, à la suite des épreuves précédentes.", "",
              "---", ""]
    cle = []
    for n, (k, code, b, vrai, source) in enumerate(items, 1):
        ident = f"D{n:03d}"
        fasc, aid = k.split("|")
        lab = leaves[code]["label_fr"]
        lignes += [f"## {ident}", "", f"**Étiquette proposée : {lab}**", ""]
        if titres.get((fasc, aid)):
            lignes += [f"*Titre relevé dans la carte logique :* {titres[(fasc, aid)]}", ""]
        lignes += ["> " + v.extrait(textes.get((fasc, aid), "")), ""]
        cle.append({"item": ident, "epreuve": "D", "vrai": vrai, "bande": b,
                    "source": source, "code": code, "libelle": lab,
                    "fascicule": fasc, "article": aid})
    (SORTIE / "D_cascades.md").write_text("\n".join(lignes), encoding="utf-8")
    (SORTIE / "cle_cascades.json").write_text(
        json.dumps(cle, ensure_ascii=False, indent=1), encoding="utf-8")
    n_l = sum(1 for x in items if not x[3])
    print(f"\n  épreuve D : {len(items)} items, dont {n_l} leurres")
    for nom in sources:
        print(f"     cascade {nom:<12} {sum(1 for x in items if x[4]==nom and x[3])} vrais, "
              f"{sum(1 for x in items if x[4]==nom and not x[3])} leurres")
    print(f"  → {SORTIE/'D_cascades.md'}")


def depouiller():
    """Dépouille les réponses en séparant les deux cascades, que l'épreuve avait
    confondues à dessein."""
    v = _module("verification_etiquettes.py")
    cle = {c["item"]: c for c in
           json.loads((SORTIE / "cle_cascades.json").read_text(encoding="utf-8"))}
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
        raise SystemExit("aucune réponse pour l'épreuve D.")
    print(f"{len(faits)} items répondus sur {len(cle)}\n")
    for source in ("commerciale", "locale"):
        print(f"CASCADE {source.upper()}")
        taux([i for i in faits if cle[i]["source"] == source and cle[i]["vrai"]],
             "vraies étiquettes acceptées")
        taux([i for i in faits if cle[i]["source"] == source and not cle[i]["vrai"]],
             "leurres acceptés")
        print()
    print("Tous leurres confondus, pour calibrer la séance")
    taux([i for i in faits if not cle[i]["vrai"]], "leurres acceptés")
    print("\n  Pour mémoire~: régime groupé 47,8 %, appel unitaire 55,5 % estimé,")
    print("  étiquettes propres à l'appel unitaire 39,7 %. Leurres~: 1,7 % puis 0,0 %.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--depouiller", action="store_true")
    a = ap.parse_args()
    depouiller() if a.depouiller else fabriquer()
