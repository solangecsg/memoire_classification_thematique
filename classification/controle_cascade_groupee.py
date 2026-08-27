"""
controle_cascade_groupee.py : la cascade à deux étages, en lots

POURQUOI CE SCRIPT

Le mémoire écarte la cascade sur un calcul de dépense : $4,57 contre $3,63 pour
le groupage simple, réduire la liste ne réduisant pas le texte qu'il faut de
toute façon transmettre. Le calcul portait sur une cascade unitaire.

Groupée, la cascade se chiffre autrement. Le calcul conduit sur les 6 699
articles du sous-corpus donne, pour mistral-large :

  groupage simple, la référence                         $3,34
  cascade unitaire                                      $7,78   +133 %
  cascade groupée, texte entier aux deux étages         $4,44    +33 %
  cascade groupée, 60 jetons au premier étage           $2,89    -13 %

La cascade groupée reste donc plus chère tant qu'elle transmet le texte entier
deux fois. Elle passe sous le groupage simple si le premier étage se contente
d'une amorce, ce qu'un choix parmi dix-sept familles rend plausible. C'est cette
hypothèse que le script éprouve.

LES DEUX ÉTAGES

Le premier soumet un lot d'articles et les dix-sept familles de premier niveau,
et rend une famille par article. Le second regroupe les articles par famille
retenue, et soumet à chaque groupe les seules étiquettes de sa famille,
vingt-six en médiane. Un article ne voit donc jamais la liste entière.

CE QUE MESURE L'OPTION --amorce

La longueur de texte transmise au premier étage. À zéro, l'article part entier.
À soixante, seuls ses soixante premiers jetons partent, ce qui divise par
plusieurs la dépense du premier étage. Comparer les familles obtenues dans les
deux cas dit si l'amorce suffit, question qui décide de tout le reste.

ENTRÉES ET SORTIES

  verification/cle.json                        les articles déjà jugés
  config/.env                                  MISTRAL_API_KEY
  verification/controle_cascade_groupee.json
  verification/controle_cascade_groupee.md

PAQUETS EMPLOYÉS

  json, time, argparse, pathlib, collections   bibliothèque standard
  les fonctions de classify_iptc_mistral_batched, importées

USAGE

    python3 controle_cascade_groupee.py --estimer
    python3 controle_cascade_groupee.py --amorce 60
    python3 controle_cascade_groupee.py
"""

import argparse
import importlib.util
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ICI = Path(__file__).resolve().parent
SORTIE = ICI / "verification"

LOT_ETAGE1 = 25
LOT_ETAGE2 = 25


def _module():
    """Charge `classify_iptc_mistral_batched.py` comme module, pour en réemployer
    le tokeniseur, l'extraction des articles et la construction de la liste
    d'étiquettes plutôt que de les redéfinir ici."""
    spec = importlib.util.spec_from_file_location(
        "classif", ICI / "classify_iptc_mistral_batched.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def articles_juges():
    """Les couples (fascicule, article) déjà soumis au jugement humain, dans
    l'ordre et sans doublon. Éprouver un régime sur eux seuls permet de
    confronter ses sorties à des étiquettes dont la justesse est connue."""
    cle = json.loads((SORTIE / "cle.json").read_text(encoding="utf-8"))
    vus, out = set(), []
    for c in cle:
        k = (c["fascicule"], c["article"])
        if k not in vus:
            vus.add(k)
            out.append(k)
    return sorted(out)


def tronquer(m, texte, amorce):
    """Rend les `amorce` premiers jetons du texte. À zéro, rend tout."""
    if not amorce:
        return texte
    mots = texte.split()
    # Découpe par approximations successives plutôt que jeton par jeton.
    haut = len(mots)
    while haut > 1 and m.count_tokens(" ".join(mots[:haut])) > amorce:
        haut = int(haut * 0.8)
    return " ".join(mots[:max(1, haut)])


def main():
    """Point d'entrée : conduit la cascade en groupant les articles à chaque
    étage, réunion des deux économies dont ce chapitre mesure les effets
    séparément."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--amorce", type=int, default=0,
                    help="jetons transmis au premier étage ; 0 pour l'article entier")
    ap.add_argument("--estimer", action="store_true")
    ap.add_argument("--limite", type=int, default=0)
    ap.add_argument("--comparer", action="store_true")
    args = ap.parse_args()

    resultat = SORTIE / (f"controle_cascade_groupee_{args.amorce}.json"
                         if args.amorce else "controle_cascade_groupee.json")
    rapport = resultat.with_suffix(".md")

    m = _module()
    leaves = m.build_leaves(m.TAXONOMY_PATH)
    familles, code_l1 = defaultdict(list), {}
    for c, v in leaves.items():
        familles[v["l1_code"]].append(c)
        code_l1[v["l1_code"]] = v["l1_label"]
    fam_str = "\n".join(f"  {c} : {code_l1[c]}" for c in sorted(code_l1))

    if args.comparer:
        return comparer(m, leaves, resultat, rapport)

    cibles = articles_juges()
    if args.limite:
        cibles = cibles[:args.limite]
    textes = {}
    for f in sorted({f for f, _ in cibles}):
        try:
            for a in m.extract_articles(f):
                textes[(f, a["id"])] = a["text"]
        except Exception:
            pass
    a_faire = [k for k in cibles if k in textes]
    print(f"  {len(a_faire)} articles · lots de {LOT_ETAGE1} au premier étage")
    print(f"  amorce : {'article entier' if not args.amorce else str(args.amorce)+' jetons'}")
    print(f"  premier étage {m.count_tokens(fam_str):,} jetons de familles, "
          f"contre {m.count_tokens(m.leaves_prompt_str(leaves)):,} pour la liste entière")

    if args.estimer:
        e1 = sum(m.count_tokens(tronquer(m, textes[k], args.amorce)) for k in a_faire)
        e2 = sum(m.count_tokens(textes[k]) for k in a_faire)
        n1 = -(-len(a_faire) // LOT_ETAGE1)
        moy2 = sum(m.count_tokens(m.leaves_prompt_str({c: leaves[c] for c in v}))
                   for v in familles.values()) / len(familles)
        entrant = n1 * (m.count_tokens(m.SYSTEM_PROMPT) + m.count_tokens(fam_str)) + e1 \
            + 12 * (m.count_tokens(m.SYSTEM_PROMPT) + moy2) + e2
        pe, ps = m.PRICING.get(m.MISTRAL_MODEL, (0.50, 1.50))
        print(f"  entrée estimée  {int(entrant):,} jetons")
        print(f"  dépense estimée ${entrant/1e6*pe + 80*len(a_faire)/1e6*ps:.3f}")
        return

    if not m.MISTRAL_API_KEY:
        raise SystemExit("MISTRAL_API_KEY absente.")

    t0 = time.time()
    # Premier étage : la famille, par lots
    par_famille = defaultdict(list)
    echecs = {}
    lots = [a_faire[i:i + LOT_ETAGE1] for i in range(0, len(a_faire), LOT_ETAGE1)]
    for n, lot in enumerate(lots, 1):
        print(f"  étage 1, lot {n}/{len(lots)}", end="\r", file=sys.stderr)
        batch = [{"id": aid, "text": tronquer(m, textes[(f, aid)], args.amorce)}
                 for f, aid in lot]
        try:
            r, _, _ = m.classify_batch(batch, fam_str, list(code_l1))
        except Exception as e:
            for k in lot:
                echecs[k] = f"étage 1 : {str(e)[:120]}"
            continue
        for f, aid in lot:
            fam = [c for c in r.get(aid, []) if c in familles]
            if fam:
                par_famille[fam[0]].append((f, aid))
            else:
                echecs[(f, aid)] = f"famille hors liste : {r.get(aid)}"
    print(" " * 60, file=sys.stderr)
    print(f"  étage 1 : {sum(len(v) for v in par_famille.values())} articles rangés "
          f"dans {len(par_famille)} familles, {len(echecs)} échec(s)")

    # Second étage : les étiquettes, par famille et par lots
    out = {}
    total = sum(-(-len(v) // LOT_ETAGE2) for v in par_famille.values())
    n = 0
    for fam, arts in par_famille.items():
        sous = {c: leaves[c] for c in familles[fam]}
        sous_str = m.leaves_prompt_str(sous)
        for i in range(0, len(arts), LOT_ETAGE2):
            n += 1
            print(f"  étage 2, appel {n}/{total}", end="\r", file=sys.stderr)
            lot = arts[i:i + LOT_ETAGE2]
            batch = [{"id": aid, "text": textes[(f, aid)]} for f, aid in lot]
            try:
                r, _, _ = m.classify_batch(batch, sous_str, list(sous))
            except Exception as e:
                for k in lot:
                    echecs[k] = f"étage 2 : {str(e)[:120]}"
                continue
            for f, aid in lot:
                out[(f, aid)] = {"famille": fam, "famille_label": code_l1[fam],
                                 "codes": [c for c in r.get(aid, []) if c in sous]}
    print(" " * 60, file=sys.stderr)
    for k, e in echecs.items():
        out.setdefault(k, {"erreur": e})
    resultat.write_text(json.dumps({"|".join(k): v for k, v in out.items()},
                                   ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  terminé en {(time.time()-t0)/60:.1f} minutes, "
          f"{sum(1 for v in out.values() if 'codes' in v)} articles classés")
    comparer(m, leaves, resultat, rapport)


def comparer(m, leaves, resultat, rapport):
    """Confronte les sorties de ce régime à celles des régimes déjà mesurés, et
    écrit le rapport. Les taux d'identité et de disjonction disent si deux
    chaînes décrivent le même article de la même manière ; ils ne disent pas
    laquelle a raison, ce que seul le jugement humain établit."""
    if not resultat.exists():
        raise SystemExit("aucun résultat.")
    casc = {tuple(k.split("|")): v
            for k, v in json.loads(resultat.read_text(encoding="utf-8")).items()
            if "codes" in v}
    refs, grp = {}, {}
    for f in sorted(m.OUTPUT_DIR.glob("*_themes.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        fasc = str(d.get("fascicule") or f.stem.split("_")[0])
        for a in d["articles"]:
            grp[(fasc, a["article_id"])] = {t["code"] for t in (a.get("themes") or [])}
    refs["Mistral groupé"] = grp
    for nom, fic in (("Mistral unitaire", "controle_unitaire.json"),
                     ("cascade unitaire", "controle_cascade_mistral.json")):
        p = SORTIE / fic
        if p.exists():
            refs[nom] = {tuple(k.split("|")): set(v["codes"])
                         for k, v in json.loads(p.read_text(encoding="utf-8")).items()
                         if "codes" in v}

    def jac(a, b):
        """Recouvrement de Jaccard entre deux jeux d'étiquettes : la part de ce qu'ils
        ont en commun sur tout ce qu'ils citent ensemble. Deux jeux vides sont tenus
        pour identiques plutôt que pour indéfinis."""
        return len(a & b) / len(a | b) if (a | b) else 1.0

    lignes = ["# Cascade groupée", "", f"{len(casc)} articles.", ""]
    for nom, ref in refs.items():
        comm = sorted(set(casc) & set(ref))
        if not comm:
            continue
        loc = {k: set(casc[k]["codes"]) for k in comm}
        ident = sum(1 for k in comm if loc[k] == ref[k])
        disj = sum(1 for k in comm if not (loc[k] & ref[k]))
        moy = sum(jac(loc[k], ref[k]) for k in comm) / len(comm)
        lignes += [f"## Contre {nom} ({len(comm)} articles)", "",
                   f"- identiques : {ident} ({100*ident//len(comm)}~%)",
                   f"- disjoints : {disj} ({100*disj//len(comm)}~%)",
                   f"- Jaccard moyen : {moy:.3f}", ""]
        print(f"  contre {nom:<20} identiques {ident}/{len(comm)}, "
              f"disjoints {disj}, Jaccard {moy:.3f}")
    n = [len(v["codes"]) for v in casc.values()]
    if n:
        lignes.append(f"La cascade groupée rend {sum(n)/len(n):.2f} étiquettes par article.")
        print(f"  {sum(n)/len(n):.2f} étiquettes par article")
    rapport.write_text("\n".join(lignes) + "\n", encoding="utf-8")
    print(f"  → {rapport}")


if __name__ == "__main__":
    main()
