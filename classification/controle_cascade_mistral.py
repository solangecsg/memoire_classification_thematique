"""
controle_cascade_mistral.py : la cascade à deux étages, sur le modèle commercial

POURQUOI CE SCRIPT

La cascade avait été écartée sur un seul critère, la dépense : $4,57 contre
$3,63 pour le groupage simple, réduire la liste ne réduisant pas le texte qu'il
faut de toute façon transmettre. Sa qualité n'a jamais été mesurée.

La question se pose autrement depuis deux constats. L'appel unitaire rend 55,5
pour cent d'étiquettes acceptables contre 47,8 au régime groupé, et le gain vient
du tri plutôt que des ajouts. Et l'essai local a montré que raccourcir la liste
transforme les sorties d'un petit modèle, ce qui invite à demander ce qu'elle
fait sur un grand.

La cascade soumet d'abord les dix-sept familles de premier niveau, puis les
seules étiquettes de la famille retenue, vingt-six en médiane. Le second appel
voit donc une liste cent fois plus courte, et l'article y pèse d'autant plus.

CE QUI EST TENU CONSTANT

Le modèle, la température, l'invite système et le corpus sont ceux des autres
contrôles. Un article par appel des deux côtés, de sorte que la comparaison avec
le régime unitaire porte sur la seule structure de l'invite.

CE QU'IL FAUT EN MESURER

Le premier étage devient le point critique : une famille mal choisie condamne le
second appel. Le script consigne la famille retenue, et le dépouillement rapporte
la part des cas où elle coïncide avec celle des étiquettes du régime unitaire.

ENTRÉES ET SORTIES

  verification/cle.json                        les articles déjà jugés
  config/.env                                  MISTRAL_API_KEY
  verification/controle_cascade_mistral.json
  verification/controle_cascade_mistral.md

PAQUETS EMPLOYÉS

  json, time, argparse, pathlib   bibliothèque standard
  les fonctions de classify_iptc_mistral_batched, importées

USAGE

    python3 controle_cascade_mistral.py --estimer
    python3 controle_cascade_mistral.py
"""

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

ICI = Path(__file__).resolve().parent
SORTIE = ICI / "verification"
RESULTAT = SORTIE / "controle_cascade_mistral.json"
RAPPORT = SORTIE / "controle_cascade_mistral.md"


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


def main():
    """Point d'entrée : conduit la cascade commerciale à deux étages, un appel de
    famille puis un appel d'étiquette sur la seule branche retenue."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--estimer", action="store_true")
    ap.add_argument("--limite", type=int, default=0)
    ap.add_argument("--comparer", action="store_true")
    args = ap.parse_args()

    m = _module()
    leaves = m.build_leaves(m.TAXONOMY_PATH)
    if args.comparer:
        return comparer(m, leaves)

    # Les familles de premier niveau, et le pseudo-code qui les désigne. Le
    # schéma strict énumère des chaînes ; on emploie le code de niveau 1.
    familles, code_l1 = {}, {}
    for c, v in leaves.items():
        familles.setdefault(v["l1_code"], []).append(c)
        code_l1[v["l1_code"]] = v["l1_label"]
    fam_str = "\n".join(f"  {c} : {code_l1[c]}" for c in sorted(code_l1))
    tailles = sorted(len(x) for x in familles.values())
    print(f"  {len(familles)} familles, de {tailles[0]} à {tailles[-1]} étiquettes, "
          f"médiane {tailles[len(tailles)//2]}")
    print(f"  premier étage : {m.count_tokens(fam_str):,} jetons, contre "
          f"{m.count_tokens(m.leaves_prompt_str(leaves)):,} pour la liste entière")

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

    # Budget : premier étage à liste courte, second à liste de branche.
    fixe1 = m.count_tokens(m.SYSTEM_PROMPT) + m.count_tokens(fam_str) + 100
    moy2 = sum(m.count_tokens(m.leaves_prompt_str({c: leaves[c] for c in v}))
               for v in familles.values()) / len(familles)
    fixe2 = m.count_tokens(m.SYSTEM_PROMPT) + moy2 + 100
    entrant = sum(fixe1 + fixe2 + 2 * m.count_tokens(textes[k]) for k in a_faire)
    pe, ps = m.PRICING.get(m.MISTRAL_MODEL, (0.50, 1.50))
    cout = entrant / 1e6 * pe + 80 * len(a_faire) / 1e6 * ps
    print(f"  {len(a_faire)} articles, deux appels chacun")
    print(f"  entrée estimée  {int(entrant):,} jetons")
    print(f"  dépense estimée ${cout:.2f}   (l'appel unitaire à liste entière coûtait $0,81)")
    if args.estimer:
        return
    if not m.MISTRAL_API_KEY:
        raise SystemExit("MISTRAL_API_KEY absente.")

    deja = {}
    if RESULTAT.exists():
        deja = {tuple(k.split("|")): v
                for k, v in json.loads(RESULTAT.read_text(encoding="utf-8")).items()
                if "codes" in v}
        print(f"  {len(deja)} articles déjà obtenus")

    reste = [k for k in a_faire if k not in deja]
    t0 = time.time()
    for n, (fasc, aid) in enumerate(reste, 1):
        print(f"  appel {n}/{len(reste)} : {fasc}/{aid}", end="\r", file=sys.stderr)
        art = [{"id": aid, "text": textes[(fasc, aid)]}]
        try:
            r1, _, _ = m.classify_batch(art, fam_str, list(code_l1))
            fam = [c for c in r1.get(aid, []) if c in familles]
            if not fam:
                deja[(fasc, aid)] = {"erreur": f"famille hors liste : {r1.get(aid)}"}
                continue
            f0 = fam[0]
            sous = {c: leaves[c] for c in familles[f0]}
            r2, _, _ = m.classify_batch(art, m.leaves_prompt_str(sous), list(sous))
            deja[(fasc, aid)] = {"famille": f0, "famille_label": code_l1[f0],
                                 "codes": [c for c in r2.get(aid, []) if c in sous]}
        except Exception as e:
            deja[(fasc, aid)] = {"erreur": str(e)[:200]}
        if n % 10 == 0 or n == len(reste):
            RESULTAT.write_text(json.dumps(
                {"|".join(k): v for k, v in deja.items()}, ensure_ascii=False, indent=1),
                encoding="utf-8")
    print(" " * 70, file=sys.stderr)
    print(f"  terminé en {(time.time()-t0)/60:.1f} minutes")
    comparer(m, leaves)


def comparer(m, leaves):
    """Confronte les sorties de ce régime à celles des régimes déjà mesurés, et
    écrit le rapport. Les taux d'identité et de disjonction disent si deux
    chaînes décrivent le même article de la même manière ; ils ne disent pas
    laquelle a raison, ce que seul le jugement humain établit."""
    if not RESULTAT.exists():
        raise SystemExit("aucun résultat.")
    casc = {tuple(k.split("|")): v
            for k, v in json.loads(RESULTAT.read_text(encoding="utf-8")).items()
            if "codes" in v}
    refs = {}
    grp = {}
    for f in sorted(m.OUTPUT_DIR.glob("*_themes.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        fasc = str(d.get("fascicule") or f.stem.split("_")[0])
        for a in d["articles"]:
            grp[(fasc, a["article_id"])] = {t["code"] for t in (a.get("themes") or [])}
    refs["Mistral groupé"] = grp
    p = SORTIE / "controle_unitaire.json"
    if p.exists():
        refs["Mistral unitaire"] = {tuple(k.split("|")): set(v["codes"])
                                    for k, v in json.loads(p.read_text(encoding="utf-8")).items()
                                    if "codes" in v}

    def jac(a, b):
        """Recouvrement de Jaccard entre deux jeux d'étiquettes : la part de ce qu'ils
        ont en commun sur tout ce qu'ils citent ensemble. Deux jeux vides sont tenus
        pour identiques plutôt que pour indéfinis."""
        return len(a & b) / len(a | b) if (a | b) else 1.0

    lignes = ["# Cascade à deux étages, modèle commercial", "",
              f"{len(casc)} articles, un par appel, deux appels chacun.", ""]
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
        print(f"  contre {nom:<18} identiques {ident}/{len(comm)}, "
              f"disjoints {disj}, Jaccard {moy:.3f}")
    bonne = total = 0
    for k, v in casc.items():
        if k not in grp or not grp[k]:
            continue
        total += 1
        bonne += v.get("famille") in {leaves[c]["l1_code"] for c in grp[k] if c in leaves}
    if total:
        lignes += [f"Le premier étage retient une famille que le régime groupé emploie "
                   f"aussi dans {bonne} cas sur {total}, soit {100*bonne//total}~%.", ""]
        print(f"  premier étage en accord de famille : {bonne}/{total} ({100*bonne//total}~%)")
    n = [len(v["codes"]) for v in casc.values()]
    lignes += [f"La cascade rend {sum(n)/len(n):.2f} étiquettes par article.", ""]
    print(f"  {sum(n)/len(n):.2f} étiquettes par article")
    RAPPORT.write_text("\n".join(lignes), encoding="utf-8")
    print(f"  → {RAPPORT}")


if __name__ == "__main__":
    main()
