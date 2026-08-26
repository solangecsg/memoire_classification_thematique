"""
classify_iptc_ollama_cascade.py : la classification locale en deux étages

POURQUOI CE SCRIPT

L'essai conduit le 24 août 2026 sur qwen2.5:3b a montré que le modèle comprend
les textes du corpus et échoue pourtant à les classer : l'invite pèse
8 860 jetons dont 8 725 pour la liste des 567 étiquettes, quand l'article médian
en compte 158, de sorte que le texte à classer représente moins de deux pour
cent de ce que le modèle lit. Présenté à une liste de vingt intitulés, le même
modèle rend les étiquettes de Mistral.

La cascade répond à ce diagnostic. Elle soumet d'abord les dix-sept familles de
premier niveau, puis les seules étiquettes de la famille retenue, vingt-six en
médiane. La liste transmise au second appel devient ainsi cent fois plus courte.

Cette conception avait été écartée sur l'interface commerciale, où deux appels
reviennent plus cher qu'un seul : $4,57 contre $3,63. La facturation au jeton
disparaissant en local, l'argument tombe et la cascade redevient praticable.

CE QU'IL FAUT EN MESURER

Le premier étage devient le point critique : une famille mal choisie condamne le
second appel, qui ne peut plus rattraper. Le script consigne donc la famille
retenue pour chaque article, de sorte que l'erreur puisse être imputée à l'un ou
l'autre étage.

ENTRÉES ET SORTIES

  verification/cle.json                  les articles déjà jugés
  verification/controle_ollama_cascade.json
  verification/controle_ollama_cascade.md

PAQUETS EMPLOYÉS

  json, time, argparse, pathlib   bibliothèque standard
  requests                        appels au serveur local

USAGE

    python3 classify_iptc_ollama_cascade.py --essai 5
    python3 classify_iptc_ollama_cascade.py
"""

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import requests

ICI = Path(__file__).resolve().parent
SORTIE = ICI / "verification"
RESULTAT = SORTIE / "controle_ollama_cascade.json"
RAPPORT = SORTIE / "controle_ollama_cascade.md"
RESULTAT_JUS = SORTIE / "controle_ollama_cascade_justification.json"
RAPPORT_JUS = SORTIE / "controle_ollama_cascade_justification.md"

URL = "http://localhost:11434/api/chat"
MODELE = "qwen2.5:3b"
FENETRE = 16384


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


def appel(m, systeme, invite, choix, modele, maxi=3, justification=False):
    """Un appel contraint à choisir parmi `choix`, énuméré dans le schéma.

    Avec `justification`, une phrase est demandée avant les étiquettes. L'ordre
    des propriétés fixe l'ordre de génération : produite avant le choix, la
    phrase condense l'article et reste sous les yeux du modèle au moment de
    choisir ; produite après, elle ne ferait que le rationaliser."""
    props, requis = {}, []
    if justification:
        props["justification"] = {"type": "string"}
        requis.append("justification")
    props["themes"] = {"type": "array", "items": {"type": "string", "enum": list(choix)},
                       "minItems": 1, "maxItems": maxi}
    requis.append("themes")
    sch = {"type": "object", "properties": props, "required": requis}
    r = requests.post(URL, timeout=300, json={
        "model": modele, "stream": False, "format": sch,
        "messages": [{"role": "system", "content": systeme},
                     {"role": "user", "content": invite}],
        "options": {"temperature": 0.0, "num_ctx": FENETRE}})
    r.raise_for_status()
    d = json.loads(r.json()["message"]["content"])
    return d.get("themes", []), d.get("justification", "")


def main():
    """Point d'entrée : conduit la cascade à deux étages sur chaque article, en
    consignant la famille retenue au premier, de sorte que l'erreur puisse
    être imputée à l'un ou l'autre étage."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--modele", default=MODELE)
    ap.add_argument("--essai", type=int, default=0)
    ap.add_argument("--justification", action="store_true",
                    help="demander une phrase de justification AVANT les étiquettes, "
                         "aux deux étages")
    args = ap.parse_args()

    global RESULTAT, RAPPORT
    if args.justification:
        RESULTAT, RAPPORT = RESULTAT_JUS, RAPPORT_JUS
    m = _module()
    leaves = m.build_leaves(m.TAXONOMY_PATH)
    familles = {}
    for c, v in leaves.items():
        familles.setdefault(v["l1_label"], []).append((c, v["label_fr"]))
    tailles = sorted(len(x) for x in familles.values())
    print(f"  {len(familles)} familles de premier niveau, de {tailles[0]} à "
          f"{tailles[-1]} étiquettes, médiane {tailles[len(tailles)//2]}")
    fam_str = "\n".join(sorted(familles))
    print(f"  premier étage : {m.count_tokens(fam_str):,} jetons, "
          f"contre {m.count_tokens(m.leaves_prompt_str(leaves)):,} pour la liste entière")

    cibles = articles_juges()
    if args.essai:
        cibles = cibles[:args.essai]
    textes = {}
    for f in sorted({f for f, _ in cibles}):
        try:
            for a in m.extract_articles(f):
                textes[(f, a["id"])] = a["text"]
        except Exception:
            pass
    a_faire = [k for k in cibles if k in textes]

    deja = {}
    if RESULTAT.exists() and not args.essai:
        deja = {tuple(k.split("|")): v
                for k, v in json.loads(RESULTAT.read_text(encoding="utf-8")).items()
                if "codes" in v}
        print(f"  {len(deja)} articles déjà obtenus")

    # correspondance intitulé vers code, par famille
    code_de = {f: {lab: c for c, lab in v} for f, v in familles.items()}
    reste = [k for k in a_faire if k not in deja]
    t0 = time.time()
    for n, (fasc, aid) in enumerate(reste, 1):
        print(f"  appel {n}/{len(reste)} : {fasc}/{aid}", end="\r", file=sys.stderr)
        texte = textes[(fasc, aid)]
        try:
            fam, jus1 = appel(m, m.SYSTEM_PROMPT,
                              f"Familles :\n{fam_str}\n\nÀ quelle famille cet article "
                              f"appartient-il ? Une seule.\n\n\"\"\"\n{texte}\n\"\"\"",
                              sorted(familles), args.modele, maxi=1, justification=args.justification)
            f0 = next((x for x in fam if x in familles), None)
            if f0 is None:
                deja[(fasc, aid)] = {"erreur": f"famille hors liste : {fam}"}
                continue
            enf = sorted(lab for _, lab in familles[f0])
            lab, jus2 = appel(m, m.SYSTEM_PROMPT,
                              "Étiquettes disponibles :\n" + "\n".join(enf) +
                              f"\n\nClasse cet article avec 1 à 3 de ces étiquettes."
                              f"\n\n\"\"\"\n{texte}\n\"\"\"",
                              enf, args.modele, justification=args.justification)
            deja[(fasc, aid)] = {"famille": f0,
                                 "codes": [code_de[f0][x] for x in lab if x in code_de[f0]],
                                 "intitules": lab,
                                 "justification_famille": jus1,
                                 "justification_etiquette": jus2}
        except Exception as e:
            deja[(fasc, aid)] = {"erreur": str(e)[:200]}
        if n % 5 == 0 or n == len(reste):
            RESULTAT.write_text(json.dumps(
                {"|".join(k): v for k, v in deja.items()}, ensure_ascii=False, indent=1),
                encoding="utf-8")
    print(" " * 70, file=sys.stderr)
    print(f"  terminé en {(time.time()-t0)/60:.1f} minutes")
    comparer(m, args.modele, leaves)


def comparer(m, modele, leaves):
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
    for nom, fic in (("Mistral unitaire", "controle_unitaire.json"),
                     ("local, liste entière", "controle_ollama.json")):
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

    lignes = ["# Cascade locale à deux étages", "",
              f"Modèle {modele}. {len(casc)} articles.", ""]
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
        print(f"  contre {nom:<22} identiques {ident}/{len(comm)}, "
              f"disjoints {disj}, Jaccard {moy:.3f}")
    # part de l'erreur imputable au premier étage
    if grp:
        bonne, total = 0, 0
        for k, v in casc.items():
            if k not in grp or not grp[k]:
                continue
            total += 1
            fam_ref = {leaves[c]["l1_label"] for c in grp[k] if c in leaves}
            bonne += v.get("famille") in fam_ref
        if total:
            lignes += [f"Le premier étage retient une famille que Mistral emploie aussi "
                       f"dans {bonne} cas sur {total}, soit {100*bonne//total}~%.", ""]
            print(f"  premier étage en accord de famille : {bonne}/{total} "
                  f"({100*bonne//total}~%)")
    RAPPORT.write_text("\n".join(lignes), encoding="utf-8")
    print(f"  → {RAPPORT}")


if __name__ == "__main__":
    main()
