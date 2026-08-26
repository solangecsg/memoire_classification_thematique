"""
controle_justification_mistral.py : une phrase de justification avant l'étiquette

POURQUOI CE SCRIPT

Un essai conduit le 25 août 2026 sur le modèle local a montré qu'une phrase de
justification demandée AVANT les étiquettes transforme ses sorties. Une cote de
bourse qui recevait « Homicide » et « Crime de guerre » reçoit « Médicaments »
dès lors que le modèle écrit d'abord de quoi l'article traite. Le mécanisme se
comprend : la phrase condense l'article et reste sous les yeux du modèle au
moment de choisir, là où le texte d'origine était noyé sous la liste.

Reste à savoir si le procédé vaut aussi pour un grand modèle, dont l'essai a
montré que la longueur de la liste ne le gêne pas. Le plafond de justesse y est
de 47,8 pour cent au régime groupé, 47,5 en cascade, 55,5 estimé en appel
unitaire ; la question est de savoir si la justification le déplace.

L'ORDRE DES CHAMPS DÉCIDE DE TOUT

Le schéma de sortie place `justification` avant `themes`, et l'ordre des
propriétés fixe l'ordre de génération. Produite après le choix, une
justification ne ferait que le rationaliser ; produite avant, elle le construit.

CE QUE CELA COÛTE

La justification ajoute une trentaine de jetons de sortie par appel, quand
l'entrée en pèse 9 456. Le surcoût mesuré est de 1 pour cent en appel unitaire
et de 8 pour cent en cascade, où elle se paie aux deux étages.

ENTRÉES ET SORTIES

  verification/cle.json                            les articles déjà jugés
  config/.env                                      MISTRAL_API_KEY
  verification/controle_justification_{mode}.json
  verification/controle_justification_{mode}.md

PAQUETS EMPLOYÉS

  json, time, argparse, pathlib   bibliothèque standard
  requests                        appel direct, le schéma différant de celui
                                  du script de campagne
  les fonctions de classify_iptc_mistral_batched, importées

USAGE

    python3 controle_justification_mistral.py --estimer
    python3 controle_justification_mistral.py --mode unitaire
    python3 controle_justification_mistral.py --mode cascade
"""

import argparse
import importlib.util
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import requests

ICI = Path(__file__).resolve().parent
SORTIE = ICI / "verification"
URL = "https://api.mistral.ai/v1/chat/completions"


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


def schema(codes, maxi):
    """Sortie contrainte, la justification précédant les étiquettes."""
    return {"type": "json_schema", "json_schema": {
        "name": "iptc_avec_justification",
        "schema": {"type": "object",
                   "properties": {
                       "justification": {"type": "string"},
                       "themes": {"type": "array",
                                  "items": {"type": "string", "enum": list(codes)},
                                  "minItems": 1, "maxItems": maxi}},
                   "required": ["justification", "themes"],
                   "additionalProperties": False},
        "strict": True}}


def appel(m, liste_str, texte, codes, maxi=5, retries=5):
    """Un appel dont le schéma place `justification` avant `themes`. L'ordre des
    propriétés fixant l'ordre de génération, la phrase est produite avant le
    choix et le construit, là où produite après elle ne ferait que le
    rationaliser."""
    invite = (f"Étiquettes disponibles :\n{liste_str}\n\n"
              "Voici un article à classer. Écris d'abord une phrase disant de quoi il "
              f"traite, puis choisis 1 à {maxi} étiquettes parmi celles listées "
              f"ci-dessus et rends leur code.\n\n\"\"\"\n{texte}\n\"\"\"")
    charge = {"model": m.MISTRAL_MODEL,
              "messages": [{"role": "system", "content": m.SYSTEM_PROMPT},
                           {"role": "user", "content": invite}],
              "temperature": 0.0,
              "response_format": schema(codes, maxi)}
    entetes = {"Content-Type": "application/json",
               "Authorization": f"Bearer {m.MISTRAL_API_KEY}"}
    derniere = None
    for essai in range(retries):
        try:
            r = requests.post(URL, json=charge, headers=entetes, timeout=180)
            if r.status_code in (401, 403):
                raise SystemExit("authentification refusée par Mistral.")
            if r.status_code == 200:
                d = r.json()
                c = json.loads(d["choices"][0]["message"]["content"])
                return ([x for x in c.get("themes", []) if x in codes],
                        c.get("justification", ""), d.get("usage", {}))
            derniere = f"HTTP {r.status_code}"
        except SystemExit:
            raise
        except Exception as e:
            derniere = str(e)[:120]
        if essai < retries - 1:
            time.sleep(2 * (essai + 1))
    raise RuntimeError(f"échec après {retries} tentatives : {derniere}")


def main():
    """Point d'entrée : reprend les articles déjà jugés en demandant une phrase de
    justification avant les étiquettes, en appel unitaire ou en cascade."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--mode", choices=("unitaire", "cascade"), default="unitaire")
    ap.add_argument("--estimer", action="store_true")
    ap.add_argument("--limite", type=int, default=0)
    ap.add_argument("--comparer", action="store_true")
    args = ap.parse_args()

    resultat = SORTIE / f"controle_justification_{args.mode}.json"
    rapport = resultat.with_suffix(".md")
    m = _module()
    leaves = m.build_leaves(m.TAXONOMY_PATH)
    if args.comparer:
        return comparer(m, resultat, rapport, args.mode)

    familles, code_l1 = defaultdict(list), {}
    for c, v in leaves.items():
        familles[v["l1_code"]].append(c)
        code_l1[v["l1_code"]] = v["l1_label"]
    fam_str = "\n".join(f"  {c} : {code_l1[c]}" for c in sorted(code_l1))
    liste_str = m.leaves_prompt_str(leaves)

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
    print(f"  mode {args.mode} · {len(a_faire)} articles · modèle {m.MISTRAL_MODEL}")

    if args.estimer:
        pe, ps = m.PRICING.get(m.MISTRAL_MODEL, (0.50, 1.50))
        if args.mode == "unitaire":
            e = sum(m.count_tokens(m.SYSTEM_PROMPT) + m.count_tokens(liste_str)
                    + m.count_tokens(textes[k]) for k in a_faire)
            n_app = len(a_faire)
        else:
            moy2 = sum(m.count_tokens(m.leaves_prompt_str({c: leaves[c] for c in v}))
                       for v in familles.values()) / len(familles)
            e = sum(2 * m.count_tokens(m.SYSTEM_PROMPT) + m.count_tokens(fam_str) + moy2
                    + 2 * m.count_tokens(textes[k]) for k in a_faire)
            n_app = 2 * len(a_faire)
        print(f"  entrée estimée  {int(e):,} jetons · {n_app} appels")
        print(f"  dépense estimée ${e/1e6*pe + 100*n_app/1e6*ps:.3f}")
        return

    if not m.MISTRAL_API_KEY:
        raise SystemExit("MISTRAL_API_KEY absente.")

    deja = {}
    if resultat.exists():
        deja = {tuple(k.split("|")): v
                for k, v in json.loads(resultat.read_text(encoding="utf-8")).items()
                if "codes" in v}
        print(f"  {len(deja)} articles déjà obtenus")
    reste = [k for k in a_faire if k not in deja]
    t0 = time.time()
    for n, (fasc, aid) in enumerate(reste, 1):
        print(f"  appel {n}/{len(reste)} : {fasc}/{aid}", end="\r", file=sys.stderr)
        texte = textes[(fasc, aid)]
        try:
            if args.mode == "unitaire":
                codes, jus, usage = appel(m, liste_str, texte, set(leaves))
                deja[(fasc, aid)] = {"codes": codes, "justification": jus,
                                     "jetons": usage.get("total_tokens")}
            else:
                fam, jus1, u1 = appel(m, fam_str, texte, set(code_l1), maxi=1)
                if not fam:
                    deja[(fasc, aid)] = {"erreur": "famille hors liste"}
                    continue
                sous = {c: leaves[c] for c in familles[fam[0]]}
                codes, jus2, u2 = appel(m, m.leaves_prompt_str(sous), texte, set(sous), maxi=3)
                deja[(fasc, aid)] = {"famille": fam[0], "codes": codes,
                                     "justification_famille": jus1,
                                     "justification": jus2,
                                     "jetons": (u1.get("total_tokens", 0)
                                                + u2.get("total_tokens", 0))}
        except Exception as e:
            deja[(fasc, aid)] = {"erreur": str(e)[:200]}
        if n % 10 == 0 or n == len(reste):
            resultat.write_text(json.dumps(
                {"|".join(k): v for k, v in deja.items()}, ensure_ascii=False, indent=1),
                encoding="utf-8")
    print(" " * 70, file=sys.stderr)
    print(f"  terminé en {(time.time()-t0)/60:.1f} minutes")
    comparer(m, resultat, rapport, args.mode)


def comparer(m, resultat, rapport, mode):
    """Confronte les sorties de ce régime à celles des régimes déjà mesurés, et
    écrit le rapport. Les taux d'identité et de disjonction disent si deux
    chaînes décrivent le même article de la même manière ; ils ne disent pas
    laquelle a raison, ce que seul le jugement humain établit."""
    if not resultat.exists():
        raise SystemExit("aucun résultat.")
    jus = {tuple(k.split("|")): v
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
                     ("cascade sans justification", "controle_cascade_mistral.json")):
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

    lignes = [f"# Justification avant l'étiquette, mode {mode}", "",
              f"{len(jus)} articles.", ""]
    for nom, ref in refs.items():
        comm = sorted(set(jus) & set(ref))
        if not comm:
            continue
        loc = {k: set(jus[k]["codes"]) for k in comm}
        ident = sum(1 for k in comm if loc[k] == ref[k])
        disj = sum(1 for k in comm if not (loc[k] & ref[k]))
        moy = sum(jac(loc[k], ref[k]) for k in comm) / len(comm)
        lignes += [f"## Contre {nom} ({len(comm)} articles)", "",
                   f"- identiques : {ident} ({100*ident//len(comm)}~%)",
                   f"- disjoints : {disj} ({100*disj//len(comm)}~%)",
                   f"- Jaccard moyen : {moy:.3f}", ""]
        print(f"  contre {nom:<28} identiques {ident}/{len(comm)}, "
              f"disjoints {disj}, Jaccard {moy:.3f}")
    n = [len(v["codes"]) for v in jus.values()]
    if n:
        lignes.append(f"Le régime rend {sum(n)/len(n):.2f} étiquettes par article.")
        print(f"  {sum(n)/len(n):.2f} étiquettes par article")
    rapport.write_text("\n".join(lignes) + "\n", encoding="utf-8")
    print(f"  → {rapport}")


if __name__ == "__main__":
    main()
