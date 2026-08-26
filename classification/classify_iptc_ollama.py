"""
classify_iptc_ollama.py : la même classification, par un modèle ouvert en local

POURQUOI CE SCRIPT

La partie du mémoire consacrée au vocabulaire contrôlé repose entièrement sur un
fournisseur commercial, dont la dépendance a été relevée comme une limite
politique : l'indexation d'un fonds patrimonial s'y trouve suspendue à un tarif
et à une disponibilité que l'établissement ne maîtrise pas. Ce script éprouve la
même chaîne sur un modèle ouvert exécuté localement, et fournit le terme de
comparaison qui manquait.

CE QUI EST TENU CONSTANT

L'invite système, la liste des 567 étiquettes, la température et le corpus sont
ceux de classify_iptc_mistral_batched.py, dont ce script importe les fonctions.
Seul le moteur change. La comparaison porte donc sur le modèle, non sur le
protocole.

LA CONTRAINTE DE FENÊTRE

La liste des étiquettes pèse 8 725 jetons, l'invite système 135, et l'article
médian 158. Un appel portant sur un seul article demande donc une fenêtre de
16 384 jetons, que ce script impose par `num_ctx`. Ollama tronque en silence ce
qui dépasse la fenêtre configurée : ne pas la fixer reviendrait à classer un
article amputé sans que rien ne le signale.

LE SCHÉMA DE SORTIE

La chaîne Mistral énumère les codes admissibles dans un schéma strict, de sorte
qu'aucune sortie hors liste n'est possible. Ollama accepte un schéma équivalent,
mais une énumération de 567 valeurs se compile en grammaire, opération coûteuse
sur une machine modeste. Le script tente d'abord le schéma énuméré ; si le
serveur le refuse ou dépasse le délai, il se rabat sur une sortie JSON libre et
filtre les codes inconnus après coup, en consignant combien ont été écartés.
Cette différence est elle-même un résultat à rapporter.

ENTRÉES

  ../config/.env                          rien n'est requis, aucune clé
  ../re-ocr/corpus/reocr_mistral/...      texte ré-océrisé
  verification/cle.json                   les articles déjà jugés

SORTIES

  verification/controle_ollama.json       une entrée par article
  verification/controle_ollama.md         la comparaison avec Mistral

PRÉALABLE

  ollama serve            (le serveur doit tourner)
  ollama pull qwen2.5:3b  (ou le modèle passé par --modele)

PAQUETS EMPLOYÉS

  json, time, argparse, pathlib   bibliothèque standard
  requests                        appels au serveur local

USAGE

    python3 classify_iptc_ollama.py --essai 3     # trois articles, pour voir
    python3 classify_iptc_ollama.py               # les articles déjà jugés
    python3 classify_iptc_ollama.py --intitules   # même chose, en intitulés
    python3 classify_iptc_ollama.py --comparer    # écrit la comparaison
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
RESULTAT = SORTIE / "controle_ollama.json"
RAPPORT = SORTIE / "controle_ollama.md"
RESULTAT_JUS = SORTIE / "controle_ollama_justification.json"
RAPPORT_JUS = SORTIE / "controle_ollama_justification.md"
RESULTAT_LIB = SORTIE / "controle_ollama_intitules.json"
RAPPORT_LIB = SORTIE / "controle_ollama_intitules.md"

URL = "http://localhost:11434/api/chat"
MODELE = "qwen2.5:3b"
FENETRE = 16384
DELAI = 300


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


def schema(codes=None, justification=False):
    """Le schéma de sortie. Avec `codes`, la liste est énumérée et le modèle ne
    peut rendre qu'un code existant ; sans, la sortie est un JSON libre dont les
    codes inconnus seront écartés après coup.

    Avec `justification`, une phrase est demandée AVANT les étiquettes. L'ordre
    des propriétés fixe l'ordre de génération, et c'est tout l'enjeu : une
    justification produite après le choix ne ferait que le rationaliser, quand
    produite avant elle condense l'article en une phrase qui reste sous les yeux
    du modèle au moment de choisir."""
    item = {"type": "string"}
    if codes:
        item = {"type": "string", "enum": list(codes)}
    props, requis = {}, []
    if justification:
        props["justification"] = {"type": "string"}
        requis.append("justification")
    props["themes"] = {"type": "array", "items": item, "minItems": 1, "maxItems": 5}
    requis.append("themes")
    return {"type": "object", "properties": props, "required": requis}


def classer(m, texte, leaves_str, codes, strict, modele, intitules=False,
            justification=False):
    """Un appel, un article. Rend (entrées retenues, écartées, secondes).

    `intitules` change ce qui est demandé au modèle : l'intitulé lui-même plutôt
    que le code qui le désigne. La liste transmise change en conséquence, et la
    comparaison des deux modes isole ce que coûte l'indirection."""
    quoi = "et rends leur intitulé exact" if intitules else "et rends leur code"
    prealable = ("Écris d'abord une phrase disant de quoi cet article traite, puis "
                 if justification else "")
    invite = (f"Étiquettes disponibles :\n{leaves_str}\n\n"
              f"Voici un article à classer. {prealable}"
              f"{'choisis' if justification else 'Choisis'} 1 à 5 étiquettes parmi "
              f"celles listées ci-dessus, {quoi}.\n\n"
              f'"""\n{texte}\n"""')
    charge = {
        "model": modele,
        "messages": [{"role": "system", "content": m.SYSTEM_PROMPT},
                     {"role": "user", "content": invite}],
        "stream": False,
        "format": schema(codes if strict else None, justification),
        "options": {"temperature": 0.0, "num_ctx": FENETRE},
    }
    t0 = time.perf_counter()
    r = requests.post(URL, json=charge, timeout=DELAI)
    r.raise_for_status()
    secondes = time.perf_counter() - t0
    d = json.loads(r.json()["message"]["content"])
    rendus = d.get("themes", [])
    retenus = [c for c in rendus if c in codes]
    return (retenus, [c for c in rendus if c not in codes], secondes,
            d.get("justification", ""))


def main():
    """Point d'entrée : soumet chaque article au modèle local avec la liste
    entière, puis compare les étiquettes obtenues à celles du service
    commercial."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--modele", default=MODELE)
    ap.add_argument("--essai", type=int, default=0,
                    help="ne traiter que n articles, pour éprouver la chaîne")
    ap.add_argument("--souple", action="store_true",
                    help="ne pas énumérer les codes dans le schéma")
    ap.add_argument("--intitules", action="store_true",
                    help="transmettre et demander des intitulés plutôt que des codes, "
                         "pour isoler le coût de l'indirection code-libellé")
    ap.add_argument("--justification", action="store_true",
                    help="demander une phrase de justification AVANT les étiquettes")
    ap.add_argument("--comparer", action="store_true")
    args = ap.parse_args()

    global RESULTAT, RAPPORT
    if args.intitules:
        RESULTAT, RAPPORT = RESULTAT_LIB, RAPPORT_LIB
    if args.justification:
        RESULTAT, RAPPORT = RESULTAT_JUS, RAPPORT_JUS
    m = _module()
    if args.comparer:
        return comparer(m, args.modele)

    try:
        dispo = requests.get("http://localhost:11434/api/tags", timeout=10).json()
    except Exception:
        raise SystemExit("serveur Ollama injoignable : lancer « ollama serve ».")
    noms = [x["name"] for x in dispo.get("models", [])]
    if not any(n.startswith(args.modele.split(":")[0]) for n in noms):
        raise SystemExit(f"modèle {args.modele} absent : lancer « ollama pull {args.modele} ». "
                         f"Modèles présents : {noms or 'aucun'}")

    leaves = m.build_leaves(m.TAXONOMY_PATH)
    codes = set(leaves)
    if args.intitules:
        # Cinq intitulés sont portés par deux codes ; on les retient tous les
        # deux, la comparaison tenant l'attribution pour juste si l'un d'eux
        # correspond.
        par_libelle = {}
        for c, v in leaves.items():
            par_libelle.setdefault(v["label_fr"], []).append(c)
        leaves_str = "\n".join(sorted(par_libelle))
        admis = set(par_libelle)
    else:
        leaves_str = m.leaves_prompt_str(leaves)
        admis = codes
    print(f"  modèle {args.modele}, fenêtre {FENETRE:,} jetons, "
          f"réponse en {'intitulés' if args.intitules else 'codes'}")
    print(f"  invite fixe {m.count_tokens(m.SYSTEM_PROMPT)+m.count_tokens(leaves_str):,} jetons, "
          f"dont {m.count_tokens(leaves_str):,} pour les {len(admis)} entrées")

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
        print(f"  {len(deja)} articles déjà obtenus, repris tels quels")

    strict = not args.souple
    reste = [k for k in a_faire if k not in deja]
    t0 = time.time()
    for n, (fasc, aid) in enumerate(reste, 1):
        print(f"  appel {n}/{len(reste)} : {fasc}/{aid}", end="\r", file=sys.stderr)
        try:
            retenus, ecartes, secondes, jus = classer(
                m, textes[(fasc, aid)], leaves_str, admis, strict, args.modele,
                args.intitules, args.justification)
            if args.intitules:
                rendus = [c for lab in retenus for c in par_libelle.get(lab, [])]
                deja[(fasc, aid)] = {"codes": rendus, "intitules": retenus,
                                     "hors_liste": ecartes, "secondes": round(secondes, 1)}
            else:
                deja[(fasc, aid)] = {"codes": retenus, "hors_liste": ecartes,
                                     "secondes": round(secondes, 1), "justification": jus}
        except Exception as e:
            if strict and n == 1:
                print(f"\n  schéma énuméré refusé ({str(e)[:70]}), repli sur JSON libre",
                      file=sys.stderr)
                strict = False
                try:
                    retenus, ecartes, secondes, jus = classer(
                        m, textes[(fasc, aid)], leaves_str, admis, False, args.modele,
                        args.intitules, args.justification)
                    rendus = ([c for lab in retenus for c in par_libelle.get(lab, [])]
                              if args.intitules else retenus)
                    deja[(fasc, aid)] = {"codes": rendus, "hors_liste": ecartes,
                                         "secondes": round(secondes, 1), "justification": jus}
                    continue
                except Exception as e2:
                    e = e2
            deja[(fasc, aid)] = {"erreur": str(e)[:200]}
        if n % 5 == 0 or n == len(reste):
            RESULTAT.write_text(json.dumps(
                {"|".join(k): v for k, v in deja.items()}, ensure_ascii=False, indent=1),
                encoding="utf-8")
    print(" " * 70, file=sys.stderr)
    ok = [v for v in deja.values() if "codes" in v]
    if ok:
        secs = sorted(v["secondes"] for v in ok if "secondes" in v)
        print(f"  {len(ok)} articles classés en {(time.time()-t0)/60:.1f} minutes, "
              f"médiane {secs[len(secs)//2]:.1f}~s par article")
        hors = sum(len(v.get("hors_liste", [])) for v in ok)
        print(f"  schéma {'énuméré' if strict else 'libre'} ; "
              f"{hors} code(s) hors taxinomie écarté(s)")
    comparer(m, args.modele)


def comparer(m, modele):
    """Confronte les étiquettes locales à celles de Mistral, groupé et unitaire."""
    if not RESULTAT.exists():
        raise SystemExit("aucun résultat local.")
    loc = {tuple(k.split("|")): set(v["codes"])
           for k, v in json.loads(RESULTAT.read_text(encoding="utf-8")).items()
           if "codes" in v}
    grp = {}
    for f in sorted(m.OUTPUT_DIR.glob("*_themes.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        fasc = str(d.get("fascicule") or f.stem.split("_")[0])
        for a in d["articles"]:
            grp[(fasc, a["article_id"])] = {t["code"] for t in (a.get("themes") or [])}
    uni = {}
    p = SORTIE / "controle_unitaire.json"
    if p.exists():
        uni = {tuple(k.split("|")): set(v["codes"])
               for k, v in json.loads(p.read_text(encoding="utf-8")).items() if "codes" in v}

    def jaccard(a, b):
        """Recouvrement de Jaccard entre deux jeux d'étiquettes : la part de ce qu'ils
        ont en commun sur tout ce qu'ils citent ensemble. Deux jeux vides sont tenus
        pour identiques plutôt que pour indéfinis."""
        return len(a & b) / len(a | b) if (a | b) else 1.0

    lignes = ["# Classification locale contre Mistral", "",
              f"Modèle local {modele}, fenêtre {FENETRE:,} jetons, température nulle.", ""]
    for lib, ref in (("Mistral groupé", grp), ("Mistral unitaire", uni)):
        comm = sorted(set(loc) & set(ref))
        if not comm:
            continue
        ident = sum(1 for k in comm if loc[k] == ref[k])
        disj = sum(1 for k in comm if not (loc[k] & ref[k]))
        moy = sum(jaccard(loc[k], ref[k]) for k in comm) / len(comm)
        lignes += [f"## Contre {lib} ({len(comm)} articles)", "",
                   f"- étiquettes identiques : {ident} ({100*ident//len(comm)}~%)",
                   f"- aucune étiquette commune : {disj} ({100*disj//len(comm)}~%)",
                   f"- recouvrement de Jaccard moyen : {moy:.3f}", ""]
        print(f"  contre {lib:<18} identiques {ident}/{len(comm)}, "
              f"disjoints {disj}, Jaccard {moy:.3f}")
    n_et = [len(v) for v in loc.values()]
    lignes += [f"Le modèle local rend {sum(n_et)/len(n_et):.2f} étiquettes par article.", ""]
    RAPPORT.write_text("\n".join(lignes), encoding="utf-8")
    print(f"  → {RAPPORT}")


if __name__ == "__main__":
    main()
