"""
controle_appel_unitaire.py : reprend un article par appel, pour éprouver le groupage

POURQUOI CE SCRIPT

La campagne de classification a réuni jusqu'à vingt-cinq articles dans un même
appel, ce qui divise la dépense par neuf. La vérification humaine conduite le
23 août 2026 a montré que la position de l'article dans son lot ne prédit pas la
justesse de son étiquette, sur cent onze items et sans qu'aucun test ne détecte
de différence entre les trois tiers. Le corpus ne permettait pas, en revanche,
de conclure sur la taille du lot elle-même, les articles ayant presque tous été
traités dans des lots de treize à vingt-cinq.

Ce script lève la réserve. Il reprend les articles déjà jugés, un par appel,
avec la même invite, le même référentiel et le même modèle, et compare les
étiquettes obtenues à celles du traitement groupé. Si elles coïncident, le
groupage est hors de cause. Si elles divergent, une seconde épreuve en aveugle
sur les sorties individuelles devient justifiée.

CE QUI EST TENU CONSTANT

Tout, hors la composition des lots. L'invite système, la liste des étiquettes,
le modèle, la température et le schéma de sortie sont ceux de
classify_iptc_mistral_batched.py, dont ce script importe les fonctions plutôt
que d'en recopier le comportement.

ENTRÉES

  verification/cle.json                              les items jugés
  ../re-ocr/corpus/reocr_mistral/{fascicule}_reocr/   texte ré-océrisé
  config/.env                                        MISTRAL_API_KEY

SORTIES

  verification/controle_unitaire.json    une entrée par article repris
  verification/controle_unitaire.md      la comparaison, lisible

Le fichier de sortie est relu au démarrage : une exécution interrompue reprend
où elle s'était arrêtée, sans repayer les appels déjà faits.

PAQUETS EMPLOYÉS

  json, time, argparse, pathlib, collections   bibliothèque standard
  les fonctions de classify_iptc_mistral_batched, importées

USAGE

    python3 controle_appel_unitaire.py --estimer   # chiffre la dépense, n'appelle rien
    python3 controle_appel_unitaire.py             # exécute
    python3 controle_appel_unitaire.py --comparer  # écrit la comparaison
"""

import argparse
import importlib.util
import json
import sys
import time
from collections import Counter
from pathlib import Path

ICI = Path(__file__).resolve().parent
SORTIE = ICI / "verification"
RESULTAT = SORTIE / "controle_unitaire.json"
TEMOIN = SORTIE / "controle_unitaire_bis.json"
RAPPORT = SORTIE / "controle_unitaire.md"


def _module():
    """Importe le script de classification, dont tout le comportement est repris."""
    spec = importlib.util.spec_from_file_location(
        "classif", ICI / "classify_iptc_mistral_batched.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def articles_juges():
    """Rend les (fascicule, article) distincts dont une étiquette a été jugée."""
    cle = json.loads((SORTIE / "cle.json").read_text(encoding="utf-8"))
    vus, out = set(), []
    for c in cle:
        k = (c["fascicule"], c["article"])
        if k not in vus:
            vus.add(k)
            out.append(k)
    return sorted(out)


def attributions_groupees():
    """Les étiquettes que la campagne groupée a produites, par article."""
    m = _module()
    out = {}
    for f in sorted((m.OUTPUT_DIR).glob("*_themes.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        fasc = str(d.get("fascicule") or f.stem.split("_")[0])
        for a in d["articles"]:
            out[(fasc, a["article_id"])] = {t["code"] for t in (a.get("themes") or [])}
    return out


def main():
    """Point d'entrée : reprend un par un les articles déjà jugés, pour mesurer ce
    que le groupage retire à la description. Avec --temoin, écrit dans un
    second fichier, ce qui donne deux exécutions comparables."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--estimer", action="store_true",
                    help="chiffre la dépense sans passer d'appel")
    ap.add_argument("--comparer", action="store_true",
                    help="écrit la comparaison depuis les résultats déjà obtenus")
    ap.add_argument("--limite", type=int, default=0,
                    help="ne traiter que les n premiers articles")
    ap.add_argument("--temoin", action="store_true",
                    help="refaire les mêmes appels unitaires dans un second fichier, "
                         "pour mesurer la part de non-reproductibilité du modèle")
    args = ap.parse_args()

    global RESULTAT
    if getattr(args, 'temoin', False):
        RESULTAT = TEMOIN
    m = _module()
    cibles = articles_juges()
    if args.limite:
        cibles = cibles[:args.limite]

    if args.comparer:
        return comparer(m)

    # Textes et budget
    leaves = m.build_leaves(m.TAXONOMY_PATH)
    leaves_str = m.leaves_prompt_str(leaves)
    fixe = m.count_tokens(m.SYSTEM_PROMPT) + m.count_tokens(leaves_str) + 100

    textes, manquants = {}, []
    for fasc in sorted({f for f, _ in cibles}):
        try:
            for a in m.extract_articles(fasc):
                textes[(fasc, a["id"])] = a["text"]
        except Exception as e:                       # fascicule illisible
            manquants.append((fasc, str(e)[:60]))

    a_faire = [(f, i) for f, i in cibles if (f, i) in textes]
    absents = [(f, i) for f, i in cibles if (f, i) not in textes]

    entrant = sum(fixe + m.count_tokens(textes[k]) + m.PER_ARTICLE_WRAPPER_TOKENS
                  for k in a_faire)
    sortant = 40 * len(a_faire)                      # ordre de grandeur observé
    pe, ps = m.PRICING.get(m.MISTRAL_MODEL, (0.50, 1.50))
    cout = entrant / 1e6 * pe + sortant / 1e6 * ps

    print(f"  {len(a_faire)} articles à reprendre, un par appel")
    if absents:
        print(f"  {len(absents)} introuvables dans le corpus, écartés")
    if manquants:
        print(f"  {len(manquants)} fascicules illisibles : {manquants[:2]}")
    print(f"  modèle          {m.MISTRAL_MODEL}")
    print(f"  overhead fixe   {fixe:,} jetons par appel, dont "
          f"{m.count_tokens(leaves_str):,} pour la liste")
    print(f"  entrée estimée  {entrant:,} jetons")
    print(f"  dépense estimée ${cout:.2f}")

    if args.estimer:
        return

    if not m.MISTRAL_API_KEY:
        raise SystemExit("MISTRAL_API_KEY absente : renseigner config/.env")

    deja = {}
    if RESULTAT.exists():
        deja = {tuple(k.split("|")): v
                for k, v in json.loads(RESULTAT.read_text(encoding="utf-8")).items()
                if "codes" in v}
        print(f"  {len(deja)} articles déjà obtenus, repris tels quels ; "
              "les entrées en erreur sont retentées")

    reste = [k for k in a_faire if k not in deja]
    t0 = time.time()
    for n, (fasc, aid) in enumerate(reste, 1):
        print(f"  appel {n}/{len(reste)} : {fasc}/{aid}", end="\r", file=sys.stderr)
        try:
            rep, usage, secondes = m.classify_batch(
                [{"id": aid, "text": textes[(fasc, aid)]}], leaves_str, list(leaves))
            deja[(fasc, aid)] = {
                "codes": rep.get(aid, []),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "secondes": round(secondes, 2),
            }
        except Exception as e:
            deja[(fasc, aid)] = {"erreur": str(e)[:200]}
        if n % 10 == 0 or n == len(reste):
            RESULTAT.write_text(json.dumps(
                {"|".join(k): v for k, v in deja.items()}, ensure_ascii=False, indent=1),
                encoding="utf-8")
    print(" " * 70, file=sys.stderr)
    print(f"  terminé en {(time.time()-t0)/60:.1f} minutes")
    comparer(m)


def comparer(m):
    """Confronte les étiquettes unitaires à celles du traitement groupé."""
    if not RESULTAT.exists():
        raise SystemExit("aucun résultat : exécuter le script sans --comparer.")
    uni = {tuple(k.split("|")): v
           for k, v in json.loads(RESULTAT.read_text(encoding="utf-8")).items()}
    grp = attributions_groupees()

    lignes, ident, inter, disj, err = [], 0, 0, 0, 0
    for k, v in sorted(uni.items()):
        if "erreur" in v:
            err += 1
            continue
        a = set(v["codes"])
        b = grp.get(k, set())
        if a == b:
            ident += 1
            etat = "identique"
        elif a & b:
            inter += 1
            etat = "recoupement partiel"
        else:
            disj += 1
            etat = "disjoint"
        lignes.append((k, etat, sorted(a), sorted(b)))

    n = ident + inter + disj
    txt = ["# Contrôle : un article par appel", "",
           f"{n} articles repris, {err} en erreur. Modèle {m.MISTRAL_MODEL}.", "",
           "| Résultat | Effectif | Part |", "|---|---:|---:|",
           f"| étiquettes identiques | {ident} | {100*ident//n if n else 0}~% |",
           f"| recoupement partiel | {inter} | {100*inter//n if n else 0}~% |",
           f"| aucune étiquette commune | {disj} | {100*disj//n if n else 0}~% |", ""]
    if disj:
        txt += ["## Les cas disjoints", ""]
        for k, etat, a, b in lignes:
            if etat == "disjoint":
                txt.append(f"- `{k[0]}/{k[1]}` : unitaire {a} · groupé {sorted(b)}")
    RAPPORT.write_text("\n".join(txt) + "\n", encoding="utf-8")
    print(f"\n  identiques {ident}/{n}, recoupement partiel {inter}, disjoints {disj}")
    print(f"  → {RAPPORT}")


if __name__ == "__main__":
    main()
