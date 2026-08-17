"""
lda_mallet_corpus.py : modélisation thématique sur l'ensemble du sous-corpus

CE QUE FAIT CE SCRIPT

Lit les fichiers ALTO et METS des cent fascicules, reconstitue les unités
demandées, normalise leur texte, puis estime un modèle thématique par MALLET.

Il constitue également le corpus pour les autres scripts du dossier, qui
l'importent. Cette mise en commun garantit que les trois familles de méthodes
portent sur les mêmes documents, condition sans laquelle leur comparaison ne
vaudrait rien.

CE QUI LE DISTINGUE DES PREMIERS ESSAIS

  1. Un seul modèle sur tous les fascicules réunis, au lieu d'un modèle par
     fascicule. Cent modèles donnaient mille thèmes incomparables entre eux.
  2. Graine aléatoire fixée et enregistrée. Les runs deviennent reproductibles,
     et les écarts entre conditions cessent d'être confondus avec le bruit de
     l'échantillonneur.
  3. Le nombre d'itérations est réellement transmis à MALLET. Les premiers
     essais l'affichaient et l'enregistraient sans jamais le passer, et
     tournaient donc sous la valeur par défaut de cent itérations.
  4. Deux granularités : le bloc de texte, que produit la reconnaissance de mise
     en page, et l'article, que reconstitue la carte logique du METS.
  5. Appel direct au binaire MALLET. L'enveloppe little_mallet_wrapper,
     employée d'abord, liait la liste de mots vides au paramètre de mise en
     minuscules, de sorte que le filtrage se faisait avec une liste anglaise.
     Elle est responsable du point 3.

POURQUOI MALLET PLUTÔT QU'UNE AUTRE IMPLÉMENTATION

Trois bibliothèques mettent la méthode à portée. MALLET estime le modèle par
échantillonnage de Gibbs, scikit-learn et gensim par inférence variationnelle.
Les deux procédés ne calculent pas la même chose, et le premier est celui
qu'emploie le projet impresso sur un corpus de presse comparable, ce qui rend
les résultats confrontables.

ENTRÉES

  ../re-ocr/corpus/original/{fascicule}/ocr/*.xml     ALTO du texte hérité
  ../re-ocr/corpus/original/{fascicule}/toc/*.xml     METS, carte logique
  ../re-ocr/corpus/reocr_mistral/{fascicule}_reocr/   texte ré-océrisé
  stopwords_fr_presse.txt, stopwords_extra.txt        formes à retirer
  stoplocs_fr_presse.txt                              locutions à retirer
  Mallet-202108/bin/mallet                            binaire à installer

SORTIES, dans resultats/{nom_du_run}/

  topics.json        thèmes, effectifs et mots de tête
  span_topic.json    affectation de chaque document
  meta.json          paramètres exacts, durée, nombre de documents et de jetons
  mallet_training/   fichiers intermédiaires de MALLET

PAQUETS EMPLOYÉS

  argparse, json, re, subprocess, sys, time, datetime, collections, pathlib
  xml.etree.ElementTree     lecture des fichiers ALTO et METS, bibliothèque
                            standard. Le format est simple et régulier, ce qui
                            rend une bibliothèque tierce inutile.
  spacy                     étiquetage grammatical, importé seulement lorsque
                            l'option --pos est employée

MALLET est un programme Java. Il n'est pas redistribué avec ce dépôt et doit
être installé séparément.

USAGE

    python3 lda_mallet_corpus.py --source mistral --granularite article \
            --k 20 --iterations 2000 --graine 1 --freq-min 5 --longueur-min 3
"""

import argparse
import json
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

# ── Chemins ───────────────────────────────────────────────────────────────────

ICI = Path(__file__).resolve().parent

def _trouver_corpus() -> Path:
    """Remonte l'arborescence jusqu'au dossier contenant le corpus, pour que le
    script reste valide si l'ensemble du dossier de travail est déplacé.

    Deux dispositions sont admises : celle du dossier de travail, où le dépôt de
    classification est un sous-dossier de github/, et celle du dépôt lui-même,
    où ces scripts vivent dans topic-modeling/ à côté de re-ocr/."""
    relatifs = (Path("github") / "classification-iptc" / "re-ocr" / "corpus",
                Path("re-ocr") / "corpus")
    for base in [ICI, *ICI.parents]:
        for relatif in relatifs:
            if (base / relatif).is_dir():
                return base / relatif
    return ICI / relatifs[0]   # l'erreur sera signalée au lancement

CORPUS = _trouver_corpus()

SOURCES = {
    "bnf":     {"racine": CORPUS / "original",       "suffixe": ""},
    "mistral": {"racine": CORPUS / "reocr_mistral",  "suffixe": "_reocr"},
}

STOPWORDS_PATH  = ICI / "stopwords_fr_presse.txt"
STOPWORDS_EXTRA = ICI / "stopwords_extra.txt"
STOPLOCS_PATH   = ICI / "stoplocs_fr_presse.txt"
OUTPUT_DIR      = ICI / "resultats"
MALLET          = ICI / "Mallet-202108" / "bin" / "mallet"

NS_ALTO = "http://www.loc.gov/standards/alto/ns-v3#"
NS_METS = "http://www.loc.gov/METS/"
TITRAILLE = {"title", "subtitle", "heading"}

# ── Ressources linguistiques ──────────────────────────────────────────────────

def charger_stopwords() -> set[str]:
    """Lit les listes de formes à retirer et les rend en un seul ensemble.

    Deux fichiers sont lus. Le premier réunit les mots vides du français et les
    formes propres à la presse dont le mémoire a établi qu'elles ne portent
    aucune information thématique. Le second recueille les ajouts faits au fil
    des runs.

    Les lignes vides et celles qui commencent par un croisillon sont ignorées,
    ce qui permet de commenter les listes.
    """
    mots = set()
    for chemin in (STOPWORDS_PATH, STOPWORDS_EXTRA):
        if chemin.exists():
            for ligne in chemin.read_text(encoding="utf-8").splitlines():
                m = ligne.strip().lower()
                if m and not m.startswith("#"):
                    mots.add(m)
    return mots


def charger_stoplocs() -> list[str]:
    """Lit les locutions à retirer et les rend triées par longueur décroissante.

    Une locution compte plusieurs mots et doit être retirée avant que le texte
    ne soit découpé, sans quoi ses éléments survivraient isolément. Le tri par
    longueur décroissante est nécessaire : il évite qu'une locution courte
    incluse dans une plus longue ne soit retirée d'abord et ne détruise la
    seconde.
    """
    locs = []
    if STOPLOCS_PATH.exists():
        for ligne in STOPLOCS_PATH.read_text(encoding="utf-8").splitlines():
            l = ligne.strip().lower()
            if l and not l.startswith("#"):
                locs.append(l)
    return sorted(locs, key=len, reverse=True)


RE_NOMBRE     = re.compile(r"\d[\d.,]*")
RE_NON_LETTRE = re.compile(r"[^a-zàâäçéèêëîïôöùûüÿñæœ]+")

# Formes élidées du français : l', d', qu', j', n', s', c', m', t'. L'apostrophe
# sert de séparateur, si bien que ces formes deviennent des jetons d'une ou deux
# lettres. Celles de deux lettres ne tombent pas sous le filtre de longueur.
ELISIONS = {"qu", "jusqu", "lorsqu", "puisqu", "quoiqu", "aujourd", "hui"}


def normaliser(texte: str, stopwords: set[str], stoplocs: list[str],
               longueur_min: int = 2) -> str:
    """Minuscules, retrait des locutions, nombres ramenés à num, apostrophe
    traitée en séparateur, mots vides et lettres isolées retirés."""
    t = texte.lower()
    for loc in stoplocs:
        t = t.replace(loc, " ")
    t = RE_NOMBRE.sub(" num ", t)
    t = RE_NON_LETTRE.sub(" ", t)          # l'apostrophe devient une espace
    return " ".join(j for j in t.split()
                    if len(j) >= longueur_min and j not in stopwords
                    and j not in ELISIONS) or ""

# ── Lecture des fichiers ──────────────────────────────────────────────────────

def blocs_du_fascicule(dossier_ocr: Path) -> dict[str, dict]:
    """Tous les TextBlock d'un fascicule, indexés par identifiant."""
    blocs = {}
    for alto in sorted(dossier_ocr.glob("X*.xml")):
        try:
            racine = ET.parse(alto).getroot()
        except ET.ParseError as e:
            print(f"    ALTO illisible, ignoré : {alto.name} ({e})", file=sys.stderr)
            continue
        etiquettes = {st.get("ID", ""): (st.get("LABEL") or "").lower()
                      for st in racine.iter(f"{{{NS_ALTO}}}StructureTag")}
        for tb in racine.iter(f"{{{NS_ALTO}}}TextBlock"):
            bid = tb.get("ID", "")
            if not bid:
                continue
            label = ""
            for ref in (tb.get("TAGREFS") or "").split():
                if ref in etiquettes:
                    label = etiquettes[ref]
                    break
            lignes = []
            for tl in tb.iter(f"{{{NS_ALTO}}}TextLine"):
                mots = []
                for s in tl.iter(f"{{{NS_ALTO}}}String"):
                    # Mots coupés en fin de ligne : ALTO porte la forme entière
                    # dans SUBS_CONTENT, répétée sur les deux moitiés. Sans ce
                    # recollement, les fichiers de la bibliothèque livrent des
                    # fragments (« ment », « tion ») que la ré-océrisation
                    # Mistral ne produit pas, ce qui fausserait la comparaison.
                    subs = (s.get("SUBS_TYPE") or "")
                    if subs == "HypPart2":
                        continue
                    contenu = s.get("SUBS_CONTENT") if subs == "HypPart1" \
                              else s.get("CONTENT", "")
                    if contenu:
                        mots.append(contenu)
                if mots:
                    lignes.append(" ".join(mots).strip())
            blocs[bid] = {"texte": " ".join(lignes).strip(),
                          "page": alto.stem,
                          "titraille": label in TITRAILLE}
    return blocs


def articles_du_fascicule(chemin_toc: Path) -> list[dict]:
    """Divisions ARTICLE du structMap logique, avec la liste ordonnée
    des blocs qu'elles rassemblent."""
    if not chemin_toc.exists():
        return []
    try:
        racine = ET.parse(chemin_toc).getroot()
    except ET.ParseError as e:
        print(f"    METS illisible, ignoré : {chemin_toc.name} ({e})", file=sys.stderr)
        return []
    logique = [sm for sm in racine.iter(f"{{{NS_METS}}}structMap")
               if (sm.get("TYPE") or "").lower() == "logical"]
    if not logique:
        return []
    articles = []
    for div in logique[0].iter(f"{{{NS_METS}}}div"):
        if (div.get("TYPE") or "").upper() != "ARTICLE":
            continue
        ids = [a.get("BEGIN") for a in div.iter(f"{{{NS_METS}}}area")
               if a.get("BEGIN")]
        if ids:
            articles.append({"div_id": div.get("ID", ""), "blocs": ids})
    return articles

# ── Filtrage morphosyntaxique ─────────────────────────────────────────────────

def lemmes_par_categorie(textes: list[str], categories: set[str],
                         longueur_min: int) -> list[str]:
    """Ne conserve que les lemmes des catégories demandées, à la manière
    d'impresso, qui retient les noms communs et les noms propres. Écarte donc
    verbes, adjectifs, adverbes et mots grammaticaux sans recourir à une liste
    de mots vides."""
    try:
        import spacy
    except ImportError:
        sys.exit("spaCy est requis pour --pos. Installer spacy et "
                 "fr_core_news_sm dans l'environnement d'exécution.")
    try:
        nlp = spacy.load("fr_core_news_sm", disable=["parser", "ner"])
    except OSError:
        sys.exit("Modèle fr_core_news_sm introuvable. "
                 "Lancer : python -m spacy download fr_core_news_sm")
    nlp.max_length = 3_000_000
    sortie = []
    for n, doc in enumerate(nlp.pipe(textes, batch_size=200), 1):
        sortie.append(" ".join(
            t.lemma_.lower() for t in doc
            if t.pos_ in categories and len(t.lemma_) >= longueur_min
            and t.lemma_.isalpha()))
        if n % 5000 == 0:
            print(f"    étiquetage {n}/{len(textes)}")
    return sortie

# ── Constitution du corpus ────────────────────────────────────────────────────

def constituer(source: str, granularite: str, fascicules: list[str],
               min_mots: int, stopwords, stoplocs,
               longueur_min: int = 2, freq_min: int = 0,
               pos: set[str] | None = None) -> list[dict]:
    """Constitue le corpus et rend une liste de documents.

    C'est la fonction que tous les autres scripts du dossier importent. Chaque
    document rendu porte cinq clés.

      doc_id      identifiant complet, fascicule et unité séparés par deux points
      fascicule   numéro du fascicule
      unite       identifiant du bloc ou de la division d'article
      page        fichier de page où l'unité commence
      texte       texte brut, employé par les modèles de plongement
      traite      texte normalisé, employé par la modélisation thématique

    Le traitement diffère selon la granularité. En mode bloc, chaque TextBlock
    des fichiers ALTO devient un document, les blocs de titraille étant écartés
    puisqu'ils ne portent pas de texte courant. En mode article, la carte
    logique du METS donne pour chaque division ARTICLE la liste des blocs qui la
    composent, dont les textes sont concaténés dans l'ordre déclaré.

    Trois filtres s'appliquent ensuite au texte normalisé.

      longueur_min   longueur minimale d'une forme, en caractères
      freq_min       nombre minimal de documents où une forme doit paraître
      min_mots       nombre minimal de mots qu'un document doit conserver

    Le paramètre pos active un étiquetage grammatical qui ne retient que les
    catégories demandées. Il est traité à part, l'étiquetage devant se faire sur
    le texte entier avant tout filtrage.
    """
    conf = SOURCES[source]
    docs = []
    ignores_vides, ignores_courts = 0, 0

    for n, fid in enumerate(fascicules, 1):
        dossier = conf["racine"] / f"{fid}{conf['suffixe']}"
        ocr = dossier / "ocr"
        if not ocr.is_dir():
            print(f"    fascicule sans dossier ocr, ignoré : {fid}", file=sys.stderr)
            continue
        blocs = blocs_du_fascicule(ocr)

        if granularite == "bloc":
            brut = [{"doc_id": f"{fid}:{bid}", "fascicule": fid, "unite": bid,
                     "page": b["page"], "texte": b["texte"]}
                    for bid, b in blocs.items()
                    if b["texte"] and not b["titraille"]]
        else:
            toc = next(iter(sorted((dossier / "toc").glob("*.xml"))), None) \
                  if (dossier / "toc").is_dir() else None
            brut = []
            for art in articles_du_fascicule(toc) if toc else []:
                morceaux = [blocs[b]["texte"] for b in art["blocs"]
                            if b in blocs and blocs[b]["texte"]]
                if morceaux:
                    pages = {blocs[b]["page"] for b in art["blocs"] if b in blocs}
                    brut.append({"doc_id": f"{fid}:{art['div_id']}",
                                 "fascicule": fid, "unite": art["div_id"],
                                 "page": sorted(pages)[0] if pages else "",
                                 "texte": " ".join(morceaux)})

        for d in brut:
            if pos:
                d["traite"] = None          # rempli après étiquetage groupé
                docs.append(d)
                continue
            traite = normaliser(d["texte"], stopwords, stoplocs, longueur_min)
            if not traite:
                ignores_vides += 1
                continue
            if len(traite.split()) < min_mots:
                ignores_courts += 1
                continue
            d["traite"] = traite
            docs.append(d)

        if n % 20 == 0:
            print(f"    {n}/{len(fascicules)} fascicules, {len(docs)} documents")

    if pos:
        print(f"  étiquetage morphosyntaxique de {len(docs)} documents "
              f"({','.join(sorted(pos))})…")
        lemmes = lemmes_par_categorie([d["texte"] for d in docs], pos, longueur_min)
        gardes = []
        for d, l in zip(docs, lemmes):
            if not l:
                ignores_vides += 1
            elif len(l.split()) < min_mots:
                ignores_courts += 1
            else:
                d["traite"] = l
                gardes.append(d)
        docs = gardes

    print(f"  {len(docs)} documents retenus "
          f"({ignores_vides} vides, {ignores_courts} sous le seuil de {min_mots} mots)")

    if freq_min > 1:
        # Seuil de fréquence minimale, à la manière d'impresso : les formes trop
        # rares sont écartées du vocabulaire. Les débris d'océrisation étant par
        # nature peu répétés, le seuil les vise sans les nommer.
        from collections import Counter
        freq = Counter(j for d in docs for j in d["traite"].split())
        garde = {j for j, n in freq.items() if n >= freq_min}
        retires = len(freq) - len(garde)
        vides = 0
        filtres = []
        for d in docs:
            t = " ".join(j for j in d["traite"].split() if j in garde)
            if len(t.split()) >= min_mots:
                d["traite"] = t
                filtres.append(d)
            else:
                vides += 1
        print(f"  seuil de fréquence {freq_min}~: {retires} formes retirées sur "
              f"{len(freq)}, {vides} documents vidés, {len(filtres)} conservés")
        docs = filtres
    return docs

# ── MALLET ────────────────────────────────────────────────────────────────────

def lancer_mallet(docs: list[dict], k: int, iterations: int, graine: int,
                  dossier: Path, optimisation: int = 0, burn_in: int = 20) -> tuple[list[list[str]], list[list[float]]]:
    """Écrit les fichiers d'entrée, appelle MALLET et lit ses sorties.

    MALLET travaille en deux temps. La commande import-file convertit le texte
    en un format binaire interne. La commande train-topics estime le modèle.

    Les arguments transmis à train-topics sont les suivants.

      --num-topics          nombre de thèmes, choisi par l'analyste. Le
                            paramètre commande le grain de la description et ne
                            se déduit pas des données.
      --num-iterations      passes de l'échantillonneur sur le corpus. Deux
                            mille est la valeur que recommande la configuration
                            de production du projet impresso.
      --random-seed         graine du générateur aléatoire. Sans elle, deux runs
                            identiques donnent des résultats différents, et un
                            écart entre conditions se confond avec le bruit.
      --optimize-interval   intervalle entre deux réestimations de la loi a
                            priori sur les mélanges. À zéro, la loi reste
                            symétrique et impose des thèmes de fréquence égale,
                            hypothèse qu'une collection de presse contredit.
      --optimize-burn-in    itérations à attendre avant que l'optimisation ne
                            commence, le temps que l'échantillonneur se
                            stabilise. Optimiser trop tôt fige un état transitoire.

    Le filtrage des mots vides est fait en amont, dans normaliser, plutôt que
    confié à MALLET par son option de liste d'arrêt. Cela garantit que le corpus
    d'entraînement et le corpus de référence des métriques portent exactement le
    même vocabulaire.

    Rend les mots de tête de chaque thème et la distribution thématique de
    chaque document.
    """
    entrainement = dossier / "mallet_training"
    entrainement.mkdir(parents=True, exist_ok=True)

    f_txt     = entrainement / "training_data.txt"
    f_mallet  = entrainement / "training_data.mallet"
    f_keys    = entrainement / "mallet.topic_keys.txt"
    f_distrib = entrainement / "mallet.topic_distributions.txt"
    f_poids   = entrainement / "mallet.word_weights.txt"
    f_diag    = entrainement / "mallet.diagnostics.xml"

    # Format natif de MALLET : identifiant, étiquette, texte, séparés par des
    # tabulations, ce qui préserve l'identifiant du document en sortie.
    f_txt.write_text(
        "\n".join(f"{d['doc_id']}\tno_label\t{d['traite']}" for d in docs),
        encoding="utf-8")

    subprocess.run([str(MALLET), "import-file",
                    "--input", str(f_txt),
                    "--output", str(f_mallet),
                    "--keep-sequence",
                    "--token-regex", r"[\p{L}\p{N}]+"],
                   check=True, capture_output=True)

    cmd = [str(MALLET), "train-topics",
           "--input", str(f_mallet),
           "--num-topics", str(k),
           "--num-iterations", str(iterations),
           "--random-seed", str(graine),
           "--output-topic-keys", str(f_keys),
           "--output-doc-topics", str(f_distrib),
           "--topic-word-weights-file", str(f_poids),
           "--diagnostics-file", str(f_diag)]
    if optimisation > 0:
        # Optimisation des hyperparamètres : autorise une loi a priori
        # asymétrique sur les mélanges de thèmes, que Wallach, Mimno et
        # McCallum tiennent pour préférable à la loi symétrique par défaut.
        cmd += ["--optimize-interval", str(optimisation),
                "--optimize-burn-in", str(burn_in)]
    subprocess.run(cmd, check=True, capture_output=True)

    topics = []
    for ligne in f_keys.read_text(encoding="utf-8").splitlines():
        parts = ligne.rstrip("\n").split("\t")
        if len(parts) >= 3:
            topics.append(parts[2].split())

    distrib = []
    for ligne in f_distrib.read_text(encoding="utf-8").splitlines():
        if ligne.startswith("#"):
            continue
        parts = ligne.rstrip("\n").split("\t")
        if len(parts) < 3:
            continue
        try:
            distrib.append([float(x) for x in parts[2:] if x])
        except ValueError:
            continue
    return topics, distrib

# ── Programme principal ───────────────────────────────────────────────────────

def main() -> None:
    """Constitue le corpus, lance les runs demandés et enregistre les sorties.

    Plusieurs valeurs de K peuvent être demandées en une fois, et l'option
    --repetitions rejoue chaque configuration avec des graines successives. Ces
    répétitions sont nécessaires : l'échantillonnage de Gibbs est stochastique,
    et quatre runs conduits dans une configuration identique sur un seul
    fascicule donnent des valeurs de NPMI qui s'étendent sur 0,164. Un écart
    mesuré sans répétitions ne peut donc pas être distingué du bruit.

    Le seuil min_mots vaut 20 pour l'article et 5 pour le bloc lorsqu'il n'est
    pas donné. Les deux unités n'ont pas le même ordre de grandeur, l'article
    comptant 114 mots en médiane et le bloc 17.
    """
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=sorted(SOURCES), required=True)
    p.add_argument("--granularite", choices=["bloc", "article"], required=True)
    p.add_argument("--k", type=int, nargs="+", default=[6, 10],
                   help="une ou plusieurs valeurs de K")
    p.add_argument("--iterations", type=int, default=1000)
    p.add_argument("--graine", type=int, default=1)
    p.add_argument("--optimisation", type=int, default=0,
                   help="intervalle d'optimisation des hyperparamètres (0 = désactivée)")
    p.add_argument("--burn-in", type=int, default=20,
                   help="itérations avant la première optimisation")
    p.add_argument("--repetitions", type=int, default=1,
                   help="runs successifs, graines graine, graine+1, …")
    p.add_argument("--longueur-min", type=int, default=2,
                   help="longueur minimale d'une forme, en caractères")
    p.add_argument("--freq-min", type=int, default=0,
                   help="fréquence minimale d'une forme dans le corpus")
    p.add_argument("--pos", default=None,
                   help="catégories à conserver, séparées par des virgules, par exemple NOUN,PROPN ; suppose spaCy et fr_core_news_sm")
    p.add_argument("--min-mots", type=int, default=None,
                   help="seuil par défaut : 5 au bloc, 20 à l'article")
    p.add_argument("--fascicules", nargs="+", default=None,
                   help="par défaut, tous ceux du dossier source")
    args = p.parse_args()

    if not MALLET.exists():
        sys.exit(f"Binaire MALLET introuvable : {MALLET}")
    racine = SOURCES[args.source]["racine"]
    if not racine.is_dir():
        sys.exit(f"Dossier source introuvable : {racine}")

    suffixe = SOURCES[args.source]["suffixe"]
    fascicules = args.fascicules or sorted(
        d.name[:-len(suffixe)] if suffixe and d.name.endswith(suffixe) else d.name
        for d in racine.iterdir() if d.is_dir())
    min_mots = args.min_mots if args.min_mots is not None else (5 if args.granularite == "bloc" else 20)

    print(f"\n{'='*66}")
    print(f"LDA MALLET : {args.source} | {args.granularite} | "
          f"{len(fascicules)} fascicules | K={args.k} | "
          f"{args.iterations} itérations | {args.repetitions} répétition(s) | "
          f"optimisation {args.optimisation or 'désactivée'}")
    print(f"{'='*66}")

    print("\n1. Constitution du corpus…")
    stopwords, stoplocs = charger_stopwords(), charger_stoplocs()
    print(f"  {len(stopwords)} mots vides, {len(stoplocs)} locutions")
    docs = constituer(args.source, args.granularite, fascicules,
                      min_mots, stopwords, stoplocs,
                      args.longueur_min, args.freq_min,
                      set(args.pos.upper().split(',')) if args.pos else None)
    if not docs:
        sys.exit("Aucun document retenu.")
    total_mots = sum(len(d["traite"].split()) for d in docs)
    print(f"  {total_mots} jetons après filtrage, "
          f"{total_mots/len(docs):.1f} en moyenne par document")

    for k in args.k:
        for r in range(args.repetitions):
            graine = args.graine + r
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            opt = f"_opt{args.optimisation}b{args.burn_in}" if args.optimisation else ""
            filt = (f"_pos" if args.pos else "") + \
                   (f"_f{args.freq_min}" if args.freq_min > 1 else "") + \
                   (f"_l{args.longueur_min}" if args.longueur_min != 2 else "")
            run_id = (f"lda_corpus_{args.source}_{args.granularite}"
                      f"_k{k}_g{graine}{opt}{filt}_{ts}")
            dossier = OUTPUT_DIR / run_id
            dossier.mkdir(parents=True, exist_ok=True)

            print(f"\n2. Entraînement K={k}, graine={graine}…")
            t0 = time.time()
            try:
                topics, distrib = lancer_mallet(docs, k, args.iterations,
                                                graine, dossier,
                                                args.optimisation, args.burn_in)
            except subprocess.CalledProcessError as e:
                print(f"  MALLET a échoué : {e.stderr.decode('utf-8', 'replace')[:500]}",
                      file=sys.stderr)
                continue
            duree = time.time() - t0
            print(f"  terminé en {duree:.1f}s")

            if len(distrib) != len(docs):
                print(f"  Attention : {len(distrib)} distributions pour "
                      f"{len(docs)} documents", file=sys.stderr)

            dominants = [max(range(len(d)), key=d.__getitem__) for d in distrib]
            tailles = [dominants.count(i) for i in range(k)]

            (dossier / "topics.json").write_text(json.dumps(
                [{"topic_id": i,
                  "top_words": " ".join(topics[i][:10]) if i < len(topics) else "",
                  "size": tailles[i]} for i in range(k)],
                ensure_ascii=False, indent=2), encoding="utf-8")

            (dossier / "span_topic.json").write_text(json.dumps(
                [{"doc_id": d["doc_id"], "fascicule": d["fascicule"],
                  "unite": d["unite"], "page": d["page"],
                  "topic_id": dominants[i],
                  "weight": round(dist[dominants[i]], 4),
                  "dist": [round(x, 4) for x in dist]}
                 for i, (d, dist) in enumerate(zip(docs, distrib))],
                ensure_ascii=False, indent=2), encoding="utf-8")

            (dossier / "meta.json").write_text(json.dumps(
                {"model": "lda_mallet", "run_id": run_id,
                 "source": args.source, "granularite": args.granularite,
                 "k": k, "iterations": args.iterations, "graine": graine,
                 "optimisation": args.optimisation,
                 "burn_in": args.burn_in if args.optimisation else None,
                 "n_docs": len(docs), "n_fascicules": len(fascicules),
                 "min_mots": min_mots, "n_tokens": total_mots,
                 "longueur_min": args.longueur_min, "freq_min": args.freq_min,
                 "pos": args.pos,
                 "duree_s": round(duree, 1)},
                ensure_ascii=False, indent=2), encoding="utf-8")

            print(f"  → {dossier.name}")
            for i in range(k):
                mots = " ".join(topics[i][:10]) if i < len(topics) else ""
                print(f"    Thème {i:2d} ({tailles[i]:5d} doc.) : {mots}")

    print(f"\nTerminé. Résultats dans {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
