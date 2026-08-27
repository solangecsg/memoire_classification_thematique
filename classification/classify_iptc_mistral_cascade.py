"""
classify_iptc_mistral_cascade.py : classification IPTC en deux étages

CE QUE FAIT CE SCRIPT

Même tâche que classify_iptc_mistral.py, dont il faut lire la docstring pour le
détail du traitement. La différence tient au découpage de la liste d'étiquettes
en deux passages successifs.

LE RAISONNEMENT QUI Y CONDUIT

La liste des 567 étiquettes pèse 92 pour cent du volume facturé. Il paraît donc
logique de la découper, chaque appel ne recevant qu'une fraction du référentiel.

LE PROCÉDÉ

  ÉTAGE 1. Tous les articles reçoivent une étiquette parmi les 120 concepts de
  deuxième niveau, liste 4,7 fois plus courte que la liste complète. Le
  traitement se fait par lots, comme dans la variante groupée.

  REGROUPEMENT. Les articles sont ensuite rassemblés par branche assignée, en
  les mélangeant entre fascicules. Le mélange est nécessaire : une branche rare
  ne réunirait que quelques articles dans un seul fascicule, et ses lots
  seraient trop petits pour amortir le coût fixe.

  ÉTAGE 2. Pour les 68 branches qui ont des enfants de troisième niveau, les
  articles sont classés une seconde fois contre les seuls enfants de leur
  branche, soit de 1 à 111 étiquettes selon la branche, avec une médiane de 4.
  Les 52 branches terminales n'ont pas d'étage 2, leur code de deuxième niveau
  étant directement le thème final.

POURQUOI CETTE VARIANTE EST ÉCARTÉE

Le raisonnement est juste et la conclusion fausse. La cascade revient à 4,60
dollars contre 3,63 pour le groupage simple, et l'écart ne se résorbe pas avec
le volume.

La cause tient à ce que le raisonnement n'avait pas compté. Réduire la liste ne
réduit pas le texte des articles, et la cascade envoie ce texte deux fois. Les
jetons de texte facturés passent de 3,50 à 6,87 millions, soit un facteur de
1,96, quand l'économie faite sur la liste ne compense pas ce doublement.

Une seconde objection s'ajoute à celle du coût. Une erreur du premier étage ne
peut plus être corrigée au second, les vraies étiquettes n'étant même pas
proposées. La cascade ajoute donc un risque de propagation d'erreur que le
passage unique ne connaît pas.

Le script est conservé parce que sa mesure vaut par elle-même : une optimisation
peut être correctement raisonnée sur le facteur dominant et manquer un facteur
secondaire que le raisonnement ne mentionnait pas. Seule la mesure l'a montré.

ENTRÉES ET SORTIES

Identiques à celles de classify_iptc_mistral.py, la sortie étant écrite dans
results/feuilles_mistral_cascade/. Un fichier _stage1_assignments.json conserve
en outre les branches assignées au premier étage.

PAQUETS EMPLOYÉS

Les mêmes que la variante groupée, dont mistral-common pour le comptage exact
des jetons.

USAGE

    python classify_iptc_mistral_cascade.py --fascicule 4109000 --dry-run
    python classify_iptc_mistral_cascade.py --fascicule 4109000
    python classify_iptc_mistral_cascade.py
"""

import argparse
import json
import os
import statistics
import threading
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

#  Chargement .env 


def load_env(env_path: Path):
    """Charge les variables d'un fichier .env dans os.environ (sans dépendance
    externe) : ne remplace jamais une variable déjà définie dans l'environnement.
    """
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


# Arborescence de ce dépôt (voir README.md) :
#   classification-iptc/
#   ├── classification/                            <- ce fichier (+ les 3 autres variantes)
#   ├── analyse_couts/                              <- estimation de coûts (importe ce module)
#   ├── config/.env                                <- MISTRAL_API_KEY=... (à créer soi-même)
#   ├── results/feuilles_mistral_cascade/           <- sorties de ce script
#   └── re-ocr/corpus/
#       ├── original/{fascicule}/{toc,ocr}                  <- ALTO/METS d'origine
#       └── reocr_mistral/{fascicule}_reocr/{toc,ocr}        <- ALTO corrigé par Mistral
SCRIPTS_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPTS_DIR.parent

ENV_FILE = PROJECT_DIR / "config" / ".env"
load_env(ENV_FILE)

MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "")
MISTRAL_MODEL = os.environ.get("MISTRAL_MODEL", "mistral-large-latest")
MISTRAL_URL = "https://api.mistral.ai/v1/chat/completions"

SAMPLE_DIR = PROJECT_DIR / "re-ocr" / "corpus" / "original"
MISTRAL_RESULTS_DIR = PROJECT_DIR / "re-ocr" / "corpus" / "reocr_mistral"
TAXONOMY_PATH = SCRIPTS_DIR / "iptc_mediatopic_official.json"
OUTPUT_DIR = PROJECT_DIR / "results" / "feuilles_mistral_cascade"

MIN_WORDS = 10          # articles plus courts que ça sont ignorés (titres/rubriques vides)
MAX_THEMES = 5
MIN_THEMES = 1
MAX_WORKERS = 5         # lots Mistral en parallèle

# Groupage d'articles par appel (même logique que classify_iptc_mistral_batched)
MAX_BATCH_TOKENS = 40_000
MAX_BATCH_SIZE = 25
PER_ARTICLE_WRAPPER_TOKENS = 15

# $ / 1M tokens (input, output) : mistral.ai/pricing/api/
PRICING = {
    "mistral-large-latest": (0.50, 1.50),
    "mistral-medium-latest": (1.50, 7.50),
    "mistral-small-latest": (0.15, 0.60),
    "ministral-8b-latest": (0.15, 0.15),
    "ministral-3b-latest": (0.10, 0.10),
}

# Taxonomie IPTC officielle (SKOS) 

FR_LABEL_OVERRIDES = {
    "20001360": "Association fraternelle et communautaire",
    "20001361": "Cyberguerre",
    "20001365": "Reporting et performance des entreprises",
    "20001366": "Restructuration d'entreprise",
    "20001370": "Services financiers",
    "20001371": "Services aux entreprises",
    "20001373": "Diversité, équité et inclusion",
    "20001374": "Durabilité",
    "20001380": "Élection partielle",
    "20001381": "Élection révocatoire",
    "20001382": "Formation de coalition",
    "20001383": "Politique de zonage",
    "20001384": "Parti politique",
    "20001385": "Mouvement et association politique",
    "20001391": "Végétarisme et véganisme",
}


def _label(concept, lang="fr", fallback="en-GB"):
    """Libellé français d'un concept IPTC : reprend la traduction manuelle si le
    concept n'en a pas côté IPTC (FR_LABEL_OVERRIDES), sinon la traduction
    officielle française, sinon l'anglais, sinon la première langue disponible.
    """
    code = _code(concept)
    if code in FR_LABEL_OVERRIDES:
        return FR_LABEL_OVERRIDES[code]
    labels = concept.get("prefLabel", {})
    return labels.get(lang) or labels.get(fallback) or next(iter(labels.values()), "")


def _code(concept):
    """Extrait le code à 8 chiffres d'un concept IPTC depuis son qcode
    (ex. "medtop:20000002" → "20000002").
    """
    return concept["qcode"].split(":", 1)[1]


def build_leaves(taxonomy_path, max_level=3):
    """code → {label_fr, l1_code, l1_label, l2_code, l2_label, l3_code, l3_label}
    (voir classify_iptc_mistral_batched.py pour la docstring complète)."""
    with open(taxonomy_path, encoding="utf-8") as f:
        data = json.load(f)

    concepts = {c["uri"]: c for c in data["conceptSet"] if not c.get("retired")}

    children = {}
    roots = []
    for uri, c in concepts.items():
        broader = c.get("broader")
        if broader and broader[0] in concepts:
            children.setdefault(broader[0], []).append(uri)
        else:
            roots.append(uri)

    depth = {}
    queue = list(roots)
    for r in roots:
        depth[r] = 1
    while queue:
        u = queue.pop(0)
        for ch in children.get(u, []):
            if ch not in depth:
                depth[ch] = depth[u] + 1
                queue.append(ch)

    def ancestor_at(uri, level):
        """Remonte jusqu'à l'ancêtre du concept `uri` situé au niveau `level`
        (en suivant `broader` vers la racine).
        """
        cur = uri
        while depth.get(cur, 0) > level:
            cur = concepts[cur]["broader"][0]
        return cur

    leaves = {}
    for uri, c in concepts.items():
        d = depth.get(uri)
        if d is None:
            continue
        is_candidate = d == max_level or (d < max_level and not children.get(uri))
        if not is_candidate:
            continue

        code = _code(c)
        l1_uri = ancestor_at(uri, 1)
        l2_uri = ancestor_at(uri, 2) if d >= 2 else None
        l3_uri = uri if d >= 3 else None

        leaves[code] = {
            "label_fr": _label(c),
            "l1_code": _code(concepts[l1_uri]), "l1_label": _label(concepts[l1_uri]),
            "l2_code": _code(concepts[l2_uri]) if l2_uri else "",
            "l2_label": _label(concepts[l2_uri]) if l2_uri else "",
            "l3_code": _code(concepts[l3_uri]) if l3_uri else "",
            "l3_label": _label(concepts[l3_uri]) if l3_uri else "",
        }
    return leaves


def build_stage_indices(taxonomy_path):
    """Construit les 3 structures nécessaires à la cascade :
      - l2_candidates : {code2 → {label_fr, l1_label}} : les 120 candidats de l'étage 1
      - l3_by_branch : {code2 → {code3 → {label_fr, l1_label, l2_label}}} : pour les
        68 branches qui ont des enfants niveau 3 (candidats de l'étage 2)
      - l2_leaf_codes : {code2, ...} : les 52 branches dépourvues d'enfant de niveau 3
        (le code niveau 2 est alors directement le thème final)
    """
    l2_candidates = build_leaves(taxonomy_path, max_level=2)
    leaves_l3_full = build_leaves(taxonomy_path, max_level=3)

    l3_by_branch = defaultdict(dict)
    l2_leaf_codes = set()
    for code, leaf in leaves_l3_full.items():
        if leaf["l3_code"]:
            l3_by_branch[leaf["l2_code"]][code] = leaf
        else:
            l2_leaf_codes.add(code)

    return l2_candidates, dict(l3_by_branch), l2_leaf_codes


def leaves_prompt_str(leaves):
    """Une ligne par étiquette : code : libellé (+ contexte parent seulement
    si le libellé est ambigu dans cette liste précise)."""
    label_counts = {}
    for leaf in leaves.values():
        label_counts[leaf["label_fr"]] = label_counts.get(leaf["label_fr"], 0) + 1

    lines = []
    for code, leaf in leaves.items():
        if label_counts[leaf["label_fr"]] > 1:
            parent = leaf["l2_label"] if leaf["l3_label"] else ""
            ctx = f" ({parent} > {leaf['l1_label']})" if parent else f" ({leaf['l1_label']})"
        else:
            ctx = ""
        lines.append(f"  {code} : {leaf['label_fr']}{ctx}")
    return "\n".join(lines)


def branch_prompt_str(branch_leaves):
    """Comme leaves_prompt_str, mais sans contexte parent (redondant : tous
    les candidats d'une branche partagent déjà le même niveau 2, rappelé une
    seule fois dans le prompt système de l'étage 2)."""
    return "\n".join(f"  {code} : {leaf['label_fr']}" for code, leaf in branch_leaves.items())


#  Comptage de tokens (réel via mistral-common, repli en caractères sinon) 

_tokenizer = None
_tokenizer_load_failed = False
_TOKENIZER_HF_REPO = "mistralai/Ministral-8B-Instruct-2410"
_FALLBACK_CHARS_PER_TOKEN = 3.5


def _get_tokenizer():
    """Charge le tokenizer Mistral une seule fois (télécharge depuis Hugging Face
    au premier appel, réutilise ensuite le même objet). Retourne None si le
    téléchargement échoue (pas de réseau, paquet manquant) : count_tokens()
    bascule alors sur une estimation par caractères.
    """
    global _tokenizer, _tokenizer_load_failed
    if _tokenizer is not None or _tokenizer_load_failed:
        return _tokenizer
    try:
        from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
        _tokenizer = MistralTokenizer.from_hf_hub(_TOKENIZER_HF_REPO)
    except Exception as e:
        print(f"  ⚠ tokenizer Mistral indisponible ({e}) : repli sur l'estimation par caractères")
        _tokenizer_load_failed = True
    return _tokenizer


def count_tokens(text):
    """Nombre de tokens réels d'un texte, mesuré avec le tokenizer Mistral si
    disponible, sinon estimé par caractères (repli volontairement pessimiste :
    sous-estimer risquerait de dépasser le budget d'un lot).
    """
    tok = _get_tokenizer()
    if tok is not None:
        return len(tok.instruct_tokenizer.tokenizer.encode(text, bos=False, eos=False))
    return int(len(text) / _FALLBACK_CHARS_PER_TOKEN) + 1


def make_batches(articles, fixed_overhead_tokens):
    """Empile des articles entiers dans des lots tant que le budget de jetons
    et la taille max ne sont pas dépassés (jamais de troncature)."""
    batches = []
    current = []
    current_tokens = fixed_overhead_tokens
    for art in articles:
        art_tokens = count_tokens(art["text"]) + PER_ARTICLE_WRAPPER_TOKENS
        if current and (
            current_tokens + art_tokens > MAX_BATCH_TOKENS or len(current) >= MAX_BATCH_SIZE
        ):
            batches.append(current)
            current = []
            current_tokens = fixed_overhead_tokens
        current.append(art)
        current_tokens += art_tokens
    if current:
        batches.append(current)
    return batches


#  Extraction des articles (TOC/METS + ALTO corrigé) 

METS_NS = "http://www.loc.gov/METS/"
XLINK = "http://www.w3.org/1999/xlink"
MODS_NS = "http://www.loc.gov/mods/v3"


def _mets(local):
    """Construit un nom de balise qualifié dans le namespace METS
    (ex. "file" → "{http://www.loc.gov/METS/}file").
    """
    return f"{{{METS_NS}}}{local}"


def parse_toc(toc_folder):
    """Lit le fichier TOC/METS (T*.xml) d'un fascicule et retourne la liste des
    articles logiques qu'il décrit : pour chaque article, son ID, son titre
    (LABEL du div ou titre MODS du dmdSec associé), et la liste ordonnée des
    blocs ALTO qui le composent (nom du fichier ALTO + ID du TextBlock).
    Retourne None si aucun fichier T*.xml n'est trouvé.
    """
    toc_files = list(toc_folder.glob("T*.xml"))
    if not toc_files:
        return None
    root = ET.parse(toc_files[0]).getroot()

    file_map = {}
    for f in root.iter(_mets("file")):
        fid = f.get("ID", "")
        floc = f.find(_mets("FLocat"))
        if floc is not None:
            href = floc.get(f"{{{XLINK}}}href", "")
            name = Path(href).name
            if name:
                file_map[fid] = name

    dmd_titles = {}
    for dmd in root.iter(_mets("dmdSec")):
        did = dmd.get("ID", "")
        ti = dmd.find(f".//{{{MODS_NS}}}title")
        if ti is not None and ti.text:
            dmd_titles[did] = ti.text.strip()

    articles = []
    for sm in root.iter(_mets("structMap")):
        if (sm.get("TYPE") or "").lower() != "logical":
            continue
        for art_div in sm.iter(_mets("div")):
            if (art_div.get("TYPE") or "").upper() != "ARTICLE":
                continue
            art_id = art_div.get("ID", "")
            title = art_div.get("LABEL") or dmd_titles.get(art_div.get("DMDID", ""), art_id)
            blocks = []
            for area in art_div.iter(_mets("area")):
                fid = area.get("FILEID", "")
                bid = area.get("BEGIN", "")
                if fid in file_map and bid:
                    blocks.append((file_map[fid], bid))
            if blocks:
                articles.append({"id": art_id, "title": title, "blocks": blocks})
    return articles if articles else None


_alto_cache = {}


def extract_tb_text(alto_path, tb_id):
    """Retourne le texte d'un TextBlock ALTO précis (par son ID), en concaténant
    les mots de chaque ligne. Met en cache le contenu de tout le fichier ALTO
    au premier accès (`_alto_cache`) pour éviter de re-parser le XML à chaque
    appel : un même fichier ALTO contient plusieurs blocs d'articles différents.
    """
    key = str(alto_path)
    if key not in _alto_cache:
        if not alto_path.is_file():
            _alto_cache[key] = {}
        else:
            root = ET.parse(alto_path).getroot()
            ns = root.tag.split("}")[0] + "}" if "{" in root.tag else ""
            blocks = {}
            for tb in root.iter(f"{ns}TextBlock"):
                aid = tb.get("ID", "")
                lines = []
                for line in tb.iter(f"{ns}TextLine"):
                    words = [s.get("CONTENT", "") for s in line.iter(f"{ns}String") if s.get("CONTENT")]
                    if words:
                        lines.append(" ".join(words))
                blocks[aid] = "\n".join(lines)
            _alto_cache[key] = blocks
    return _alto_cache[key].get(tb_id, "")


def extract_articles(fascicule):
    """Reconstitue le texte complet de chaque article d'un fascicule : lit le
    TOC/METS pour connaître l'ordre des blocs, puis va chercher le texte de
    chaque bloc dans l'ALTO corrigé par Mistral (pas l'original). Un article
    sans texte (blocs vides) est exclu du résultat.
    """
    toc_folder = SAMPLE_DIR / fascicule / "toc"
    ocr_folder = MISTRAL_RESULTS_DIR / f"{fascicule}_reocr" / "ocr"

    articles_meta = parse_toc(toc_folder) if toc_folder.exists() else None
    if not articles_meta:
        return []

    results = []
    for art in articles_meta:
        parts, pages = [], set()
        for alto_name, tb_id in art["blocks"]:
            txt = extract_tb_text(ocr_folder / alto_name, tb_id)
            if txt:
                parts.append(txt)
            pages.add(alto_name)
        text = "\n\n".join(parts)
        if text.strip():
            results.append({"id": art["id"], "title": art["title"], "text": text, "pages": sorted(pages)})
    return results


#  Prompts et schémas : ÉTAGE 1 (niveau 2) 

STAGE1_SYSTEM_PROMPT = """Tu es un documentaliste spécialisé dans le classement thématique d'archives de presse française ancienne.
On te donne le texte complet de plusieurs articles et une liste fermée de 120 catégories IPTC de niveau 2.
Pour CHAQUE article, choisis UNE SEULE catégorie parmi cette liste qui décrit le mieux le sujet PRINCIPAL de l'article. Choisis UNIQUEMENT parmi les codes fournis.
Réponds uniquement avec un objet JSON de la forme :
{"articles": [{"article_id": "<id>", "level2_code": "<code>"}, ...]}"""


def _stage1_response_schema(valid_l2_codes, article_ids):
    """Schéma JSON strict de l'étage 1 : un tableau d'exactement len(article_ids)
    entrées {article_id, level2_code}, article_id contraint aux IDs réels du
    lot et level2_code contraint aux 120 codes niveau 2 valides : le modèle ne
    peut inventer ni l'un ni l'autre.
    """
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "iptc_stage1_level2_selection",
            "schema": {
                "type": "object",
                "properties": {
                    "articles": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "article_id": {"type": "string", "enum": article_ids},
                                "level2_code": {"type": "string", "enum": valid_l2_codes},
                            },
                            "required": ["article_id", "level2_code"],
                            "additionalProperties": False,
                        },
                        "minItems": len(article_ids),
                        "maxItems": len(article_ids),
                    }
                },
                "required": ["articles"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }


def classify_stage1_batch(batch, l2_leaves_str, valid_l2_codes, retries=5):
    """Un appel HTTP : assigne 1 code niveau 2 à chaque article du lot.
    Retourne ({article_id: level2_code}, usage, temps_s)."""
    ids = [a["id"] for a in batch]
    articles_block = "\n\n".join(f'### Article {a["id"]}\n"""\n{a["text"]}\n"""' for a in batch)

    user_prompt = f"""Catégories niveau 2 disponibles :
{l2_leaves_str}

Voici {len(batch)} article(s) à classer. Pour CHAQUE article, choisis 1 catégorie niveau 2.

{articles_block}

Réponds avec un objet JSON contenant une entrée par article listé ci-dessus (article_id + level2_code)."""

    payload = {
        "model": MISTRAL_MODEL,
        "messages": [
            {"role": "system", "content": STAGE1_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
        "response_format": _stage1_response_schema(valid_l2_codes, ids),
    }
    return _post_with_retries(payload, ids, retries, parse_stage1_response)


def parse_stage1_response(data, ids):
    """Extrait {article_id: level2_code} de la réponse JSON de l'étage 1, en
    ignorant silencieusement toute entrée dont l'article_id ne fait pas partie
    du lot envoyé (garde-fou en plus du schéma strict).
    """
    content = data["choices"][0]["message"]["content"]
    parsed = json.loads(content)
    results = {}
    for entry in parsed.get("articles", []):
        aid = entry.get("article_id")
        code = entry.get("level2_code")
        if aid in ids and isinstance(code, str):
            results[aid] = code
    return results


def classify_stage1_batch_with_fallback(batch, l2_leaves_str, valid_l2_codes):
    """Comme classify_batch_with_fallback (voir classify_iptc_mistral_batched.py)
    mais pour l'étage 1 : si le lot échoue entièrement, le coupe en deux et
    retente chaque moitié récursivement jusqu'à des lots d'1 article si besoin.
    """
    try:
        results, usage, elapsed = classify_stage1_batch(batch, l2_leaves_str, valid_l2_codes)
    except Exception as e:
        if len(batch) == 1:
            print(f"      ✗ [étage 1] {batch[0]['id']} : erreur : {e}")
            return {}, [], []
        mid = len(batch) // 2
        print(f"      ⚠ [étage 1] échec du lot de {len(batch)} article(s) ({e}) : 2 lots ({mid}/{len(batch) - mid})")
        r1, u1, t1 = classify_stage1_batch_with_fallback(batch[:mid], l2_leaves_str, valid_l2_codes)
        r2, u2, t2 = classify_stage1_batch_with_fallback(batch[mid:], l2_leaves_str, valid_l2_codes)
        return {**r1, **r2}, u1 + u2, t1 + t2

    missing = [a["id"] for a in batch if a["id"] not in results]
    if missing:
        print(f"      ⚠ [étage 1] {len(missing)} article(s) absent(s) de la réponse : {missing}")
    return results, [usage], [elapsed]


#  Prompts et schémas : ÉTAGE 2 (niveau 3, par branche) 

STAGE2_SYSTEM_PROMPT_TEMPLATE = """Tu es un documentaliste spécialisé dans le classement thématique d'archives de presse française ancienne.
Ces articles ont déjà été classés dans la catégorie IPTC de niveau 2 « {l2_label} ». On te donne le texte complet de plusieurs articles et la liste fermée des sous-catégories de niveau 3 possibles pour cette catégorie.
Pour CHAQUE article, choisis entre 1 et 5 sous-catégories parmi cette liste qui décrivent le mieux le sujet de l'article. Choisis UNIQUEMENT parmi les codes fournis, classe-les du plus au moins pertinent.
Réponds uniquement avec un objet JSON de la forme :
{{"articles": [{{"article_id": "<id>", "themes": ["<code1>", ...]}}, ...]}}"""


def _stage2_response_schema(valid_l3_codes, article_ids):
    """Schéma JSON strict de l'étage 2 : un tableau d'exactement len(article_ids)
    entrées {article_id, themes}, article_id contraint aux IDs du lot et
    themes (1 à 5 codes) contraint aux enfants de niveau 3 de la seule branche en cours
    (valid_l3_codes) : jamais aux 567 codes complets.
    """
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "iptc_stage2_level3_selection",
            "schema": {
                "type": "object",
                "properties": {
                    "articles": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "article_id": {"type": "string", "enum": article_ids},
                                "themes": {
                                    "type": "array",
                                    "items": {"type": "string", "enum": valid_l3_codes},
                                    "minItems": MIN_THEMES,
                                    "maxItems": MAX_THEMES,
                                },
                            },
                            "required": ["article_id", "themes"],
                            "additionalProperties": False,
                        },
                        "minItems": len(article_ids),
                        "maxItems": len(article_ids),
                    }
                },
                "required": ["articles"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }


def classify_stage2_batch(batch, branch_leaves_str, valid_l3_codes, l2_label, retries=5):
    """Un appel HTTP : classe 1-5 thèmes niveau 3 (au sein d'UNE branche) pour
    chaque article du lot. Retourne ({article_id: [codes]}, usage, temps_s)."""
    ids = [a["id"] for a in batch]
    articles_block = "\n\n".join(f'### Article {a["id"]}\n"""\n{a["text"]}\n"""' for a in batch)

    system_prompt = STAGE2_SYSTEM_PROMPT_TEMPLATE.format(l2_label=l2_label)
    user_prompt = f"""Sous-catégories niveau 3 disponibles (branche « {l2_label} ») :
{branch_leaves_str}

Voici {len(batch)} article(s) à classer.

{articles_block}

Réponds avec un objet JSON contenant une entrée par article listé ci-dessus (article_id + themes)."""

    payload = {
        "model": MISTRAL_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
        "response_format": _stage2_response_schema(valid_l3_codes, ids),
    }
    return _post_with_retries(payload, ids, retries, parse_stage2_response)


def parse_stage2_response(data, ids):
    """Extrait {article_id: [codes niveau 3]} de la réponse JSON de l'étage 2,
    en ignorant les entrées hors du lot envoyé.
    """
    content = data["choices"][0]["message"]["content"]
    parsed = json.loads(content)
    results = {}
    for entry in parsed.get("articles", []):
        aid = entry.get("article_id")
        codes = entry.get("themes", [])
        if aid in ids and isinstance(codes, list):
            results[aid] = [c for c in codes if isinstance(c, str)]
    return results


def classify_stage2_batch_with_fallback(batch, branch_leaves_str, valid_l3_codes, l2_label):
    """Comme classify_stage1_batch_with_fallback, mais pour l'étage 2 : split
    récursif du lot en cas d'échec total, jusqu'à 1 article si nécessaire.
    """
    try:
        results, usage, elapsed = classify_stage2_batch(batch, branch_leaves_str, valid_l3_codes, l2_label)
    except Exception as e:
        if len(batch) == 1:
            print(f"      ✗ [étage 2 : {l2_label}] {batch[0]['id']} : erreur : {e}")
            return {}, [], []
        mid = len(batch) // 2
        print(f"      ⚠ [étage 2 : {l2_label}] échec du lot de {len(batch)} article(s) ({e}) : 2 lots ({mid}/{len(batch) - mid})")
        r1, u1, t1 = classify_stage2_batch_with_fallback(batch[:mid], branch_leaves_str, valid_l3_codes, l2_label)
        r2, u2, t2 = classify_stage2_batch_with_fallback(batch[mid:], branch_leaves_str, valid_l3_codes, l2_label)
        return {**r1, **r2}, u1 + u2, t1 + t2

    missing = [a["id"] for a in batch if a["id"] not in results]
    if missing:
        print(f"      ⚠ [étage 2 : {l2_label}] {len(missing)} article(s) absent(s) : {missing}")
    return results, [usage], [elapsed]


#  HTTP commun (retry différencié, comme les autres scripts) 


def _post_with_retries(payload, ids, retries, parse_fn):
    """Requête HTTP commune aux 2 étages (POST + parsing de la réponse via
    `parse_fn`). Lève SystemExit sur 401/403 (clé invalide : inutile de
    réessayer, on arrête tout), RuntimeError sur 400 (requête invalide : pas de
    nouvelle tentative), retente avec backoff exponentiel sur timeout/429/5xx.
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
    }
    last_err = None
    for attempt in range(retries):
        t0 = time.perf_counter()
        try:
            r = requests.post(MISTRAL_URL, json=payload, headers=headers, timeout=180)
        except requests.exceptions.RequestException as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(2 * (attempt + 1))
            continue
        elapsed = time.perf_counter() - t0

        if r.status_code in (401, 403):
            raise SystemExit(
                f"Erreur d'authentification Mistral ({r.status_code}) : vérifie MISTRAL_API_KEY dans {ENV_FILE}"
            )
        if r.status_code == 400:
            raise RuntimeError(f"Requête invalide (400), pas de nouvelle tentative : {r.text[:300]}")
        if r.status_code == 429 or r.status_code >= 500:
            last_err = f"HTTP {r.status_code}"
            if attempt < retries - 1:
                default_wait = min(30, 2 ** (attempt + 1))
                wait = float(r.headers.get("Retry-After", default_wait))
                time.sleep(wait)
            continue

        try:
            r.raise_for_status()
            data = r.json()
            results = parse_fn(data, ids)
            return results, data.get("usage", {}), elapsed
        except Exception as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(2 * (attempt + 1))

    raise RuntimeError(f"Échec appel Mistral après {retries} tentatives : {last_err}")


#  Suivi usage/temps (identique aux autres scripts) 

USAGE_KEYS = ("prompt_tokens", "completion_tokens", "total_tokens", "cached_tokens")


def _empty_usage():
    """Compteur d'usage initialisé à zéro (tokens, nb d'appels) : un par fascicule,
    rempli au fil des appels API puis agrégé au niveau du run complet.
    """
    return {k: 0 for k in USAGE_KEYS} | {"n_calls": 0}


def _flatten_usage(usage):
    """Aplati le bloc `usage` renvoyé par Mistral (les tokens en cache sont
    nichés dans usage.prompt_tokens_details.cached_tokens) en un dict simple.
    """
    flat = {k: usage.get(k, 0) for k in ("prompt_tokens", "completion_tokens", "total_tokens")}
    flat["cached_tokens"] = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
    return flat


def _accumulate(acc, usages, temps, temps_list):
    """Ajoute au compteur `acc` les tokens d'une liste d'appels HTTP (usages),
    et leurs durées à `temps_list` : mutualisé entre l'étage 1 et l'étage 2.
    """
    for usage, elapsed in zip(usages, temps):
        acc["n_calls"] += 1
        for k, v in _flatten_usage(usage).items():
            acc[k] += v
        temps_list.append(elapsed)


#  Étage 1 : niveau 2, par fascicule 


def run_stage1_fascicule(fascicule, l2_candidates, l2_leaves_str, valid_l2_codes, fixed_overhead_l2, dry_run=False):
    """Extrait les articles d'un fascicule et leur assigne 1 code niveau 2
    chacun (par lots). Retourne (articles_avec_text, {article_id: level2_code},
    usage, temps_reponse[])."""
    articles = extract_articles(fascicule)
    kept = [a for a in articles if len(a["text"].split()) >= MIN_WORDS]
    print(f"  [{fascicule}] {len(articles)} article(s) extrait(s), {len(kept)} retenu(s)")

    usage = _empty_usage()
    temps = []

    if not kept:
        return kept, {}, usage, temps

    batches = make_batches(kept, fixed_overhead_l2)
    print(f"  [{fascicule}] étage 1 : {len(batches)} lot(s) : tailles {[len(b) for b in batches]}")

    if dry_run:
        return kept, {}, usage, temps

    assignments = {}
    usage_lock = threading.Lock()

    def worker(batch):
        """Classe un lot (étage 1) dans un thread du pool : capture les
        variables de la fonction englobante (liste niveau 2, codes valides)."""
        return classify_stage1_batch_with_fallback(batch, l2_leaves_str, valid_l2_codes)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        future_to_batch = {pool.submit(worker, batch): batch for batch in batches}
        for future in as_completed(future_to_batch):
            try:
                results, usages, elapsed_list = future.result()
            except SystemExit:
                for f in future_to_batch:
                    f.cancel()
                raise
            except Exception as e:
                print(f"      ✗ [étage 1] erreur inattendue : {e}")
                continue
            with usage_lock:
                _accumulate(usage, usages, elapsed_list, temps)
            assignments.update(results)

    return kept, assignments, usage, temps


#  Étage 2 : niveau 3, par branche (regroupé sur tous les fascicules) 


def run_stage2_branch(l2_code, branch_articles, l3_by_branch, l2_candidates, fixed_overhead_by_branch):
    """Classe en niveau 3 tous les articles assignés à la branche `l2_code`
    (mélangés entre fascicules), par lots avec la liste réduite aux enfants de
    cette branche. Retourne ({(fascicule, article_id): [codes]}, usage, temps[])."""
    branch_leaves = l3_by_branch[l2_code]
    branch_leaves_str = branch_prompt_str(branch_leaves)
    valid_l3_codes = list(branch_leaves.keys())
    l2_label = l2_candidates[l2_code]["label_fr"]

    fixed_overhead = fixed_overhead_by_branch[l2_code]
    batches = make_batches(branch_articles, fixed_overhead)

    usage = _empty_usage()
    temps = []
    results = {}
    usage_lock = threading.Lock()

    def worker(batch):
        """Classe un lot (étage 2, une branche) dans un thread du pool : 
        capture les variables de la fonction englobante (liste de la
        branche, codes valides, libellé niveau 2)."""
        return classify_stage2_batch_with_fallback(batch, branch_leaves_str, valid_l3_codes, l2_label)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        future_to_batch = {pool.submit(worker, batch): batch for batch in batches}
        for future in as_completed(future_to_batch):
            batch = future_to_batch[future]
            try:
                batch_results, usages, elapsed_list = future.result()
            except SystemExit:
                for f in future_to_batch:
                    f.cancel()
                raise
            except Exception as e:
                print(f"      ✗ [étage 2 : {l2_label}] erreur inattendue : {e}")
                continue
            with usage_lock:
                _accumulate(usage, usages, elapsed_list, temps)
            for art in batch:
                if art["id"] in batch_results:
                    results[(art["_fascicule"], art["id"])] = batch_results[art["id"]]

    return results, usage, temps


def main():
    """Point d'entrée CLI : construit les index de taxonomie (niveau 2 + branches),
    lance l'étage 1 sur tous les fascicules demandés, regroupe les articles par
    branche assignée, lance l'étage 2 par branche, fusionne et sauvegarde un
    JSON par fascicule, puis affiche le bilan tokens/temps/coût de la cascade
    entière (étage 1 + étage 2).
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--fascicule", nargs="+", help="ne traiter que ces fascicules")
    parser.add_argument("--dry-run", action="store_true", help="extraction + composition des lots, sans appel API")
    parser.add_argument("--force", action="store_true", help="retraiter même si le fichier de sortie existe déjà")
    args = parser.parse_args()

    if not args.dry_run and (not MISTRAL_API_KEY or MISTRAL_API_KEY == "mets-ta-cle-ici"):
        raise SystemExit(f"Clé API Mistral manquante : édite {ENV_FILE}")

    l2_candidates, l3_by_branch, l2_leaf_codes = build_stage_indices(TAXONOMY_PATH)
    l2_leaves_str = leaves_prompt_str(l2_candidates)
    valid_l2_codes = list(l2_candidates.keys())
    print(f"✓ étage 1 : {len(l2_candidates)} catégories niveau 2 "
          f"({len(l2_leaf_codes)} terminales, {len(l3_by_branch)} avec enfants niveau 3)")

    fixed_overhead_l2 = count_tokens(STAGE1_SYSTEM_PROMPT) + count_tokens(l2_leaves_str) + 100
    print(f"✓ overhead fixe étage 1 : {fixed_overhead_l2:,} tokens (contre {count_tokens(STAGE1_SYSTEM_PROMPT) + 8775:,} tokens en liste complète)")

    # Overhead fixe par branche pour l'étage 2 (system prompt variable selon la branche + sa liste)
    fixed_overhead_by_branch = {}
    for l2_code, children in l3_by_branch.items():
        l2_label = l2_candidates[l2_code]["label_fr"]
        system_prompt = STAGE2_SYSTEM_PROMPT_TEMPLATE.format(l2_label=l2_label)
        branch_str = branch_prompt_str(children)
        fixed_overhead_by_branch[l2_code] = count_tokens(system_prompt) + count_tokens(branch_str) + 100

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.fascicule:
        fascicules = args.fascicule
    else:
        fascicules = sorted(
            p.name.replace("_reocr", "")
            for p in MISTRAL_RESULTS_DIR.iterdir()
            if p.is_dir() and p.name.endswith("_reocr")
        )

    # Ne retraite pas les fascicules déjà sortis, sauf --force
    todo = [f for f in fascicules if args.force or not (OUTPUT_DIR / f"{f}_themes.json").exists()]
    skipped = [f for f in fascicules if f not in todo]
    if skipped:
        print(f"↷ {len(skipped)} fascicule(s) déjà traité(s) (utilise --force pour retraiter) : {skipped[:5]}{'...' if len(skipped) > 5 else ''}")
    if not todo:
        print("Rien à faire.")
        return

    #  ÉTAGE 1 : tous les fascicules demandés 
    print(f"\n{'=' * 60}\nÉTAGE 1 : classification niveau 2 ({len(todo)} fascicule(s))\n{'=' * 60}")

    all_articles_by_fascicule = {}   # fascicule -> [articles avec text]
    all_assignments = {}             # (fascicule, article_id) -> level2_code
    stage1_usage_total = _empty_usage()
    stage1_temps = []

    for fascicule in todo:
        kept, assignments, usage, temps = run_stage1_fascicule(
            fascicule, l2_candidates, l2_leaves_str, valid_l2_codes, fixed_overhead_l2, dry_run=args.dry_run
        )
        all_articles_by_fascicule[fascicule] = kept
        for art in kept:
            art["_fascicule"] = fascicule
        for aid, code in assignments.items():
            all_assignments[(fascicule, aid)] = code
        for k in stage1_usage_total:
            stage1_usage_total[k] += usage.get(k, 0)
        stage1_temps.extend(temps)

    # Sauvegarde de transparence : quelle branche a été assignée à chaque article
    debug_path = OUTPUT_DIR / "_stage1_assignments.json"
    debug_data = json.loads(debug_path.read_text(encoding="utf-8")) if debug_path.exists() else {}
    for (fascicule, aid), code in all_assignments.items():
        debug_data[f"{fascicule}::{aid}"] = {"level2_code": code, "level2_label": l2_candidates.get(code, {}).get("label_fr", "?")}
    debug_path.write_text(json.dumps(debug_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✓ assignations niveau 2 sauvegardées → {debug_path}")

    if args.dry_run:
        # En dry-run on n'a pas de vraies assignations niveau 2 (pas d'appel API) :
        # on affiche seulement la composition des lots de l'étage 1, comme
        # classify_iptc_mistral_batched.py --dry-run.
        for fascicule, kept in all_articles_by_fascicule.items():
            out = {
                "fascicule": fascicule,
                "usage": _empty_usage(),
                "articles": [{"article_id": a["id"], "title": a["title"], "themes": None} for a in kept],
            }
            with open(OUTPUT_DIR / f"{fascicule}_themes.json", "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)
        print("\n(dry-run : étage 2 non simulé : les branches dépendent de vraies réponses de l'étage 1)")
        return

    #  REGROUPEMENT par branche niveau 2 (mélange entre fascicules) 
    by_branch = defaultdict(list)
    terminal_results = {}   # (fascicule, article_id) -> [code2]  (branches sans enfant niveau 3)

    for fascicule, kept in all_articles_by_fascicule.items():
        for art in kept:
            code2 = all_assignments.get((fascicule, art["id"]))
            if code2 is None:
                continue  # échec étage 1 pour cet article, déjà loggé
            if code2 in l2_leaf_codes:
                terminal_results[(fascicule, art["id"])] = [code2]
            elif code2 in l3_by_branch:
                by_branch[code2].append(art)
            else:
                print(f"  ⚠ code niveau 2 inconnu reçu : {code2} pour {fascicule}/{art['id']}")

    print(f"\n{len(terminal_results)} article(s) terminé(s) dès l'étage 1 (branche sans enfant niveau 3)")
    print(f"{sum(len(v) for v in by_branch.values())} article(s) à classer en étage 2, répartis sur {len(by_branch)} branche(s)")

    #  ÉTAGE 2 : par branche 
    print(f"\n{'=' * 60}\nÉTAGE 2 : classification niveau 3 (par branche)\n{'=' * 60}")

    stage2_usage_total = _empty_usage()
    stage2_temps = []
    stage2_results = {}

    for l2_code, branch_articles in sorted(by_branch.items(), key=lambda x: -len(x[1])):
        l2_label = l2_candidates[l2_code]["label_fr"]
        n_children = len(l3_by_branch[l2_code])
        print(f"\n  Branche « {l2_label} » ({l2_code}) : {len(branch_articles)} article(s), {n_children} sous-catégorie(s) candidate(s)")
        results, usage, temps = run_stage2_branch(l2_code, branch_articles, l3_by_branch, l2_candidates, fixed_overhead_by_branch)
        stage2_results.update(results)
        for k in stage2_usage_total:
            stage2_usage_total[k] += usage.get(k, 0)
        stage2_temps.extend(temps)
        for (fascicule, aid), codes in results.items():
            title = next(a["title"] for a in all_articles_by_fascicule[fascicule] if a["id"] == aid)
            labels = [l3_by_branch[l2_code][c]["label_fr"] for c in codes if c in l3_by_branch[l2_code]]
            print(f"    {aid} : {title[:45]!r} → {', '.join(labels)}")

    # Fusion et sauvegarde par fascicule
    final_themes = dict(terminal_results)
    final_themes.update(stage2_results)

    for fascicule, kept in all_articles_by_fascicule.items():
        out_articles = []
        for art in kept:
            codes = final_themes.get((fascicule, art["id"]))
            if not codes:
                continue
            if (fascicule, art["id"]) in terminal_results:
                themes = [{"code": c, "label_fr": l2_candidates[c]["label_fr"]} for c in codes]
            else:
                l2_code = all_assignments.get((fascicule, art["id"]))
                branch = l3_by_branch.get(l2_code, {})
                themes = [{"code": c, "label_fr": branch[c]["label_fr"]} for c in codes if c in branch]
            out_articles.append({"article_id": art["id"], "title": art["title"], "themes": themes})

        out = {"fascicule": fascicule, "usage": _empty_usage(), "articles": out_articles}
        with open(OUTPUT_DIR / f"{fascicule}_themes.json", "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"\n✓ {fascicule} sauvegardé → {len(out_articles)} article(s) classé(s)")

    # Résumé global (étage 1 + étage 2, cross-fascicule donc pas de détail par fascicule) 
    total_calls = stage1_usage_total["n_calls"] + stage2_usage_total["n_calls"]
    total_prompt = stage1_usage_total["prompt_tokens"] + stage2_usage_total["prompt_tokens"]
    total_cached = stage1_usage_total["cached_tokens"] + stage2_usage_total["cached_tokens"]
    total_completion = stage1_usage_total["completion_tokens"] + stage2_usage_total["completion_tokens"]
    all_temps = stage1_temps + stage2_temps

    print(f"\n{'=' * 60}")
    print(f"Résumé de la cascade : {len(todo)} fascicule(s)")
    print(f"  Étage 1 : {stage1_usage_total['n_calls']} appel(s), {stage1_usage_total['prompt_tokens']:,} tokens prompt, {stage1_usage_total['completion_tokens']:,} tokens sortie")
    print(f"  Étage 2 : {stage2_usage_total['n_calls']} appel(s), {stage2_usage_total['prompt_tokens']:,} tokens prompt, {stage2_usage_total['completion_tokens']:,} tokens sortie")
    print(f"  Total : {total_calls} appel(s), {total_prompt:,} tokens prompt (dont {total_cached:,} en cache), {total_completion:,} tokens sortie")
    if all_temps:
        print(f"  temps de réponse : min {min(all_temps):.2f}s : moyen {statistics.mean(all_temps):.2f}s : max {max(all_temps):.2f}s (total {sum(all_temps):.1f}s)")

    pin, pout = PRICING.get(MISTRAL_MODEL, (None, None))
    if pin is not None:
        uncached = total_prompt - total_cached
        cost = uncached / 1e6 * pin + total_cached / 1e6 * pin * 0.10 + total_completion / 1e6 * pout
        print(f"  coût estimé ({MISTRAL_MODEL}, cache pris en compte) : ${cost:.2f}")


if __name__ == "__main__":
    main()
