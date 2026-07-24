"""
classify_iptc_mistral_batched.py — Variante de classify_iptc_mistral.py qui
groupe plusieurs articles par appel Mistral (voir classify_iptc_mistral.py
pour la version de référence 1-article/appel, non modifiée).

Principe :
  - Le coût FIXE (system + 567 étiquettes, ~8900 tokens réels — mesuré avec
    le tokenizer Mistral, pas juste estimé en caractères) est payé une seule
    fois par appel, peu importe le nombre d'articles dedans. Regrouper N
    articles par appel divise ce coût fixe par N.
  - Les articles d'un fascicule sont empilés dans des lots ("batches") tant
    que le total (overhead fixe + tokens des articles déjà dans le lot) reste
    sous MAX_BATCH_TOKENS et que le lot a moins de MAX_BATCH_SIZE articles.
    Un article est TOUJOURS envoyé en entier — jamais coupé — y compris s'il
    dépasse à lui seul le plafond (il part alors seul dans son propre lot).
  - Si un appel groupé échoue après ses tentatives, le lot est coupé en deux
    et chaque moitié est retentée récursivement (jusqu'à des lots d'1 article
    si besoin) plutôt que de perdre tout le lot d'un coup.
  - Comptage de tokens réel via `mistral-common` (tokenizer Mistral ouvert,
    téléchargé une fois depuis Hugging Face) ; repli sur une estimation par
    caractères si le paquet ou le réseau ne sont pas disponibles.

Pour chaque article d'un fascicule :
  1. Extrait le texte complet de l'article (blocs ALTO corrigés, dans l'ordre
     donné par le TOC/METS du fascicule)
  2. Regroupe les articles en lots selon le budget de tokens
  3. Envoie chaque lot + la liste des étiquettes IPTC "terminales" à Mistral
  4. Récupère 1 à 5 thèmes par article choisis par Mistral parmi cette liste
  5. Sauvegarde un JSON par fascicule : {article_id: thèmes}

Entrées (voir README.md pour l'arborescence complète du dépôt) :
  - re-ocr/corpus/original/{fascicule}/toc/T*.xml         → structure logique (METS) : ordre des blocs par article
  - re-ocr/corpus/reocr_mistral/{fascicule}_reocr/ocr/*.xml → ALTO avec OCR corrigé (texte des blocs)
  - classification/iptc_mediatopic_official.json                 → taxonomie IPTC officielle (SKOS)

Sortie :
  - results/feuilles_mistral_batched/{fascicule}_themes.json

Usage :
    python classify_iptc_mistral_batched.py --fascicule 4109000 --dry-run   # extraction + composition des lots, sans appel API
    python classify_iptc_mistral_batched.py --fascicule 4109000             # test sur 1 fascicule
    python classify_iptc_mistral_batched.py                                 # tous les fascicules
"""

import argparse
import json
import os
import statistics
import threading
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

# ── Chargement .env ───────────────────────────────────────────────────────────


def load_env(env_path: Path):
    """Charge les variables d'un fichier .env dans os.environ (sans dépendance
    externe) — ne remplace jamais une variable déjà définie dans l'environnement.
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
#   ├── results/feuilles_mistral_batched/            <- sorties de ce script
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
OUTPUT_DIR = PROJECT_DIR / "results" / "feuilles_mistral_batched"

MIN_WORDS = 10          # articles plus courts que ça sont ignorés (titres/rubriques vides)
MAX_THEMES = 5
MIN_THEMES = 1
MAX_WORKERS = 5         # lots Mistral en parallèle (par fascicule)

# ── Groupage d'articles par appel ──────────────────────────────────────────────
MAX_BATCH_TOKENS = 40_000   # budget de tokens par appel (fixe + articles) ; très
                            # sous les 128k-256k de contexte réels, marge large
MAX_BATCH_SIZE = 25         # nb max d'articles par appel, indépendamment des tokens
PER_ARTICLE_WRAPPER_TOKENS = 15  # overhead du gabarit "### Article {id}\n\"\"\"…\"\"\"" par article, marge incluse

# $ / 1M tokens (input, output) — mistral.ai/pricing/api/, pour l'estimation de coût
PRICING = {
    "mistral-large-latest": (0.50, 1.50),
    "mistral-medium-latest": (1.50, 7.50),
    "mistral-small-latest": (0.15, 0.60),
    "ministral-8b-latest": (0.15, 0.15),
    "ministral-3b-latest": (0.10, 0.10),
}

# ── Taxonomie IPTC officielle (SKOS) : index des étiquettes candidates ────────
#
# La taxonomie officielle va jusqu'à 6 niveaux de profondeur, encodés via des
# relations SKOS broader/narrower (pas via la structure du code à 8 chiffres).
# On la plafonne au niveau 3 : tous les concepts niveau 3, plus les concepts
# niveau 2 qui n'ont pas de descendant niveau 3 (branche qui s'arrête là).


# Traductions manuelles pour les concepts que l'IPTC n'a pas encore traduits
# en français (repli automatique sur l'anglais sinon).
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

    Étiquette candidate = concept de niveau `max_level`, ou concept de niveau
    < max_level sans aucun enfant (branche qui s'arrête avant).
    Profondeur calculée par parcours BFS des relations broader depuis les 17
    concepts racine (niveau 1), en ignorant les concepts retirés (retired).
    """
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
        """Remonte jusqu'à l'ancêtre du concept `uri` situé au niveau `level`."""
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


def leaves_prompt_str(leaves):
    """Une ligne par étiquette : code — libellé.

    Le contexte parent entre parenthèses n'est ajouté que pour les libellés
    qui apparaissent plusieurs fois dans la liste (ambigus sans lui) — pour
    toutes les autres étiquettes (l'immense majorité), le libellé seul suffit.
    """
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
        lines.append(f"  {code} — {leaf['label_fr']}{ctx}")
    return "\n".join(lines)


# ── Comptage de tokens (réel via mistral-common, repli en caractères sinon) ───

_tokenizer = None
_tokenizer_load_failed = False
_TOKENIZER_HF_REPO = "mistralai/Ministral-8B-Instruct-2410"  # tokenizer ouvert, non gated
_FALLBACK_CHARS_PER_TOKEN = 3.5  # mesuré ~3.6-4.2 sur nos articles ; marge de sécurité


def _get_tokenizer():
    """Charge le tokenizer Mistral une seule fois (télécharge depuis HF au
    premier appel). Retourne None si indisponible (pas de réseau, paquet
    manquant...) — count_tokens() bascule alors sur l'estimation par
    caractères, volontairement pessimiste (sous-estimer coûterait de dépasser
    le budget ; surestimer fait juste des lots un peu plus petits)."""
    global _tokenizer, _tokenizer_load_failed
    if _tokenizer is not None or _tokenizer_load_failed:
        return _tokenizer
    try:
        from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
        _tokenizer = MistralTokenizer.from_hf_hub(_TOKENIZER_HF_REPO)
    except Exception as e:
        print(f"  ⚠ tokenizer Mistral indisponible ({e}) — repli sur l'estimation par caractères")
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
    """Empile des articles ENTIERS dans des lots tant que le budget de tokens
    et la taille max ne sont pas dépassés. Un article est toujours envoyé en
    entier, jamais coupé — s'il dépasse le budget à lui seul, il part seul
    dans son propre lot (jamais rejeté ni tronqué)."""
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


# ── Extraction des articles (TOC/METS + ALTO corrigé) ─────────────────────────

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
    appel — un même fichier ALTO contient plusieurs blocs d'articles différents.
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


# ── Appel Mistral ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """Tu es un documentaliste spécialisé dans le classement thématique d'archives de presse française ancienne.
On te donne le texte complet d'un article et une liste fermée d'étiquettes.
Choisis entre 1 et 5 étiquettes de cette liste qui décrivent le mieux le sujet de l'article. Choisis UNIQUEMENT parmi les codes fournis, ne choisis que les plus pertinents (pas besoin d'en mettre 5 si 1 ou 2 suffisent), et classe-les du plus au moins pertinent.
Réponds uniquement avec un objet JSON de la forme :
{"themes": ["<code1>", "<code2>", ...]}"""


def _batch_response_schema(valid_codes, article_ids):
    """JSON Schema pour un lot de N articles : un tableau d'exactement N
    entrées {article_id, themes}, `article_id` contraint aux IDs réels du lot
    (le modèle ne peut pas en inventer) et `themes` aux codes IPTC valides."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "iptc_batch_theme_selection",
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
                                    "items": {"type": "string", "enum": valid_codes},
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


def classify_batch(batch, leaves_str, valid_codes, retries=5):
    """Un seul appel HTTP pour classer tous les articles de `batch` (liste de
    {id, title, text, ...}). Retourne ({article_id: codes}, usage, temps_s).
    `temps_s` est la durée de la requête HTTP qui a effectivement réussi (hors
    attente de backoff des tentatives précédentes).

    Lève SystemExit sur erreur d'authentification (401/403 — inutile de
    réessayer, on arrête tout), RuntimeError sur requête invalide (400 — pas
    de nouvelle tentative), et retente avec backoff sur timeout/réseau/429/5xx.
    """
    ids = [a["id"] for a in batch]
    articles_block = "\n\n".join(f'### Article {a["id"]}\n"""\n{a["text"]}\n"""' for a in batch)

    user_prompt = f"""Étiquettes disponibles :
{leaves_str}

Voici {len(batch)} article(s) à classer. Pour CHAQUE article ci-dessous, choisis 1 à 5 étiquettes.

{articles_block}

Réponds avec un objet JSON contenant une entrée par article listé ci-dessus (article_id + themes)."""

    payload = {
        "model": MISTRAL_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
        "response_format": _batch_response_schema(valid_codes, ids),
    }
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
                f"Erreur d'authentification Mistral ({r.status_code}) — vérifie MISTRAL_API_KEY dans {ENV_FILE}"
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
            content = data["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            results = {}
            for entry in parsed.get("articles", []):
                aid = entry.get("article_id")
                codes = entry.get("themes", [])
                if aid in ids and isinstance(codes, list):
                    results[aid] = [c for c in codes if isinstance(c, str)]
            return results, data.get("usage", {}), elapsed
        except Exception as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(2 * (attempt + 1))

    raise RuntimeError(f"Échec appel Mistral après {retries} tentatives : {last_err}")


def classify_batch_with_fallback(batch, leaves_str, valid_codes):
    """Essaie classify_batch() sur tout le lot. Si l'appel échoue entièrement
    (après ses propres tentatives), coupe le lot en deux et retente chaque
    moitié récursivement — jusqu'à des lots d'1 article si nécessaire — plutôt
    que de perdre tous les articles du lot d'un coup.

    Retourne (results, usages, temps) où `usages` et `temps` sont les listes
    de `usage`/durée de chaque appel HTTP effectivement réalisé (un lot qui a
    dû être scindé en donne plusieurs)."""
    try:
        results, usage, elapsed = classify_batch(batch, leaves_str, valid_codes)
    except Exception as e:
        if len(batch) == 1:
            print(f"      ✗ {batch[0]['id']} — erreur classification : {e}")
            return {}, [], []
        mid = len(batch) // 2
        print(f"      ⚠ échec du lot de {len(batch)} article(s) ({e}) — nouvelle tentative en 2 lots ({mid}/{len(batch) - mid})")
        r1, u1, t1 = classify_batch_with_fallback(batch[:mid], leaves_str, valid_codes)
        r2, u2, t2 = classify_batch_with_fallback(batch[mid:], leaves_str, valid_codes)
        return {**r1, **r2}, u1 + u2, t1 + t2

    missing = [a["id"] for a in batch if a["id"] not in results]
    if missing:
        print(f"      ⚠ {len(missing)} article(s) absent(s) de la réponse du lot : {missing}")

    return results, [usage], [elapsed]


# ── Pipeline par fascicule ─────────────────────────────────────────────────────


USAGE_KEYS = ("prompt_tokens", "completion_tokens", "total_tokens", "cached_tokens")


def _empty_usage():
    """Compteur d'usage initialisé à zéro (tokens, nb d'appels) — un par fascicule,
    rempli au fil des appels API puis agrégé au niveau du run complet.
    """
    return {k: 0 for k in USAGE_KEYS} | {"n_calls": 0}


def _flatten_usage(usage):
    """Mistral renvoie cached_tokens niché dans prompt_tokens_details."""
    flat = {k: usage.get(k, 0) for k in ("prompt_tokens", "completion_tokens", "total_tokens")}
    flat["cached_tokens"] = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
    return flat


def process_fascicule(fascicule, leaves, leaves_str, valid_codes, fixed_overhead_tokens, dry_run=False):
    """Traite un fascicule entier : extrait ses articles, filtre les trop courts,
    les regroupe en LOTS via make_batches() (plusieurs articles par appel),
    puis classe chaque lot en parallèle (ThreadPoolExecutor). Retourne le JSON
    de sortie du fascicule (articles + usage tokens/temps cumulés) — ou la
    composition des lots seule en mode --dry-run, sans appel API.
    """
    articles = extract_articles(fascicule)
    print(f"  {len(articles)} article(s) extrait(s)")

    kept = [a for a in articles if len(a["text"].split()) >= MIN_WORDS]
    print(f"  {len(kept)} article(s) retenu(s) (≥ {MIN_WORDS} mots)")

    out = {"fascicule": fascicule, "usage": _empty_usage(), "articles": []}

    batches = make_batches(kept, fixed_overhead_tokens)
    sizes = [len(b) for b in batches]
    print(f"  {len(batches)} lot(s) constitué(s) — tailles : {sizes}")

    if dry_run:
        out["articles"] = [{"article_id": a["id"], "title": a["title"], "themes": None} for a in kept]
        return out

    order = {a["id"]: i for i, a in enumerate(kept)}
    by_id = {a["id"]: a for a in kept}
    results_by_id = {}
    temps_reponse = []
    usage_lock = threading.Lock()

    def worker(batch):
        """Classe un lot entier (plusieurs articles) dans un thread du pool — capture
        `leaves_str` et `valid_codes` de la fonction englobante.
        """
        return classify_batch_with_fallback(batch, leaves_str, valid_codes)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        future_to_batch = {pool.submit(worker, batch): batch for batch in batches}
        for future in as_completed(future_to_batch):
            batch = future_to_batch[future]
            try:
                results, usages, temps = future.result()
            except SystemExit:
                # erreur d'authentification : inutile de laisser tourner les autres
                # lots déjà en file d'attente avec la même clé cassée.
                for f in future_to_batch:
                    f.cancel()
                raise
            except Exception as e:
                ids = [a["id"] for a in batch]
                print(f"      ✗ lot {ids} — erreur inattendue : {e}")
                continue

            with usage_lock:
                for usage, elapsed in zip(usages, temps):
                    out["usage"]["n_calls"] += 1
                    for k, v in _flatten_usage(usage).items():
                        out["usage"][k] += v
                    temps_reponse.append(elapsed)

            temps_str = ", ".join(f"{t:.2f}s" for t in temps)
            print(f"    Lot de {len(batch)} article(s) — {len(temps)} appel(s) HTTP ({temps_str})")
            for art_id, codes in results.items():
                art = by_id[art_id]
                valid = codes[:MAX_THEMES]
                if len(valid) < MIN_THEMES:
                    print(f"      ⚠ aucun thème valide retourné pour {art_id}")
                    continue
                themes = [{"code": c, "label_fr": leaves[c]["label_fr"]} for c in valid]
                results_by_id[art_id] = {"article_id": art_id, "title": art["title"], "themes": themes}
                print(f"      {art_id} — {art['title'][:50]!r} → {', '.join(t['label_fr'] for t in themes)}")

    if temps_reponse:
        out["usage"]["temps_reponse_total_s"] = round(sum(temps_reponse), 2)
        out["usage"]["temps_reponse_min_s"] = round(min(temps_reponse), 2)
        out["usage"]["temps_reponse_max_s"] = round(max(temps_reponse), 2)
        out["usage"]["temps_reponse_moyen_s"] = round(statistics.mean(temps_reponse), 2)
        out["usage"]["temps_reponse_median_s"] = round(statistics.median(temps_reponse), 2)

    out["articles"] = [results_by_id[aid] for aid in sorted(results_by_id, key=lambda x: order[x])]
    return out


def main():
    """Point d'entrée CLI : charge la taxonomie, détermine les fascicules à
    traiter, calcule l'overhead fixe par appel (system + liste), saute les
    fascicules déjà sortis (sauf --force), classe chaque fascicule restant par
    lots et sauvegarde son JSON, puis affiche le bilan tokens/temps/coût du run.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--fascicule", nargs="+", help="ne traiter que ces fascicules (ex. --fascicule 4109000 4109676)")
    parser.add_argument("--dry-run", action="store_true", help="extraction seule, pas d'appel API")
    parser.add_argument("--force", action="store_true", help="retraiter même si le fichier de sortie existe déjà")
    args = parser.parse_args()

    if not args.dry_run and (not MISTRAL_API_KEY or MISTRAL_API_KEY == "mets-ta-cle-ici"):
        raise SystemExit(f"Clé API Mistral manquante — édite {ENV_FILE}")

    leaves = build_leaves(TAXONOMY_PATH)
    leaves_str = leaves_prompt_str(leaves)
    valid_codes = list(leaves.keys())
    print(f"✓ {len(leaves)} étiquettes terminales chargées depuis {TAXONOMY_PATH.name}")

    # Coût fixe (system + étiquettes) mesuré une seule fois avec le vrai
    # tokenizer, réutilisé pour composer les lots de chaque fascicule.
    fixed_overhead_tokens = count_tokens(SYSTEM_PROMPT) + count_tokens(leaves_str) + 100
    print(f"✓ overhead fixe par appel estimé à {fixed_overhead_tokens:,} tokens "
          f"(budget par lot : {MAX_BATCH_TOKENS:,} tokens, {MAX_BATCH_SIZE} articles max)")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.fascicule:
        fascicules = args.fascicule
    else:
        fascicules = sorted(
            p.name.replace("_reocr", "")
            for p in MISTRAL_RESULTS_DIR.iterdir()
            if p.is_dir() and p.name.endswith("_reocr")
        )

    grand_total = _empty_usage()
    temps_total_global = 0.0
    temps_min_global, temps_max_global = None, None

    for fascicule in fascicules:
        print(f"\n══ Fascicule {fascicule} ══")
        out_path = OUTPUT_DIR / f"{fascicule}_themes.json"

        if out_path.exists() and not args.force:
            print(f"  ↷ déjà traité — {out_path.name} (utilise --force pour retraiter)")
            try:
                prev_usage = json.loads(out_path.read_text(encoding="utf-8")).get("usage", {})
                for k in grand_total:
                    grand_total[k] += prev_usage.get(k, 0)
                temps_total_global += prev_usage.get("temps_reponse_total_s", 0)
                if "temps_reponse_min_s" in prev_usage:
                    temps_min_global = min(filter(None, [temps_min_global, prev_usage["temps_reponse_min_s"]]))
                    temps_max_global = max(filter(None, [temps_max_global, prev_usage["temps_reponse_max_s"]]))
            except Exception:
                pass
            continue

        result = process_fascicule(
            fascicule, leaves, leaves_str, valid_codes, fixed_overhead_tokens, dry_run=args.dry_run
        )
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"  ✓ sauvegardé → {out_path}")

        if not args.dry_run:
            for k in grand_total:
                grand_total[k] += result["usage"].get(k, 0)
            temps_total_global += result["usage"].get("temps_reponse_total_s", 0)
            if "temps_reponse_min_s" in result["usage"]:
                temps_min_global = min(filter(None, [temps_min_global, result["usage"]["temps_reponse_min_s"]]))
                temps_max_global = max(filter(None, [temps_max_global, result["usage"]["temps_reponse_max_s"]]))

    if not args.dry_run and grand_total["n_calls"]:
        print(f"\n{'='*55}")
        print(f"Total : {grand_total['n_calls']} appel(s) API")
        print(f"  prompt_tokens     : {grand_total['prompt_tokens']:,}  (dont en cache : {grand_total['cached_tokens']:,})")
        print(f"  completion_tokens : {grand_total['completion_tokens']:,}")
        print(f"  total_tokens      : {grand_total['total_tokens']:,}")
        if temps_total_global:
            temps_moyen = temps_total_global / grand_total["n_calls"]
            print(f"  temps de réponse  : min {temps_min_global:.2f}s — moyen {temps_moyen:.2f}s — max {temps_max_global:.2f}s (total {temps_total_global:.1f}s)")
        pin, pout = PRICING.get(MISTRAL_MODEL, (None, None))
        if pin is not None:
            uncached_input = grand_total["prompt_tokens"] - grand_total["cached_tokens"]
            cost = (
                uncached_input / 1e6 * pin
                + grand_total["cached_tokens"] / 1e6 * pin * 0.10
                + grand_total["completion_tokens"] / 1e6 * pout
            )
            print(f"  coût estimé ({MISTRAL_MODEL}, cache pris en compte) : ${cost:.2f}")
        else:
            print(f"  (tarif inconnu pour {MISTRAL_MODEL}, coût non estimé)")


if __name__ == "__main__":
    main()
