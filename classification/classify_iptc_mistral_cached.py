"""
classify_iptc_mistral_cached.py — Variante de classify_iptc_mistral.py optimisée
pour le cache de prompt Mistral (voir classify_iptc_mistral.py pour la version
de référence, non modifiée).

Différence avec la version de base :
  - Le contenu FIXE (instructions système + liste des 567 étiquettes) est
    entièrement regroupé dans le message `system`, identique à chaque appel.
    Seul le texte de l'article (variable) est dans le message `user`.
  - Un `prompt_cache_key` constant est envoyé pour aider Mistral à reconnaître
    le préfixe répété.
  Objectif : dès le 2e appel, Mistral facture le préfixe fixe à ~10% du prix
  normal au lieu de le retraiter entièrement (cache automatique côté serveur).

Pour chaque article d'un fascicule :
  1. Extrait le texte complet de l'article (blocs ALTO corrigés, dans l'ordre
     donné par le TOC/METS du fascicule)
  2. Envoie ce texte + la liste des étiquettes IPTC "terminales" (niveau 3,
     ou niveau 2 quand la branche s'arrête là) à Mistral
  3. Récupère 1 à 5 thèmes choisis par Mistral parmi cette liste
  4. Sauvegarde un JSON par fascicule : {article_id: thèmes}

Entrées (voir README.md pour l'arborescence complète du dépôt) :
  - re-ocr/corpus/original/{fascicule}/toc/T*.xml         → structure logique (METS) : ordre des blocs par article
  - re-ocr/corpus/reocr_mistral/{fascicule}_reocr/ocr/*.xml → ALTO avec OCR corrigé (texte des blocs)
  - classification/iptc_mediatopic_official.json                 → taxonomie IPTC officielle (SKOS)

Sortie :
  - results/feuilles_mistral_cached/{fascicule}_themes.json

Usage :
    python classify_iptc_mistral_cached.py --fascicule 4109000 --dry-run   # extraction seule, sans appel API
    python classify_iptc_mistral_cached.py --fascicule 4109000             # test sur 1 fascicule
    python classify_iptc_mistral_cached.py                                 # tous les fascicules
"""

import argparse
import json
import os
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
#   ├── results/feuilles_mistral_cached/            <- sorties de ce script
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
OUTPUT_DIR = PROJECT_DIR / "results" / "feuilles_mistral_cached"
PROMPT_CACHE_KEY = "classify-iptc-v1"

MIN_WORDS = 10          # articles plus courts que ça sont ignorés (titres/rubriques vides)
MAX_THEMES = 5
MIN_THEMES = 1
MAX_WORKERS = 5         # appels Mistral en parallèle (par fascicule)

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
{"themes": ["<code1>", "<code2>", ...]}
Étiquettes disponibles :"""


def _response_schema(valid_codes):
    """JSON Schema contraignant `themes` à un sous-ensemble des codes IPTC valides.

    Avec `strict: true`, Mistral ne peut plus renvoyer de code hors taxonomie
    (fini le filtrage a posteriori des codes invalides).
    """
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "iptc_theme_selection",
            "schema": {
                "type": "object",
                "properties": {
                    "themes": {
                        "type": "array",
                        "items": {"type": "string", "enum": valid_codes},
                        "minItems": MIN_THEMES,
                        "maxItems": MAX_THEMES,
                    }
                },
                "required": ["themes"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }


def classify_article(text, leaves_str, valid_codes, retries=5):
    """Retourne (codes, usage). Lève SystemExit sur erreur d'authentification
    (401/403 — inutile de réessayer, on arrête tout), RuntimeError sur requête
    invalide (400 — pas de nouvelle tentative pour ce seul article), et retente
    avec backoff sur timeout/erreur réseau/429/5xx."""
    # Tout le contenu FIXE (instructions + liste des étiquettes) est regroupé
    # ici, dans le message system — identique à chaque appel, donc mise en
    # cache possible côté Mistral. Seul `user_prompt` (le texte de l'article)
    # varie d'un appel à l'autre.
    system_content = SYSTEM_PROMPT + "\n" + leaves_str
    user_prompt = f"""Texte de l'article :
\"\"\"
{text}
\"\"\"

Réponds avec le JSON demandé."""

    payload = {
        "model": MISTRAL_MODEL,
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
        "response_format": _response_schema(valid_codes),
        "prompt_cache_key": PROMPT_CACHE_KEY,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
    }

    last_err = None
    for attempt in range(retries):
        try:
            r = requests.post(MISTRAL_URL, json=payload, headers=headers, timeout=120)
        except requests.exceptions.RequestException as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(2 * (attempt + 1))
            continue

        if r.status_code in (401, 403):
            raise SystemExit(
                f"Erreur d'authentification Mistral ({r.status_code}) — vérifie MISTRAL_API_KEY dans {ENV_FILE}"
            )
        if r.status_code == 400:
            raise RuntimeError(f"Requête invalide (400), pas de nouvelle tentative : {r.text[:300]}")
        if r.status_code == 429 or r.status_code >= 500:
            last_err = f"HTTP {r.status_code}"
            if attempt < retries - 1:
                # backoff exponentiel (429 = on tape la limite de débit avec
                # MAX_WORKERS appels en parallèle, il faut souvent plus que
                # quelques secondes pour que ça se libère)
                default_wait = min(30, 2 ** (attempt + 1))
                wait = float(r.headers.get("Retry-After", default_wait))
                time.sleep(wait)
            continue

        try:
            r.raise_for_status()
            data = r.json()
            content = data["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            codes = parsed.get("themes", [])
            if isinstance(codes, str):
                codes = [codes]
            codes = [c for c in codes if isinstance(c, str)]
            return codes, data.get("usage", {})
        except Exception as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(2 * (attempt + 1))

    raise RuntimeError(f"Échec appel Mistral après {retries} tentatives : {last_err}")


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


def process_fascicule(fascicule, leaves, leaves_str, valid_codes, dry_run=False):
    """Traite un fascicule entier : extrait ses articles, filtre les trop courts
    (< MIN_WORDS), puis classe chaque article INDIVIDUELLEMENT (1 appel HTTP par article (préfixe fixe regroupé côté `system`
    pour la mise en cache, voir l'en-tête du fichier)) en parallèle (ThreadPoolExecutor). Retourne le JSON de sortie
    du fascicule (articles + usage tokens/temps cumulés) — ou une version
    "squelette" (themes=None) en mode --dry-run, sans appel API.
    """
    articles = extract_articles(fascicule)
    print(f"  {len(articles)} article(s) extrait(s)")

    kept = [a for a in articles if len(a["text"].split()) >= MIN_WORDS]
    print(f"  {len(kept)} article(s) retenu(s) (≥ {MIN_WORDS} mots)")

    out = {"fascicule": fascicule, "usage": _empty_usage(), "articles": []}

    if dry_run:
        out["articles"] = [{"article_id": a["id"], "title": a["title"], "themes": None} for a in kept]
        return out

    results_by_index = {}
    usage_lock = threading.Lock()

    def worker(art):
        """Classe un seul article dans un thread du pool — capture `leaves_str` et
        `valid_codes` de la fonction englobante.
        """
        return classify_article(art["text"], leaves_str, valid_codes)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        future_to_item = {pool.submit(worker, art): (i, art) for i, art in enumerate(kept)}
        for future in as_completed(future_to_item):
            i, art = future_to_item[future]
            try:
                codes, usage = future.result()
            except SystemExit:
                # erreur d'authentification : inutile de laisser tourner les autres
                # appels déjà en file d'attente avec la même clé cassée.
                for f in future_to_item:
                    f.cancel()
                raise
            except Exception as e:
                print(f"      ✗ {art['id']} — erreur classification : {e}")
                continue

            with usage_lock:
                out["usage"]["n_calls"] += 1
                for k, v in _flatten_usage(usage).items():
                    out["usage"][k] += v

            valid = codes[:MAX_THEMES]
            if len(valid) < MIN_THEMES:
                print(f"      ⚠ aucun thème valide retourné pour {art['id']}")
                continue

            themes = [{"code": c, "label_fr": leaves[c]["label_fr"]} for c in valid]
            results_by_index[i] = {"article_id": art["id"], "title": art["title"], "themes": themes}
            print(f"    [{i+1}/{len(kept)}] {art['id']} — {art['title'][:50]!r} → {', '.join(t['label_fr'] for t in themes)}")

    out["articles"] = [results_by_index[i] for i in sorted(results_by_index)]
    return out


def main():
    """Point d'entrée CLI : charge la taxonomie (567 étiquettes), détermine les
    fascicules à traiter (argument --fascicule ou tout resultats_mistral/),
    saute ceux déjà sortis (sauf --force), classe chaque fascicule restant et
    sauvegarde son JSON, puis affiche le bilan tokens/temps/coût du run entier.
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

    for fascicule in fascicules:
        print(f"\n══ Fascicule {fascicule} ══")
        out_path = OUTPUT_DIR / f"{fascicule}_themes.json"

        if out_path.exists() and not args.force:
            print(f"  ↷ déjà traité — {out_path.name} (utilise --force pour retraiter)")
            try:
                prev_usage = json.loads(out_path.read_text(encoding="utf-8")).get("usage", {})
                for k in grand_total:
                    grand_total[k] += prev_usage.get(k, 0)
            except Exception:
                pass
            continue

        result = process_fascicule(fascicule, leaves, leaves_str, valid_codes, dry_run=args.dry_run)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"  ✓ sauvegardé → {out_path}")

        if not args.dry_run:
            for k in grand_total:
                grand_total[k] += result["usage"].get(k, 0)

    if not args.dry_run and grand_total["n_calls"]:
        print(f"\n{'='*55}")
        print(f"Total : {grand_total['n_calls']} appel(s) API")
        print(f"  prompt_tokens     : {grand_total['prompt_tokens']:,}  (dont en cache : {grand_total['cached_tokens']:,})")
        print(f"  completion_tokens : {grand_total['completion_tokens']:,}")
        print(f"  total_tokens      : {grand_total['total_tokens']:,}")
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
