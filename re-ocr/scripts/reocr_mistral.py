"""
reocr_mistral.py : ré-océrisation des blocs de texte par un modèle multimodal

CE QUE FAIT CE SCRIPT

Reprend la reconnaissance de caractères des fascicules numérisés, bloc par bloc,
en soumettant l'image de chaque bloc à un modèle multimodal.

POURQUOI RÉ-OCÉRISER

Le texte livré avec les fascicules provient d'une océrisation en masse conduite
à partir de 2008 sur des microformes. Sa qualité conditionne tout traitement
ultérieur. Le relevé conduit dans la deuxième partie du mémoire montre que 27
pour cent des formes du corpus ne sont attestées que par cette océrisation
d'origine : ce ne sont pas des erreurs isolées mais un vocabulaire parasite, que
toute méthode fondée sur les fréquences compte comme si de rien n'était.

LE TRAITEMENT, ÉTAPE PAR ÉTAPE

  1. Lire le fichier ALTO de la page, qui donne pour chaque bloc ses coordonnées
     en pixels de l'image d'origine.
  2. Ouvrir l'image et en découper la région du bloc, avec une marge.
  3. Encoder ce découpage en base64 et le soumettre au modèle.
  4. Remplacer le texte du bloc dans le fichier ALTO par la transcription
     obtenue, en conservant la structure du fichier.
  5. Enregistrer les deux versions côte à côte, ce qui permet de mesurer l'écart
     et de revenir en arrière.

POURQUOI PROCÉDER BLOC PAR BLOC

Soumettre la page entière donnerait au modèle une mise en page à plusieurs
colonnes qu'il lirait dans un ordre incertain. Le bloc est une région
rectangulaire d'une seule colonne, dont la lecture ne présente aucune ambiguïté.
Le découpage préserve en outre la structure du fichier ALTO, à laquelle la carte
logique du METS renvoie pour reconstituer les articles.

UNE DIFFICULTÉ PROPRE AUX MODÈLES GÉNÉRATIFS

Un modèle de ce genre produit un texte plausible, sans garantie qu'il soit
exact. Le mémoire en rapporte un cas : le modèle a restitué des noms là où
l'image ne portait plus de caractères lisibles. La conservation du texte
d'origine à côté de la transcription permet de repérer ces cas.

ENTRÉES

  Sample/pt1, pt2, pt3/          images des pages
  Sample/sample_iiif/            fichiers ALTO d'origine
  config/.env                    clé API, ou option --key

SORTIES

  resultats_mistral/{fascicule}_reocr/ocr/    ALTO avec le texte corrigé
  resultats_mistral/{fascicule}_reocr/logs/   les deux versions de chaque bloc

PAQUETS EMPLOYÉS

  argparse, base64, io, json, os, re, threading, time, pathlib   bibliothèque
                            standard
  xml.etree.ElementTree     lecture et réécriture des fichiers ALTO
  concurrent.futures        appels en parallèle
  requests                  appels HTTP à l'interface
  Pillow                    ouverture des images et découpage des régions

USAGE

    python reocr_mistral.py --key VOTRE_CLE
    python reocr_mistral.py --key VOTRE_CLE --fascicule 4109000
    python reocr_mistral.py --key VOTRE_CLE --page 1
    python reocr_mistral.py --key VOTRE_CLE --dry-run
"""

import argparse
import base64
import io
import json
import os
import re
import threading
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from PIL import Image
from tqdm import tqdm

# ── Chargement .env ───────────────────────────────────────────────────────────

def load_env(env_path: Path):
    """Charge un fichier .env dans os.environ (sans dépendance python-dotenv)."""
    if not env_path.exists():
        raise FileNotFoundError(f".env introuvable : {env_path}")
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

ENV_FILE = Path(__file__).parent / "config" / ".env"
load_env(ENV_FILE)

MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "")
if not MISTRAL_API_KEY or MISTRAL_API_KEY == "mets-ta-cle-ici":
    raise ValueError(f"Clé API manquante : édite {ENV_FILE}")

# ── Chemins ───────────────────────────────────────────────────────────────────

ROOT_DIR   = Path(__file__).parent.parent          # poc_new/
SAMPLE_DIR = ROOT_DIR / "Sample"
IMG_PARTS  = [SAMPLE_DIR / "pt1", SAMPLE_DIR / "pt2", SAMPLE_DIR / "pt3"]
ALTO_DIR   = SAMPLE_DIR / "sample_iiif"
OUTPUT_DIR = Path(__file__).parent / "resultats_mistral"

NS_ALTO = "http://www.loc.gov/standards/alto/ns-v3#"

# Modèle Mistral vision (pixtral comprend le vieux français manuscrit/imprimé)
MISTRAL_MODEL = "pixtral-12b-2409"
MISTRAL_URL   = "https://api.mistral.ai/v1/chat/completions"

# Côté minimal d'un bloc, en pixels. En deçà, la région ne porte pas de texte
# lisible : il s'agit de filets, de vignettes ou de fragments que la
# reconnaissance de mise en page a isolés à tort. Les soumettre consommerait un
# appel pour une transcription vide.
MIN_BLOCK_PX = 20

# Nombre d'appels menés en parallèle. Cinq tient sous les limites de débit de
# l'interface. Une valeur plus élevée provoque des réponses 429, que le script
# sait rejouer mais qui annulent le gain.
MAX_WORKERS = 5

# Nombre de tentatives en cas d'erreur de débit ou de réseau. Trois suffisent
# aux incidents passagers ; au-delà, la cause est durable et insister ne sert
# à rien.
MAX_RETRIES = 3

# Délai entre deux tentatives, en secondes. Il laisse au compteur de débit le
# temps de se remettre à zéro.
RETRY_DELAY = 2.0


# ── Utilitaires image ─────────────────────────────────────────────────────────

def find_image_dir(fascicule_id: str) -> Path | None:
    """Cherche le dossier master/ du fascicule dans pt1/pt2/pt3."""
    for part in IMG_PARTS:
        candidate = part / fascicule_id / "master"
        if candidate.exists():
            return candidate
    return None


def page_to_image(img_dir: Path, page_num: int) -> Path | None:
    """T0000001.jp2 pour page 1, etc."""
    candidate = img_dir / f"T{page_num:07d}.jp2"
    return candidate if candidate.exists() else None


def crop_block(img: Image.Image, x: int, y: int, w: int, h: int,
               padding: int = 4) -> Image.Image:
    """Croppe un bloc avec un léger padding, converti en JPEG pour l'API."""
    left   = max(0, x - padding)
    top    = max(0, y - padding)
    right  = min(img.width,  x + w + padding)
    bottom = min(img.height, y + h + padding)
    return img.crop((left, top, right, bottom))


MAX_B64_BYTES = 800_000  # ~600 KB base64 → taille raisonnable pour l'API

def img_to_b64(img: Image.Image) -> str:
    """Convertit une image PIL en JPEG base64, redimensionne si trop lourde."""
    img = img.convert("RGB")
    for quality in (70, 50, 30):
        # Redimensionner si trop grand
        max_dim = 2000
        if img.width > max_dim or img.height > max_dim:
            img.thumbnail((max_dim, max_dim), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        b64 = base64.b64encode(buf.getvalue()).decode()
        if len(b64) <= MAX_B64_BYTES:
            return b64
        max_dim = int(max_dim * 0.7)
    return b64  # renvoie quand même, même si trop lourd


# ── Appel Mistral vision ──────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "Tu es un expert en transcription de journaux français anciens (XIXe-XXe siècle). "
    "On te donne une image d'un bloc de texte extrait d'une page de journal. "
    "Transcris fidèlement le texte visible, en conservant la ponctuation et les majuscules. "
    "Réponds UNIQUEMENT avec le texte transcrit, sans explication ni commentaire."
)


# Compteur global de tokens (mis à jour après chaque appel)
_tokens = {"prompt": 0, "completion": 0, "total": 0}

# Prix pixtral-12b en $ par million de tokens (mai 2025)
PRICE_INPUT_PER_M  = 0.15
PRICE_OUTPUT_PER_M = 0.15


def tokens_cost_str() -> str:
    """Retourne une ligne de log avec la consommation et le coût estimé."""
    total = _tokens["total"]
    cost  = (_tokens["prompt"] * PRICE_INPUT_PER_M
             + _tokens["completion"] * PRICE_OUTPUT_PER_M) / 1_000_000
    return (f"🔢 Tokens cumulés : {total:,}  "
            f"(prompt {_tokens['prompt']:,} + completion {_tokens['completion']:,})  "
            f"≈ ${cost:.4f}")


_tokens_lock = threading.Lock()


def transcribe_crop(b64_img: str, api_key: str) -> str:
    """Envoie un crop base64 à Mistral avec retry sur 429/réseau."""
    payload = {
        "model": MISTRAL_MODEL,
        "messages": [
            {"role": "user", "content": [
                {"type": "text",      "text": SYSTEM_PROMPT},
                {"type": "image_url", "image_url": f"data:image/jpeg;base64,{b64_img}"},
            ]},
        ],
        "max_tokens": 512,
        "temperature": 0.0,
    }
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(
                MISTRAL_URL,
                headers={"Authorization": f"Bearer {api_key}",
                         "Content-Type": "application/json"},
                json=payload,
                timeout=30,
            )
            if resp.status_code == 429:
                wait = RETRY_DELAY * (attempt + 1)
                tqdm.write(f"    ⏳ 429 rate limit : attente {wait:.0f}s")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            usage = data.get("usage", {})
            with _tokens_lock:
                _tokens["prompt"]     += usage.get("prompt_tokens",     0)
                _tokens["completion"] += usage.get("completion_tokens", 0)
                _tokens["total"]      += usage.get("total_tokens",      0)
            return data["choices"][0]["message"]["content"].strip()
        except requests.RequestException as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                raise
    raise RuntimeError("Échec après plusieurs tentatives")


# ── Parseur ALTO ──────────────────────────────────────────────────────────────

TITLE_LABELS = {"title", "subtitle", "heading"}


def parse_alto(alto_path: Path) -> list[dict]:
    """Retourne la liste des blocs ALTO avec coords et texte original."""
    tree = ET.parse(alto_path)
    root = tree.getroot()

    # Construire le mapping tag_id → label (StructureTag)
    tag_labels = {}
    for st in root.iter(f"{{{NS_ALTO}}}StructureTag"):
        tag_labels[st.get("ID", "")] = (st.get("LABEL") or "").lower()

    blocks = []
    for tb in root.iter(f"{{{NS_ALTO}}}TextBlock"):
        bid = tb.get("ID", "")
        x = int(tb.get("HPOS",   0))
        y = int(tb.get("VPOS",   0))
        w = int(tb.get("WIDTH",  0))
        h = int(tb.get("HEIGHT", 0))
        if w < MIN_BLOCK_PX or h < MIN_BLOCK_PX:
            continue

        # Détection titre via TAGREFS (comme le notebook Pero)
        label = ""
        for tr in (tb.get("TAGREFS") or "").split():
            if tr in tag_labels:
                label = tag_labels[tr]
                break
        is_title = label in TITLE_LABELS

        lines = []
        for tl in tb.findall(f"{{{NS_ALTO}}}TextLine"):
            words = " ".join(s.get("CONTENT", "")
                             for s in tl.findall(f"{{{NS_ALTO}}}String"))
            if words.strip():
                lines.append(words.strip())
        blocks.append({
            "id": bid, "x": x, "y": y, "w": w, "h": h,
            "texte_original": " ".join(lines).strip(),
            "is_title": is_title,
        })
    return blocks


# ── Écriture ALTO ────────────────────────────────────────────────────────────

def write_alto(alto_path: Path, results: dict, out_alto: Path):
    """
    Réécrit un fichier ALTO en remplaçant le texte de chaque TextBlock
    par la transcription Mistral. Structure XML conservée à l'identique.
    """
    tree = ET.parse(alto_path)
    root = tree.getroot()

    for tb in root.iter(f"{{{NS_ALTO}}}TextBlock"):
        bid = tb.get("ID", "")
        if bid not in results:
            continue
        texte = results[bid]["texte_mistral"]
        if not texte.strip():
            continue

        # Récupérer les coords du bloc pour reconstruire les lignes
        bx = int(tb.get("HPOS",   0))
        by = int(tb.get("VPOS",   0))
        bw = int(tb.get("WIDTH",  0))
        bh = int(tb.get("HEIGHT", 0))

        # Vider le contenu existant du TextBlock
        for child in list(tb):
            tb.remove(child)

        # Recréer les TextLines et Strings à partir des lignes du texte Mistral
        lines = [l for l in texte.split("\n") if l.strip()]
        if not lines:
            lines = [texte]
        lh = max(1, bh // len(lines))

        for li, line_text in enumerate(lines):
            tl = ET.SubElement(tb, f"{{{NS_ALTO}}}TextLine")
            tl.set("ID",     f"{bid}_L{li}")
            tl.set("HPOS",   str(bx))
            tl.set("VPOS",   str(by + li * lh))
            tl.set("WIDTH",  str(bw))
            tl.set("HEIGHT", str(lh))

            words = line_text.split()
            ww = max(1, bw // max(len(words), 1))
            for wi, word in enumerate(words):
                s = ET.SubElement(tl, f"{{{NS_ALTO}}}String")
                s.set("ID",      f"{bid}_L{li}_S{wi}")
                s.set("HPOS",    str(bx + wi * ww))
                s.set("VPOS",    str(by + li * lh))
                s.set("WIDTH",   str(ww))
                s.set("HEIGHT",  str(lh))
                s.set("CONTENT", word)
                if wi < len(words) - 1:
                    ET.SubElement(tl, f"{{{NS_ALTO}}}SP")

    out_alto.parent.mkdir(parents=True, exist_ok=True)
    tree.write(str(out_alto), encoding="utf-8", xml_declaration=True)


# ── Pipeline ──────────────────────────────────────────────────────────────────

def process_page(fascicule_id: str, alto_path: Path, img_dir: Path,
                 out_dir: Path, api_key: str, dry_run: bool,
                 pbar_global: tqdm | None = None,
                 json_filename: str | None = None) -> dict:
    """Re-OCRise tous les blocs d'UNE page ALTO : crop chaque bloc depuis l'image
    JP2/JPG, l'envoie à Mistral (pixtral-12b), et sauvegarde {bloc_id: {texte_mistral,
    texte_original}} dans un JSON de log. Reprend un JSON existant si déjà présent
    (un bloc déjà traité n'est pas renvoyé à l'API).
    """

    pnum = int(re.search(r"X0*(\d+)", alto_path.stem).group(1))
    fname = json_filename or f"X{pnum:07d}.json"
    out_file = out_dir / fname

    try:
        blocks = parse_alto(alto_path)
    except ET.ParseError as e:
        tqdm.write(f"    p{pnum} ⚠ XML invalide : {e}")
        return {"status": "skip_xml_error", "page": pnum}
    if not blocks:
        return {"status": "skip_empty", "page": pnum}

    img_path = page_to_image(img_dir, pnum)
    if not img_path:
        return {"status": "skip_no_image", "page": pnum}

    # ── Reprise bloc par bloc ─────────────────────────────────────────────────
    # Le JSON est écrit au fil de l'eau ; les blocs déjà faits sont sautés.
    results = {}
    if out_file.exists():
        try:
            results = json.loads(out_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            results = {}

    # Les blocs en erreur sont relancés au prochain run
    already_done = sum(1 for v in results.values() if "erreur" not in v)
    remaining    = [b for b in blocks
                    if b["id"] not in results or "erreur" in results[b["id"]]]

    if not remaining:
        if pbar_global:
            pbar_global.update(len(blocks))
        return {"status": "cache", "page": pnum, "n_blocs": len(blocks)}

    if dry_run:
        tqdm.write(f"    [dry] p{pnum} : {len(blocks)} blocs "
                   f"({already_done} déjà faits), image: {img_path.name}")
        return {"status": "dry_run", "page": pnum, "n_blocs": len(blocks)}

    n_errors = sum(1 for v in results.values() if "erreur" in v)
    if already_done or n_errors:
        msg = f"    ↩ p{pnum} reprise : {already_done}/{len(blocks)} blocs déjà faits"
        if n_errors:
            msg += f" ({n_errors} erreurs à relancer)"
        tqdm.write(msg)

    img = Image.open(img_path)
    # Pré-calculer les crops (lecture image thread-unsafe → fait en avance)
    crops_b64 = {}
    for b in remaining:
        if not b.get("is_title"):
            crop = crop_block(img, b["x"], b["y"], b["w"], b["h"])
            crops_b64[b["id"]] = img_to_b64(crop)

    save_lock = threading.Lock()

    def process_block(b: dict) -> tuple[str, dict]:
        """Traite un bloc : titre → copie, texte → appel API. Thread-safe."""
        if b.get("is_title"):
            return b["id"], {
                "texte_mistral":  b["texte_original"],
                "texte_original": b["texte_original"],
                "is_title": True,
            }
        try:
            texte = transcribe_crop(crops_b64[b["id"]], api_key)
            return b["id"], {
                "texte_mistral":  texte,
                "texte_original": b["texte_original"],
            }
        except Exception as e:
            tqdm.write(f"    ⚠ {b['id']} : {e}")
            return b["id"], {
                "texte_mistral":  b["texte_original"],
                "texte_original": b["texte_original"],
                "erreur": str(e),
            }

    with tqdm(total=len(blocks), initial=already_done,
              desc=f"  p{pnum}", unit="bloc", leave=False,
              bar_format="{desc} {bar} {n}/{total} blocs [{elapsed}<{remaining}]") as pbar_page:

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(process_block, b): b for b in remaining}
            for future in as_completed(futures):
                bid, entry = future.result()
                with save_lock:
                    results[bid] = entry
                    out_file.write_text(
                        json.dumps(results, ensure_ascii=False, indent=2),
                        encoding="utf-8")
                pbar_page.update(1)
                if pbar_global:
                    pbar_global.update(1)

    n_titles = sum(1 for v in results.values() if v.get("is_title"))
    n_ok = sum(1 for v in results.values() if "erreur" not in v and not v.get("is_title"))
    tqdm.write(f"    p{pnum} ✓  {n_ok}/{len(blocks)} blocs transcrits "
               f"({n_titles} titres conservés tels quels)")
    tqdm.write(f"    {tokens_cost_str()}")
    return {"status": "ok", "page": pnum, "n_blocs": len(blocks)}


def count_total_blocs(fascicules: list[str], pages: list[int] | None) -> int:
    """Compte le nombre total de blocs à traiter (pour la barre globale)."""
    total = 0
    for fid in fascicules:
        alto_dir = ALTO_DIR / fid / "ocr"
        if not alto_dir.exists():
            continue
        alto_files = sorted(alto_dir.glob("X*.xml"))
        if pages:
            alto_files = [f for f in alto_files
                          if int(re.search(r"X0*(\d+)", f.stem).group(1)) in pages]
        for f in alto_files:
            try:
                total += len(parse_alto(f))
            except ET.ParseError:
                pass
    return total


def process_fascicule(fascicule_id: str, pages: list[int] | None,
                      out_base: Path, api_key: str, dry_run: bool,
                      pbar_global: tqdm | None = None) -> list[dict]:
    """Re-OCRise toutes les pages d'un fascicule (ou seulement `pages` si fourni) :
    appelle process_page() pour chacune, agrège les logs, et retourne le résumé
    par page (statut, nb de blocs traités) utilisé pour le rapport global.
    """

    alto_dir = ALTO_DIR / fascicule_id / "ocr"
    img_dir  = find_image_dir(fascicule_id)

    if not alto_dir.exists():
        tqdm.write(f"  ⚠ dossier ALTO absent : {alto_dir}"); return []
    if not img_dir:
        tqdm.write(f"  ⚠ images introuvables pour {fascicule_id}"); return []

    import shutil
    src_fasc    = ALTO_DIR / fascicule_id
    out_dir     = out_base / f"{fascicule_id}_reocr"
    out_ocr_dir = out_dir / "ocr"
    out_log_dir = out_dir / "logs"
    out_ocr_dir.mkdir(parents=True, exist_ok=True)
    out_log_dir.mkdir(parents=True, exist_ok=True)

    # Copier manifest.xml et toc/ : identique au pipeline Pero
    if (src_fasc / "manifest.xml").exists():
        shutil.copy2(src_fasc / "manifest.xml", out_dir / "manifest.xml")
    if (src_fasc / "toc").exists():
        shutil.copytree(src_fasc / "toc", out_dir / "toc", dirs_exist_ok=True)

    alto_files = sorted(alto_dir.glob("X*.xml"))
    if pages:
        alto_files = [f for f in alto_files
                      if int(re.search(r"X0*(\d+)", f.stem).group(1)) in pages]

    tqdm.write(f"\n══ {fascicule_id} : {len(alto_files)} page(s) ══")

    results = []
    for alto_path in alto_files:
        pnum = int(re.search(r"X0*(\d+)", alto_path.stem).group(1))
        # log JSON dans logs/, ALTO dans ocr/ : comme le pipeline Pero
        json_path = out_log_dir / f"X{pnum:07d}.json"
        out_alto  = out_ocr_dir / alto_path.name
        try:
            r = process_page(fascicule_id, alto_path, img_dir,
                             out_log_dir, api_key, dry_run, pbar_global,
                             json_filename=f"X{pnum:07d}.json")
            # Écrire l'ALTO re-OCRisé si la page a été traitée (ou était en cache)
            if r.get("status") in ("ok", "cache") and json_path.exists():
                resu = json.loads(json_path.read_text(encoding="utf-8"))
                write_alto(alto_path, resu, out_alto)
            results.append({**r, "fascicule": fascicule_id})
        except Exception as e:
            tqdm.write(f"    p{pnum} ❌  {e}")

    return results


# ── Entrée principale ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fascicule", default=None, help="Un seul fascicule (ex: 4109000)")
    parser.add_argument("--page",      type=int, nargs="+", default=None)
    parser.add_argument("--out",       default=str(OUTPUT_DIR))
    parser.add_argument("--workers",   type=int, default=MAX_WORKERS,
                        help="Appels API parallèles (défaut 5, augmenter si pas de 429)")
    parser.add_argument("--dry-run",   action="store_true")
    args = parser.parse_args()

    MAX_WORKERS = args.workers
    out_base = Path(args.out)
    out_base.mkdir(parents=True, exist_ok=True)

    if args.fascicule:
        fascicules = [args.fascicule]
    else:
        fascicules = sorted(d.name for d in ALTO_DIR.iterdir() if d.is_dir())

    print(f"📂 {len(fascicules)} fascicule(s) : sortie : {out_base}")
    print(f"🔑 Clé Mistral chargée depuis {ENV_FILE}")

    if not args.dry_run:
        print("⏳ Comptage des blocs pour la barre de progression…")
        total_blocs = count_total_blocs(fascicules, args.page)
        print(f"   {total_blocs} blocs au total\n")
    else:
        total_blocs = 0

    all_results = []
    with tqdm(total=total_blocs, desc="TOTAL", unit="bloc", position=0,
              bar_format="{desc} {bar} {n}/{total} blocs [{elapsed}<{remaining}, {rate_fmt}]",
              disable=args.dry_run) as pbar_global:

        for fid in fascicules:
            res = process_fascicule(fid, args.page, out_base,
                                    MISTRAL_API_KEY, args.dry_run, pbar_global)
            all_results.extend(res)

    ok = [r for r in all_results if r.get("status") == "ok"]
    cached = [r for r in all_results if r.get("status") == "cache"]
    print(f"\n{'='*50}")
    print(f"✅ {len(ok)} page(s) traitées  ({sum(r.get('n_blocs',0) for r in ok)} blocs)")
    if cached:
        print(f"   {len(cached)} page(s) déjà en cache (sautées)")
    print(f"   {tokens_cost_str()}")

    rapport = out_base / "rapport.json"
    rapport.write_text(json.dumps({"pages": all_results}, ensure_ascii=False, indent=2))
    print(f"   Rapport : {rapport}")
