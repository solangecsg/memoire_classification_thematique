"""
impresso_ocrqa.py — OCR Quality Assessment (impresso) sur 3 sources

Pour les 10 premiers fascicules disponibles, calcule le score OCRQA
(ratio mots reconnus / total) sur :
  - BnF       : ALTO original  (corpus/original/{fid}/ocr/)
  - Pero      : ALTO re-OCR    (corpus/reocr_pero/{fid}_reocr/ocr/)
  - Mistral   : ALTO re-OCR    (corpus/reocr_mistral/{fid}_reocr/ocr/)

Les ALTO produits par Pero ne sont pas versés dans ce dépôt : le moteur a été
écarté après cette mesure et ses sorties n'ont pas été conservées. Le script
signale la source manquante et poursuit sur les deux autres, chaque score étant
calculé indépendamment. Les valeurs obtenues à trois sources restent lisibles
dans resultats_ocrqa/, qui est versé.

Sortie :
  - resultats_ocrqa/ocrqa_results.json   (scores par bloc)
  - resultats_ocrqa/ocrqa_summary.csv    (scores moyens par fascicule × source)
  - resultats_ocrqa/ocrqa_report.html    (visualisation)

Usage :
    python3 impresso_ocrqa.py
    python3 impresso_ocrqa.py --n 5        # 5 premiers fascicules seulement
"""

import argparse
import csv
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict

NS_ALTO     = "http://www.loc.gov/standards/alto/ns-v3#"
TITLE_LABELS = {"title", "subtitle", "heading"}

ICI         = Path(__file__).resolve().parent
OUTPUT_DIR  = ICI / "resultats_ocrqa"
OUTPUT_DIR.mkdir(exist_ok=True)

SOURCES = ["BnF", "Pero", "Mistral"]

# Chaque océrisation est cherchée à deux endroits : sous corpus/, comme le dépôt
# la range, et là où sa chaîne de production l'a écrite dans le dossier de
# travail. Le second membre de chaque couple donne le nom du dossier de
# fascicule, les deux ré-océrisations ayant suffixé le leur. La recherche remonte
# l'arborescence, les deux dispositions ne plaçant pas le corpus à la même
# profondeur.
DISPOSITIONS = {
    "BnF":     [("corpus/original", "{fid}"),
                ("sample_iiif", "{fid}")],
    "Pero":    [("corpus/reocr_pero", "{fid}_reocr"),
                ("ocr/re_ocr/results/re_ocr_results_extract_01072026", "{fid}_reocr")],
    "Mistral": [("corpus/reocr_mistral", "{fid}_reocr"),
                ("resultats_mistral", "{fid}_reocr")],
}

EMPLACEMENTS: dict[str, tuple[Path, str]] = {}


def emplacements() -> dict[str, tuple[Path, str]]:
    """Localise les océrisations présentes, dans l'une ou l'autre disposition.

    Une source absente est omise plutôt que fatale : les scores se calculent
    source par source et restent comparables sur celles qui subsistent. Seule
    l'océrisation d'origine est indispensable, la liste des fascicules à traiter
    en étant tirée.
    """
    trouves = {}
    for source in SOURCES:
        for rel, motif in DISPOSITIONS[source]:
            base = next((b / rel for b in [ICI, *ICI.parents] if (b / rel).is_dir()), None)
            if base is not None:
                trouves[source] = (base, motif)
                break
    if "BnF" not in trouves:
        raise SystemExit(
            "océrisation d'origine introuvable, cherchée depuis "
            f"{ICI} en remontant l'arborescence sous "
            + " ou ".join(rel for rel, _ in DISPOSITIONS["BnF"]) + ".")
    return trouves


def dossier(source: str, fid: str) -> Path:
    """Rend le dossier ALTO d'un fascicule, pour une source donnée."""
    base, motif = EMPLACEMENTS[source]
    return base / motif.format(fid=fid) / "ocr"

# Extraction texte ALTO

def extract_text_from_alto(alto_path: Path) -> str:
    """Texte complet d'un fichier ALTO (tous les blocs, hors titres/sous-titres),
    mots séparés par des espaces — sert de base au calcul du score OCRQA
    (impresso_pipelines) pour une page donnée.
    """
    tree = ET.parse(alto_path)
    root = tree.getroot()
    tag_labels = {st.get("ID", ""): (st.get("LABEL") or "").lower()
                  for st in root.iter(f"{{{NS_ALTO}}}StructureTag")}
    words = []
    for tb in root.iter(f"{{{NS_ALTO}}}TextBlock"):
        label = ""
        for tr in (tb.get("TAGREFS") or "").split():
            if tr in tag_labels:
                label = tag_labels[tr]; break
        if label in TITLE_LABELS:
            continue
        for s in tb.iter(f"{{{NS_ALTO}}}String"):
            w = s.get("CONTENT", "").strip()
            if w:
                words.append(w)
    return " ".join(words)

def load_fascicule_texts(fid: str, source: str) -> dict[str, str]:
    """Retourne {page_stem: texte} pour un fascicule."""
    alto_dir = dossier(source, fid)
    if not alto_dir.is_dir():
        return {}
    texts = {}
    for alto in sorted(alto_dir.glob("X*.xml")):
        text = extract_text_from_alto(alto)
        if text.strip():
            texts[alto.stem] = text
    return texts

# Main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=10, help="Nb de fascicules à traiter")
    args = parser.parse_args()

    EMPLACEMENTS = emplacements()
    PRESENTES = [s for s in SOURCES if s in EMPLACEMENTS]
    absentes = [s for s in SOURCES if s not in EMPLACEMENTS]
    if absentes:
        print("source absente, écartée du calcul : " + ", ".join(absentes) + "\n")

    # Initialisation pipeline (télécharge les Bloom filters au premier lancement)
    print("Chargement pipeline OCRQA impresso…")
    from impresso_pipelines.ocrqa import OCRQAPipeline
    pipeline = OCRQAPipeline()
    print("  Pipeline prêt.\n")

    # Liste des fascicules disponibles (intersection des 3 sources)
    base, motif = EMPLACEMENTS["BnF"]
    bnf_fascicules = sorted(
        d.name for d in base.iterdir()
        if d.is_dir() and (d / "ocr").is_dir()
    )
    fascicules = bnf_fascicules[:args.n]
    print(f"Fascicules : {fascicules}\n")

    # Résultats : {fid: {source: {page: {score, lang, n_words}}}}
    results = defaultdict(lambda: defaultdict(dict))

    for fid in fascicules:
        print(f"══ {fid} ══")
        for source in PRESENTES:
            texts = load_fascicule_texts(fid, source)
            if not texts:
                print(f"  {source:8s} : aucune page ALTO")
                continue
            scores = []
            for page, text in texts.items():
                if not text.strip():
                    continue
                try:
                    r = pipeline(text)
                    score = r.get("score", 0.0)
                    lang  = r.get("language", "?")
                    n_words = len(text.split())
                    results[fid][source][page] = {
                        "score": score, "language": lang, "n_words": n_words
                    }
                    scores.append(score)
                except Exception as e:
                    print(f"    ✗ {page}: {e}")
            mean = sum(scores) / len(scores) if scores else 0
            print(f"  {source:8s} : {len(scores):3d} pages  score moyen = {mean:.3f}")

    # Export JSON
    json_out = OUTPUT_DIR / "ocrqa_results.json"
    json_out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n→ {json_out}")

    # Export CSV résumé
    csv_out = OUTPUT_DIR / "ocrqa_summary.csv"
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["fascicule", "source", "n_pages", "score_moyen", "score_min", "score_max"])
        for fid in fascicules:
            for source in PRESENTES:
                pages = results[fid].get(source, {})
                if not pages:
                    w.writerow([fid, source, 0, "", "", ""])
                    continue
                scores = [p["score"] for p in pages.values()]
                w.writerow([fid, source, len(scores),
                            round(sum(scores)/len(scores), 3),
                            round(min(scores), 3),
                            round(max(scores), 3)])
    print(f"→ {csv_out}")

    # Export HTML
    COLORS = {"BnF": "#3b82f6", "Pero": "#f97316", "Mistral": "#22c55e"}

    # Scores moyens par source (toutes pages confondues)
    global_scores = {}
    for source in PRESENTES:
        all_s = [p["score"] for fid in fascicules
                 for p in results[fid].get(source, {}).values()]
        global_scores[source] = round(sum(all_s)/len(all_s), 3) if all_s else 0

    # Tableau HTML
    rows = ""
    for fid in fascicules:
        cells = f"<td style='color:#94a3b8'>{fid}</td>"
        for source in PRESENTES:
            pages = results[fid].get(source, {})
            if not pages:
                cells += "<td style='color:#475569'>—</td>"
                continue
            scores = [p["score"] for p in pages.values()]
            mean = sum(scores) / len(scores)
            # couleur rouge→vert selon score
            t  = mean
            if t < 0.5:
                r2, g2, b2 = 210, int(60+150*t*2), int(60+150*t*2)
            else:
                r2, g2, b2 = int(210-150*(t-0.5)*2), 210, int(210-150*(t-0.5)*2)
            bg = f"rgb({r2},{g2},{b2})"
            lum = 0.299*r2 + 0.587*g2 + 0.114*b2
            fg = "#111" if lum > 130 else "#fff"
            cells += f"<td style='background:{bg};color:{fg};font-weight:700'>{mean:.3f}</td>"
        rows += f"<tr>{cells}</tr>"

    # Ligne totaux
    cells_tot = "<td style='color:#93c5fd;font-weight:700'>Moyenne globale</td>"
    for source in PRESENTES:
        s = global_scores[source]
        c = COLORS[source]
        cells_tot += f"<td style='color:{c};font-weight:700'>{s:.3f}</td>"
    rows += f"<tr style='border-top:2px solid #334155'>{cells_tot}</tr>"

    # Barres globales
    bars = ""
    for source in PRESENTES:
        s = global_scores[source]
        c = COLORS[source]
        w = int(s * 300)
        bars += f"""<div style="display:flex;align-items:center;gap:10px;margin-bottom:10px">
          <span style="width:80px;color:{c};font-weight:700;font-size:.9rem">{source}</span>
          <div style="width:{w}px;height:20px;background:{c};border-radius:4px"></div>
          <span style="color:#e2e8f0;font-size:.9rem">{s:.3f}</span>
        </div>"""

    html = f"""<!DOCTYPE html>
<html lang="fr"><head><meta charset="utf-8">
<title>OCRQA impresso</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Segoe UI',sans-serif;background:#0f172a;color:#e2e8f0;padding:32px}}
h1{{font-size:1.1rem;color:#93c5fd;margin-bottom:4px}}
.sub{{color:#475569;font-size:.8rem;margin-bottom:28px}}
.layout{{display:flex;gap:24px;flex-wrap:wrap;align-items:flex-start}}
.card{{background:#1e293b;border:1px solid #2d3f55;border-radius:10px;padding:20px 24px}}
.card h2{{font-size:.85rem;color:#93c5fd;margin-bottom:14px}}
table{{border-collapse:collapse}}
th{{background:#0f172a;color:#64748b;padding:10px 18px;font-size:.8rem;text-align:center}}
th:first-child{{text-align:left}}
td{{padding:10px 18px;text-align:center;border-top:1px solid #1e3a5f;font-size:.85rem}}
td:first-child{{text-align:left}}
</style></head><body>
<h1>OCR Quality Assessment — impresso pipeline</h1>
<div class="sub">{args.n} premiers fascicules · score = ratio mots reconnus (Bloom filter) · langue détectée automatiquement</div>
<div class="layout">
<div class="card">
<h2>Score moyen global (toutes pages)</h2>
{bars}
<p style="font-size:.75rem;color:#475569;margin-top:12px">0 = aucun mot reconnu · 1 = tous les mots reconnus</p>
</div>
<div class="card">
<h2>Score moyen par fascicule</h2>
<table>
<thead><tr>
  <th>Fascicule</th>
  {"".join(f'<th style="color:{COLORS[s]}">{s}</th>' for s in PRESENTES)}
</tr></thead>
<tbody>{rows}</tbody>
</table>
</div>
</div>
</body></html>"""

    html_out = OUTPUT_DIR / "ocrqa_report.html"
    html_out.write_text(html, encoding="utf-8")
    print(f"→ {html_out}")
    print(f"\n✅ Terminé — {len(fascicules)} fascicules × {len(PRESENTES)} source(s)")
