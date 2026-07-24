"""
impresso_ocrqa.py — OCR Quality Assessment (impresso) sur 3 sources

Pour les 10 premiers fascicules disponibles, calcule le score OCRQA
(ratio mots reconnus / total) sur :
  - BnF       : ALTO original  (Sample/sample_iiif/{fid}/ocr/)
  - Pero      : ALTO re-OCR    (re_ocr/results/re_ocr_results_extract_01072026/{fid}_reocr/ocr/)
  - Mistral   : ALTO re-OCR    (reocr_mistral/resultats_mistral/{fid}_reocr/ocr/)

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

ROOT        = Path(__file__).parent.parent
OUTPUT_DIR  = Path(__file__).parent / "resultats_ocrqa"
OUTPUT_DIR.mkdir(exist_ok=True)

ALTO_SOURCES = {
    "BnF":     lambda fid: ROOT / "Sample" / "sample_iiif" / fid / "ocr",
    "Pero":    lambda fid: ROOT / "re_ocr" / "results" / "re_ocr_results_extract_01072026" / f"{fid}_reocr" / "ocr",
    "Mistral": lambda fid: ROOT / "reocr_mistral" / "resultats_mistral" / f"{fid}_reocr" / "ocr",
}

# ── Extraction texte ALTO ─────────────────────────────────────────────────────

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
    alto_dir = ALTO_SOURCES[source](fid)
    if not alto_dir.exists():
        return {}
    texts = {}
    for alto in sorted(alto_dir.glob("X*.xml")):
        text = extract_text_from_alto(alto)
        if text.strip():
            texts[alto.stem] = text
    return texts

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10, help="Nb de fascicules à traiter")
    args = parser.parse_args()

    # Initialisation pipeline (télécharge les Bloom filters au premier lancement)
    print("Chargement pipeline OCRQA impresso…")
    from impresso_pipelines.ocrqa import OCRQAPipeline
    pipeline = OCRQAPipeline()
    print("  Pipeline prêt.\n")

    # Liste des fascicules disponibles (intersection des 3 sources)
    bnf_fascicules = sorted(
        d.name for d in (ROOT / "Sample" / "sample_iiif").iterdir()
        if d.is_dir() and (d / "ocr").exists()
    )
    fascicules = bnf_fascicules[:args.n]
    print(f"Fascicules : {fascicules}\n")

    # Résultats : {fid: {source: {page: {score, lang, n_words}}}}
    results = defaultdict(lambda: defaultdict(dict))

    for fid in fascicules:
        print(f"══ {fid} ══")
        for source in ["BnF", "Pero", "Mistral"]:
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

    # ── Export JSON ───────────────────────────────────────────────────────────
    json_out = OUTPUT_DIR / "ocrqa_results.json"
    json_out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n→ {json_out}")

    # ── Export CSV résumé ─────────────────────────────────────────────────────
    csv_out = OUTPUT_DIR / "ocrqa_summary.csv"
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["fascicule", "source", "n_pages", "score_moyen", "score_min", "score_max"])
        for fid in fascicules:
            for source in ["BnF", "Pero", "Mistral"]:
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

    # ── Export HTML ───────────────────────────────────────────────────────────
    COLORS = {"BnF": "#3b82f6", "Pero": "#f97316", "Mistral": "#22c55e"}

    # Scores moyens par source (toutes pages confondues)
    global_scores = {}
    for source in ["BnF", "Pero", "Mistral"]:
        all_s = [p["score"] for fid in fascicules
                 for p in results[fid].get(source, {}).values()]
        global_scores[source] = round(sum(all_s)/len(all_s), 3) if all_s else 0

    # Tableau HTML
    rows = ""
    for fid in fascicules:
        cells = f"<td style='color:#94a3b8'>{fid}</td>"
        for source in ["BnF", "Pero", "Mistral"]:
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
    for source in ["BnF", "Pero", "Mistral"]:
        s = global_scores[source]
        c = COLORS[source]
        cells_tot += f"<td style='color:{c};font-weight:700'>{s:.3f}</td>"
    rows += f"<tr style='border-top:2px solid #334155'>{cells_tot}</tr>"

    # Barres globales
    bars = ""
    for source in ["BnF", "Pero", "Mistral"]:
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
  <th style="color:#3b82f6">BnF</th>
  <th style="color:#f97316">Pero</th>
  <th style="color:#22c55e">Mistral</th>
</tr></thead>
<tbody>{rows}</tbody>
</table>
</div>
</div>
</body></html>"""

    html_out = OUTPUT_DIR / "ocrqa_report.html"
    html_out.write_text(html, encoding="utf-8")
    print(f"→ {html_out}")
    print(f"\n✅ Terminé — {len(fascicules)} fascicules × 3 sources")
