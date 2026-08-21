"""
stats_pages.py : mesures de vocabulaire par page et comparaison des trois sources

CE QUE FAIT CE SCRIPT

Compare le vocabulaire produit par trois océrisations du même corpus, page par
page, et en dresse un diagramme de recouvrement.

Les trois sources sont l'océrisation d'origine livrée avec les fascicules, celle
que produit le moteur Pero, et celle que produit un modèle multimodal. La
comparaison sert à établir ce que la ré-océrisation change réellement.

CE QUI EST MESURÉ, PAR PAGE ET PAR SOURCE

  formes distinctes   nombre de chaînes différentes. La quantité est
                      l'indicateur principal : une océrisation défaillante
                      produit des formes qui n'existent pas, chacune attestée
                      une seule fois, et gonfle donc ce nombre sans porter
                      d'information.
  mots                nombre total de mots
  caractères          nombre total de caractères
  articles            nombre de divisions ARTICLE couvrant la page, lues dans la
                      carte logique du METS

LE DIAGRAMME DE RECOUVREMENT

Il montre, pour l'ensemble du corpus, les formes que les trois sources ont en
commun et celles que chacune produit seule. Les formes attestées par une seule
source sont celles qui posent question : le mémoire établit que 27 pour cent des
formes du corpus ne sont attestées que par l'océrisation d'origine, proportion
qui justifie la ré-océrisation.

Les blocs de titraille sont écartés du dénombrement. Ils portent des mots en
capitales et en corps différent, dont la reconnaissance obéit à d'autres
contraintes, et les compter mêlerait deux régimes.

ENTRÉES

  corpus/original/{fascicule}/ocr/             ALTO d'origine
  corpus/reocr_pero/{fascicule}_reocr/ocr/     ALTO produits par Pero
  corpus/reocr_mistral/{fascicule}_reocr/ocr/  ALTO produits par le modèle
                                               multimodal
  corpus/original/{fascicule}/toc/             carte logique, d'où vient le
                                               compte d'articles

Les ALTO produits par Pero ne sont pas versés dans ce dépôt. Le moteur a servi
d'étape vers le modèle multimodal, et le mémoire rapporte cette comparaison pour
la démarche qu'elle documente plutôt que comme une expérience à rejouer. Le
script s'arrête sur un message qui le dit ; le relevé et le diagramme qu'il a
produits restent versés dans resultats_stats/.

SORTIES

  resultats_stats/vocab_venn.html        diagramme de recouvrement
  resultats_stats/stats_par_page.csv     détail par fascicule, page et source
  resultats_stats/stats_mistral_brut.csv mots, caractères et articles par page

PAQUETS EMPLOYÉS

  argparse, csv, collections, pathlib   bibliothèque standard
  xml.etree.ElementTree                 lecture des fichiers ALTO et METS

Le diagramme est écrit en HTML plutôt que produit par une bibliothèque de tracé,
ce qui évite une dépendance pour une figure unique.

USAGE

    python3 stats_pages.py
    python3 stats_pages.py --n 5      # limiter aux cinq premiers fascicules
"""

import argparse
import csv
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

# ── Chemins ───────────────────────────────────────────────────────────────────

ICI        = Path(__file__).resolve().parent
OUTPUT_DIR = ICI / "resultats_stats"
OUTPUT_DIR.mkdir(exist_ok=True)

NS_ALTO     = "http://www.loc.gov/standards/alto/ns-v3#"
METS_NS     = "http://www.loc.gov/METS/"
XLINK_NS    = "http://www.w3.org/1999/xlink"
TITLE_LABELS = {"title", "subtitle", "heading"}

SOURCES = ["BnF", "Pero", "Mistral"]

# Chaque océrisation est cherchée à deux endroits : sous corpus/, comme le dépôt
# la range, et là où sa chaîne de production l'a écrite dans le dossier de
# travail. Le second membre de chaque couple donne le nom du dossier de
# fascicule, les deux ré-océrisations ayant suffixé le leur.
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
    """Localise les trois océrisations, dans l'une ou l'autre disposition.

    La recherche remonte l'arborescence depuis ce fichier, les deux dispositions
    ne plaçant pas le corpus à la même profondeur : sous re-ocr/ dans le dépôt,
    à côté du script dans le dossier de travail.

    Le diagramme de recouvrement compare les trois sources entre elles. Il
    perdrait son sens s'il en manquait une, la part propre à chacune des autres
    s'en trouvant gonflée d'autant. L'absence est donc signalée plutôt que
    contournée.
    """
    trouves, manquants = {}, []
    for source in SOURCES:
        for rel, motif in DISPOSITIONS[source]:
            trouve = next((b / rel for b in [ICI, *ICI.parents] if (b / rel).is_dir()), None)
            if trouve is not None:
                trouves[source] = (trouve, motif)
                break
        else:
            manquants.append(source)
    if manquants:
        raise SystemExit(
            "comparaison à trois sources non rejouable depuis ce dépôt : "
            + ", ".join(manquants) + " manque.\n"
            "Le moteur Pero a servi d'étape vers le modèle multimodal et ses "
            "sorties n'ont pas été conservées.\n"
            "Le relevé qu'elles ont donné est versé dans resultats_stats/, avec "
            "le diagramme de recouvrement.\n"
            "Sources trouvées : " + ", ".join(trouves) + ".")
    return trouves


def dossier(source: str, fid: str, sous: str = "ocr") -> Path:
    """Rend le dossier ocr/ ou toc/ d'un fascicule, pour une source donnée."""
    base, motif = EMPLACEMENTS[source]
    return base / motif.format(fid=fid) / sous

# ── Extraction tokens depuis ALTO ─────────────────────────────────────────────

def tokens_from_alto(path: Path) -> list[str]:
    """Liste des mots (tokens) d'un fichier ALTO, en excluant les blocs tagués
    comme titre/sous-titre (StructureTag LABEL) : pour ne comparer que le texte
    courant entre BnF/Pero/Mistral, pas les éléments de mise en page.
    """
    tree = ET.parse(path)
    root = tree.getroot()
    tag_labels = {st.get("ID", ""): (st.get("LABEL") or "").lower()
                  for st in root.iter(f"{{{NS_ALTO}}}StructureTag")}
    tokens = []
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
                tokens.append(w.lower())
    return tokens

# ── Articles par page depuis TOC METS ────────────────────────────────────────

def articles_per_page(fid: str) -> dict[str, int]:
    """Retourne {page_stem: nb_articles} pour un fascicule."""
    toc_folder = dossier("BnF", fid, "toc")
    if not toc_folder.exists():
        return {}
    toc_files = list(toc_folder.glob("T*.xml"))
    if not toc_files:
        return {}
    root = ET.parse(toc_files[0]).getroot()

    file_map = {}
    for fi in root.iter(f"{{{METS_NS}}}file"):
        fid_attr = fi.get("ID", "")
        loc = fi.find(f"{{{METS_NS}}}FLocat")
        if loc is not None:
            href = loc.get(f"{{{XLINK_NS}}}href", "")
            name = Path(href).stem  # X0000001
            if name:
                file_map[fid_attr] = name

    per_page = Counter()
    for div in root.iter(f"{{{METS_NS}}}div"):
        if (div.get("TYPE") or "").upper() != "ARTICLE":
            continue
        pages_touched = set()
        for area in div.iter(f"{{{METS_NS}}}area"):
            page_stem = file_map.get(area.get("FILEID", ""))
            if page_stem:
                pages_touched.add(page_stem)
        for p in pages_touched:
            per_page[p] += 1
    return dict(per_page)

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=10,
                        help="nombre de fascicules à traiter (10 par défaut)")
    args = parser.parse_args()

    EMPLACEMENTS = emplacements()
    base, motif = EMPLACEMENTS["BnF"]
    fascicules = sorted(
        d.name for d in base.iterdir()
        if d.is_dir() and (d / "ocr").is_dir()
    )[:args.n]
    if not fascicules:
        raise SystemExit(f"aucun fascicule sous {base}")
    print(f"Fascicules ({len(fascicules)}) : {fascicules}\n")

    # Résultats : {fid: {page: {source: Counter}}}
    all_data = {}
    art_counts = {}  # {fid: {page: nb_articles}}

    for fid in fascicules:
        print(f"── {fid}")
        all_data[fid] = {}
        art_counts[fid] = articles_per_page(fid)

        # Récupérer toutes les pages disponibles (union des 3 sources)
        pages = set()
        for src in SOURCES:
            d = dossier(src, fid)
            if d.is_dir():
                pages |= {f.stem for f in d.glob("X*.xml")}

        for page in sorted(pages):
            all_data[fid][page] = {}
            for src in SOURCES:
                alto = dossier(src, fid) / f"{page}.xml"
                if alto.exists():
                    toks = tokens_from_alto(alto)
                    all_data[fid][page][src] = Counter(toks)
                else:
                    all_data[fid][page][src] = Counter()

        for src in SOURCES:
            n_pages = sum(1 for pg in all_data[fid] if all_data[fid][pg].get(src))
            mean_dist = (sum(len(all_data[fid][pg][src]) for pg in all_data[fid] if src in all_data[fid][pg]) / n_pages) if n_pages else 0
            print(f"  {src:8s}: {n_pages} pages, {mean_dist:.0f} tokens distincts/page moy.")

    # ── CSV détail par page ───────────────────────────────────────────────────
    csv_page = OUTPUT_DIR / "stats_par_page.csv"
    with open(csv_page, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["fascicule", "page", "source",
                    "tokens_distincts", "nb_mots_total", "nb_chars_total", "nb_articles"])
        for fid in fascicules:
            for page in sorted(all_data[fid]):
                nb_art = art_counts[fid].get(page, 0)
                for src in SOURCES:
                    c = all_data[fid][page].get(src, Counter())
                    if not c:
                        continue
                    nb_mots = sum(c.values())
                    nb_chars = sum(len(w2) * n for w2, n in c.items())
                    w.writerow([fid, page, src, len(c), nb_mots, nb_chars, nb_art])
    print(f"\n→ {csv_page}")

    # ── CSV Mistral brut par page ──────────────────────────────────────────────
    csv_mistral = OUTPUT_DIR / "stats_mistral_brut.csv"
    with open(csv_mistral, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["fascicule", "page",
                    "nb_mots_total", "nb_chars_total",
                    "tokens_distincts", "nb_articles"])
        for fid in fascicules:
            for page in sorted(all_data[fid]):
                c = all_data[fid][page].get("Mistral", Counter())
                if not c:
                    continue
                nb_mots  = sum(c.values())
                nb_chars = sum(len(w2) * n for w2, n in c.items())
                nb_art   = art_counts[fid].get(page, 0)
                w.writerow([fid, page, nb_mots, nb_chars, len(c), nb_art])
    print(f"→ {csv_mistral}")

    # ── Stats globales pour affichage + Venn ─────────────────────────────────
    # Agréger tokens distincts toutes pages confondues par source
    global_sets = {src: set() for src in SOURCES}
    totals = {src: {"mots": 0, "chars": 0, "pages": 0} for src in SOURCES}

    for fid in fascicules:
        for page in all_data[fid]:
            for src in SOURCES:
                c = all_data[fid][page].get(src, Counter())
                if not c:
                    continue
                global_sets[src] |= set(c.keys())
                totals[src]["mots"]  += sum(c.values())
                totals[src]["chars"] += sum(len(w2) * n for w2, n in c.items())
                totals[src]["pages"] += 1

    A, B, M = global_sets["BnF"], global_sets["Pero"], global_sets["Mistral"]
    zones = {
        "only_bnf":  len(A - B - M),
        "only_pero": len(B - A - M),
        "only_mis":  len(M - A - B),
        "bnf_pero":  len((A & B) - M),
        "bnf_mis":   len((A & M) - B),
        "pero_mis":  len((B & M) - A),
        "all3":      len(A & B & M),
        "total_bnf":     len(A),
        "total_pero":    len(B),
        "total_mistral": len(M),
    }

    # Moyennes par page
    n_pages_total = {src: totals[src]["pages"] for src in SOURCES}
    mean_mots  = {src: totals[src]["mots"]  / n_pages_total[src] if n_pages_total[src] else 0 for src in SOURCES}
    mean_chars = {src: totals[src]["chars"] / n_pages_total[src] if n_pages_total[src] else 0 for src in SOURCES}

    # nb articles moyen par page
    all_art_counts = []
    for fid in fascicules:
        for page in sorted(all_data[fid]):
            if any(all_data[fid][page].get(src) for src in SOURCES):
                all_art_counts.append(art_counts[fid].get(page, 0))
    mean_arts = sum(all_art_counts) / len(all_art_counts) if all_art_counts else 0

    print(f"\n{'─'*50}")
    print(f"Tokens distincts globaux : BnF={zones['total_bnf']}  Pero={zones['total_pero']}  Mistral={zones['total_mistral']}")
    print(f"Intersection BnF∩Pero∩Mistral : {zones['all3']}")
    for src in SOURCES:
        print(f"{src:8s}: {mean_mots[src]:.0f} mots/page  {mean_chars[src]:.0f} chars/page")
    print(f"Articles moyen/page (toutes sources) : {mean_arts:.1f}")

    # ── HTML Venn + stats ─────────────────────────────────────────────────────
    COLORS = {"BnF": "#3b82f6", "Pero": "#f97316", "Mistral": "#22c55e"}

    def stat_rows():
        """Construit les lignes <tr> du tableau HTML récapitulatif (tokens distincts,
        mots/page, caractères/page) pour chaque source (BnF, Pero, Mistral), plus
        une ligne de moyenne : utilisé dans le rapport vocab_venn.html.
        """
        rows = ""
        for src in SOURCES:
            c = COLORS[src]
            rows += f"""<tr>
              <td style="color:{c};font-weight:700">{src}</td>
              <td class="num">{zones[f'total_{src.lower()}']:,}</td>
              <td class="num">{mean_mots[src]:,.0f}</td>
              <td class="num">{mean_chars[src]:,.0f}</td>
            </tr>"""
        rows += f"""<tr style="border-top:2px solid #334155">
          <td style="color:#64748b">Moyenne toutes sources</td>
          <td></td>
          <td class="num" style="color:#94a3b8">{sum(mean_mots.values())/3:,.0f}</td>
          <td class="num" style="color:#94a3b8">{sum(mean_chars.values())/3:,.0f}</td>
        </tr>
        <tr>
          <td style="color:#64748b">Articles moy./page</td>
          <td class="num" colspan="3" style="color:#94a3b8">{mean_arts:.1f}</td>
        </tr>"""
        return rows

    html = f"""<!DOCTYPE html>
<html lang="fr"><head><meta charset="utf-8">
<title>Stats pages : comparaison OCR</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Segoe UI',sans-serif;background:#0f172a;color:#e2e8f0;padding:32px;display:flex;flex-direction:column;align-items:center}}
h1{{font-size:1.1rem;color:#93c5fd;margin-bottom:4px;text-align:center}}
.sub{{color:#475569;font-size:.8rem;margin-bottom:28px;text-align:center}}
.row{{display:flex;gap:24px;flex-wrap:wrap;justify-content:center;align-items:flex-start}}
.card{{background:#1e293b;border:1px solid #2d3f55;border-radius:10px;padding:20px 24px}}
.card h2{{font-size:.85rem;color:#93c5fd;margin-bottom:14px}}
table{{border-collapse:collapse}}
th{{background:#0f172a;color:#64748b;padding:8px 16px;font-size:.78rem;text-align:left}}
td{{padding:8px 16px;font-size:.85rem;border-top:1px solid #1e3a5f}}
td.num{{text-align:right;font-family:monospace}}
svg text{{font-family:'Segoe UI',sans-serif;fill:#e2e8f0}}
</style></head><body>
<h1>Statistiques par page : comparaison sources OCR</h1>
<div class="sub">{len(fascicules)} premiers fascicules · tokens en minuscules · hors blocs-titres</div>
<div class="row">

<div class="card">
<h2>Tokens distincts : diagramme de Venn</h2>
<svg viewBox="0 0 420 330" width="420" height="330">
  <circle cx="160" cy="145" r="115" fill="#3b82f6" fill-opacity=".2" stroke="#3b82f6" stroke-width="2"/>
  <circle cx="260" cy="145" r="115" fill="#f97316" fill-opacity=".2" stroke="#f97316" stroke-width="2"/>
  <circle cx="210" cy="215" r="115" fill="#22c55e" fill-opacity=".2" stroke="#22c55e" stroke-width="2"/>

  <text x="75"  y="65"  fill="#93c5fd" font-size="13" font-weight="700" text-anchor="middle">BnF</text>
  <text x="345" y="65"  fill="#fb923c" font-size="13" font-weight="700" text-anchor="middle">Pero</text>
  <text x="210" y="323" fill="#4ade80" font-size="13" font-weight="700" text-anchor="middle">Mistral</text>

  <text x="98"  y="140" font-size="13" text-anchor="middle">{zones['only_bnf']:,}</text>
  <text x="322" y="140" font-size="13" text-anchor="middle">{zones['only_pero']:,}</text>
  <text x="210" y="300" font-size="13" text-anchor="middle">{zones['only_mis']:,}</text>
  <text x="210" y="115" font-size="12" text-anchor="middle" fill="#94a3b8">{zones['bnf_pero']:,}</text>
  <text x="148" y="225" font-size="12" text-anchor="middle" fill="#94a3b8">{zones['bnf_mis']:,}</text>
  <text x="272" y="225" font-size="12" text-anchor="middle" fill="#94a3b8">{zones['pero_mis']:,}</text>
  <text x="210" y="185" font-size="14" font-weight="700" text-anchor="middle">{zones['all3']:,}</text>
</svg>
<p style="font-size:.72rem;color:#475569;text-align:center;margin-top:4px">
  Chiffre central = tokens communs aux 3 sources
</p>
</div>

<div class="card">
<h2>Moyennes par page</h2>
<table>
<thead><tr>
  <th>Source</th>
  <th>Tokens distincts</th>
  <th>Mots / page</th>
  <th>Chars / page</th>
</tr></thead>
<tbody>{stat_rows()}</tbody>
</table>
</div>

<div class="card">
<h2>Intersections</h2>
<table>
<tr><td style="color:#94a3b8">BnF ∩ Pero ∩ Mistral</td><td class="num" style="color:#34d399;font-weight:700">{zones['all3']:,}</td></tr>
<tr><td style="color:#94a3b8">BnF ∩ Mistral (sans Pero)</td><td class="num">{zones['bnf_mis']:,}</td></tr>
<tr><td style="color:#94a3b8">Pero ∩ Mistral (sans BnF)</td><td class="num">{zones['pero_mis']:,}</td></tr>
<tr><td style="color:#94a3b8">BnF ∩ Pero (sans Mistral)</td><td class="num">{zones['bnf_pero']:,}</td></tr>
<tr><td style="color:#f87171">Uniquement BnF</td><td class="num">{zones['only_bnf']:,}</td></tr>
<tr><td style="color:#f87171">Uniquement Pero</td><td class="num">{zones['only_pero']:,}</td></tr>
<tr><td style="color:#f87171">Uniquement Mistral</td><td class="num">{zones['only_mis']:,}</td></tr>
</table>
</div>

</div>
</body></html>"""

    html_out = OUTPUT_DIR / "vocab_venn.html"
    html_out.write_text(html, encoding="utf-8")
    print(f"→ {html_out}")
    print(f"\n✅ Terminé")
