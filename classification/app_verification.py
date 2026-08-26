"""
app_verification.py : interface d'annotation pour la vérification des étiquettes

CE QUE FAIT CE SCRIPT

Présente un à un les items produits par verification_etiquettes.py et par
verification_unitaire.py, et enregistre chaque réponse dès qu'elle est donnée.

L'enregistrement immédiat évite de perdre une séance si la fenêtre se ferme, et
permet de reprendre à l'item suivant. C'est le même parti que app_jugement.py,
pour la même raison : une séance de deux à trois heures ne se conduit pas d'un
trait, et une grille CSV de 191 lignes remplie à la main désaligne tôt ou tard.

Le temps passé sur chaque item est mesuré. La mesure sert au dépouillement :
une hésitation longue sur un leurre accepté ne dit pas la même chose qu'une
acceptation immédiate.

L'ORDRE DES ÉPREUVES

A, puis B, puis C, puis D. Les trois ne partagent aucun item, et leurs identifiants ne se
recouvrent pas, de sorte qu'une grille de réponse unique les porte toutes. Fixer
l'ordre rend les séances comparables si l'épreuve devait être refaite.

CE QUE L'INTERFACE NE MONTRE PAS

Ni la fréquence de l'étiquette, ni la bande dont l'item est tiré, ni le fait
qu'un item de l'épreuve B soit un leurre. La clé n'est jamais lue par ce
script : elle ne sert qu'au dépouillement.

ENTRÉES

  verification/A_hapax.md          les 62 items de la première épreuve
  verification/B_acceptabilite.md  les 129 items de la deuxième
  verification/C_unitaire.md       les 116 items de la troisième, sur les
                                   étiquettes propres au régime un article
                                   par appel
  verification/D_cascades.md       les 160 items de la quatrième, sur les deux
                                   cascades à deux étages, mêlées et anonymes

SORTIES

  verification/reponses.csv        une ligne par item annoté

PAQUETS EMPLOYÉS

  csv, re, time, datetime, pathlib   bibliothèque standard
  streamlit                          interface web locale, comme pour
                                     app_jugement.py et pour la même raison :
                                     une page utilisable sans rien installer
                                     d'autre du côté de l'annotateur.

USAGE

    streamlit run app_verification.py

Le dépouillement se fait ensuite par :

    python3 verification_etiquettes.py --depouiller   # épreuves A et B
    python3 verification_unitaire.py --depouiller     # épreuve C
"""

import csv
import re
import time
from datetime import datetime
from pathlib import Path

import streamlit as st

ICI = Path(__file__).resolve().parent
SORTIE = ICI / "verification"
REPONSES = SORTIE / "reponses.csv"
COLONNES = ["item", "reponse", "secondes", "horodatage"]


def lire_items(fichier):
    """Extrait les items d'un kit. Le format est celui qu'écrit
    verification_etiquettes.py : un titre de niveau 2 portant l'identifiant,
    l'étiquette en gras, le titre de la carte logique s'il existe, et le texte
    en citation."""
    texte = (SORTIE / fichier).read_text(encoding="utf-8")
    items = []
    for bloc in re.split(r"\n## ", texte)[1:]:
        ident = bloc.split("\n", 1)[0].strip()
        etiq = re.search(r"\*\*Étiquette proposée : (.+?)\*\*", bloc)
        titre = re.search(r"\*Titre relevé dans la carte logique :\* (.+)", bloc)
        corps = re.findall(r"^> (.+)$", bloc, flags=re.MULTILINE)
        items.append({
            "item": ident,
            "etiquette": etiq.group(1).strip() if etiq else "",
            "titre": titre.group(1).strip() if titre else "",
            "texte": " ".join(corps).strip(),
        })
    return items


def deja_faits():
    """Les items dont une réponse est déjà enregistrée. La séance reprend ainsi où
    elle s'est arrêtée, ce qu'une annotation de plusieurs heures demande."""
    if not REPONSES.exists():
        return set()
    with REPONSES.open(encoding="utf-8") as f:
        return {l["item"] for l in csv.DictReader(f)
                if l.get("item") and l.get("reponse")}


def enregistrer(ligne):
    """Écrit une réponse et la met sur le disque aussitôt. La grille vide laissée
    par le script de fabrication n'a que deux colonnes ; elle est remplacée à
    la première réponse plutôt que complétée."""
    neuf = not REPONSES.exists() or REPONSES.stat().st_size == 0
    # La grille vide écrite par verification_etiquettes.py n'a que deux
    # colonnes ; on la remplace à la première réponse.
    if not neuf:
        with REPONSES.open(encoding="utf-8") as f:
            neuf = (f.readline().strip() != ",".join(COLONNES))
        if neuf:
            faits = []
            with REPONSES.open(encoding="utf-8") as f:
                for l in csv.DictReader(f):
                    if l.get("reponse"):
                        faits.append({**{c: "" for c in COLONNES}, **l})
            with REPONSES.open("w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, COLONNES)
                w.writeheader()
                w.writerows(faits)
            neuf = False
    with REPONSES.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, COLONNES)
        if neuf:
            w.writeheader()
        w.writerow(ligne)


# ── Mise en place ─────────────────────────────────────────────────────────────

st.set_page_config(page_title="Vérification des étiquettes", layout="centered")

# Les kits sont lus dans l'ordre où les épreuves ont été conduites. Les
# identifiants ne se recouvrent pas, et la grille de réponse est commune.
ITEMS = []
for _kit in ("A_hapax.md", "B_acceptabilite.md", "C_unitaire.md", "D_cascades.md",
             "E_justification.md"):
    if (SORTIE / _kit).exists():
        ITEMS += lire_items(_kit)
FAITS = deja_faits()
RESTE = [i for i in ITEMS if i["item"] not in FAITS]

if "debut_item" not in st.session_state:
    st.session_state.debut_item = time.time()
if "demarre" not in st.session_state:
    st.session_state.demarre = time.time()

# ── Fin ───────────────────────────────────────────────────────────────────────

if not RESTE:
    total = int(time.time() - st.session_state.demarre)
    st.balloons()
    st.title("Terminé")
    st.markdown(f"""
Les {len(ITEMS)} items sont annotés. Les réponses se trouvent dans
`verification/reponses.csv`.

Le dépouillement se lance par :

```
python3 verification_etiquettes.py --depouiller
```
""")
    st.caption(f"Séance de {total // 60} minutes.")
    st.stop()

# ── Un item ───────────────────────────────────────────────────────────────────

it = RESTE[0]
n_faits = len(ITEMS) - len(RESTE)
epreuve = {"A": "A — étiquettes employées une seule fois",
           "B": "B — acceptabilité",
           "C": "C — étiquettes du régime un article par appel",
           "D": "D — cascades à deux étages",
           "E": "E — justification demandée avant l'étiquette"}\
          .get(it["item"][0], it["item"][0])

st.progress(n_faits / len(ITEMS))
st.caption(f"Épreuve {epreuve} · item {n_faits + 1} sur {len(ITEMS)}")

st.markdown(f"## {it['etiquette']}")
if it["titre"]:
    st.caption(f"Titre relevé dans la carte logique : {it['titre']}")
st.markdown(f"> {it['texte']}")

st.markdown("**Cette étiquette convient-elle à cet article ?**")
c1, c2, c3 = st.columns(3)
rep = None
if c1.button("oui", key=f"o{it['item']}", use_container_width=True,
             help="l'étiquette décrit ce dont l'article traite"):
    rep = "o"
if c2.button("non", key=f"n{it['item']}", use_container_width=True,
             help="l'étiquette ne convient pas"):
    rep = "n"
if c3.button("j'hésite", key=f"h{it['item']}", use_container_width=True,
             help="ni clairement juste ni clairement fausse"):
    rep = "?"

if rep:
    enregistrer({"item": it["item"], "reponse": rep,
                 "secondes": round(time.time() - st.session_state.debut_item, 1),
                 "horodatage": datetime.now().isoformat(timespec="seconds")})
    st.session_state.debut_item = time.time()
    st.rerun()

st.caption("Jugez sur le texte montré, sans chercher à deviner ce que le modèle "
           "a fait. Dans les épreuves B et C, une partie des étiquettes n'a pas "
           "été attribuée à l'article présenté.")
