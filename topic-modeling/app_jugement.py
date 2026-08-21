"""
app_jugement.py : interface d'annotation pour l'évaluation humaine des thèmes

CE QUE FAIT CE SCRIPT

Présente à l'annotateur les soixante items produits par jugement_humain.py, un
à la fois, et enregistre chaque réponse dès qu'elle est donnée.

L'enregistrement immédiat évite de perdre une séance si la fenêtre se ferme, et
permet de reprendre à l'item suivant. Le temps passé sur chaque item est mesuré,
information que le protocole d'origine ne relève pas et qui s'est révélée
instructive : l'hésitation suit la cohérence.

Les deux tâches sont enchaînées dans l'ordre qu'impose le protocole. La
dénomination ne s'ouvre qu'une fois l'intrusion terminée, l'intrus étant par
construction absent des dix mots de tête présentés à la seconde tâche.

ENTRÉES

  jugement/kit_annotateur_{n}/1_intrusion.csv
  jugement/kit_annotateur_{n}/2_denomination.csv

SORTIES

  jugement/reponses/{kit}_intrusion.csv
  jugement/reponses/{kit}_denomination.csv

Chaque ligne enregistrée porte la réponse, le temps en secondes et un
horodatage. Le dépouillement se fait par jugement_humain.py --depouiller.

PAQUETS EMPLOYÉS

  csv, time, datetime, pathlib   bibliothèque standard
  streamlit                      interface web locale. Le choix tient à ce
                                 qu'elle produit une page utilisable sans
                                 installation côté annotateur, et qu'elle gère
                                 seule l'état de la session.

DEUX DÉFAUTS RENCONTRÉS ET CORRIGÉS

L'écran de transition entre les deux tâches se réaffichait indéfiniment. Sa
condition d'affichage était l'absence de réponse à la seconde tâche, laquelle
reste vraie tant que la première réponse n'est pas donnée. Un drapeau explicite
dans l'état de session la remplace.

Le formulaire de la seconde tâche échouait en silence lorsque aucun bouton radio
n'était coché, le bouton de validation ne produisant alors aucun effet visible.
Trois boutons directs le remplacent, chacun valant réponse et enregistrement.

USAGE

    streamlit run app_jugement.py

Voir LANCEMENT.md dans le kit envoyé à l'annotateur pour l'installation.
"""

import csv
import time
from datetime import datetime
from pathlib import Path

import streamlit as st

# Lancée par « python3 app_jugement.py », l'interface s'exécute hors de
# Streamlit : les widgets rendent None et l'erreur qui s'ensuit ne dit pas la
# cause. Le cas est intercepté ici pour donner la commande attendue.
if not st.runtime.exists():
    raise SystemExit(
        "cette interface s'exécute sous Streamlit, non par l'interpréteur "
        "seul :\n    streamlit run app_jugement.py")

ICI = Path(__file__).resolve().parent
JUGEMENT = ICI / "jugement"
REPONSES = JUGEMENT / "reponses"

st.set_page_config(page_title="Évaluation des thèmes", page_icon="📰",
                   layout="centered")


# ── Données ───────────────────────────────────────────────────────────────────

def kits() -> list[str]:
    """Liste les kits présents, un par annotateur.

    Chaque annotateur emploie le sien. Les items y sont identiques, seul le
    fichier de réponses diffère, ce qui permet de mesurer l'accord entre deux
    jugements portant sur les mêmes thèmes.
    """
    return sorted(d.name for d in JUGEMENT.glob("kit_annotateur_*") if d.is_dir())


@st.cache_data
def charger(kit: str, fichier: str) -> list[dict]:
    """Lit un fichier d'items et rend ses lignes.

    Le décorateur cache_data évite de relire le fichier à chaque interaction :
    Streamlit réexécute le script entier à chaque clic, et la lecture serait
    répétée soixante fois par séance.
    """
    with (JUGEMENT / kit / fichier).open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def chemin_reponses(kit: str, tache: str) -> Path:
    """Chemin du fichier de réponses d'un kit pour une tâche.

    Le dossier est créé au besoin, de sorte que la première réponse d'une séance
    ne demande aucune préparation.
    """
    REPONSES.mkdir(parents=True, exist_ok=True)
    return REPONSES / f"{kit}_{tache}.csv"


def deja_repondu(kit: str, tache: str) -> dict[int, dict]:
    """Rend les réponses déjà enregistrées, indexées par numéro d'item.

    C'est ce dictionnaire qui permet la reprise : les items qui y figurent sont
    écartés de la liste à présenter, et la séance repart à l'item suivant.
    """
    f = chemin_reponses(kit, tache)
    if not f.exists():
        return {}
    with f.open(encoding="utf-8") as fh:
        return {int(l["item"]): l for l in csv.DictReader(fh)}


def enregistrer(kit: str, tache: str, ligne: dict, colonnes: list[str]) -> None:
    """Ajoute une réponse au fichier, en écrivant l'en-tête au premier appel.

    L'ouverture en mode ajout garantit qu'une réponse déjà écrite ne peut être
    perdue par la suite de la séance.
    """
    f = chemin_reponses(kit, tache)
    neuf = not f.exists()
    with f.open("a", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=colonnes)
        if neuf:
            w.writeheader()
        w.writerow(ligne)


# ── État de session ───────────────────────────────────────────────────────────

if "demarre" not in st.session_state:
    st.session_state.demarre = None       # horodatage de début de séance
    st.session_state.debut_item = None    # horodatage d'affichage de l'item
    st.session_state.kit = None
    # L'écran de transition ne peut pas se déduire des réponses enregistrées :
    # tant que la seconde tâche n'a reçu aucune réponse, la condition « aucune
    # réponse » reste vraie et l'écran se réafficherait indéfiniment.
    st.session_state.phase2 = False

# ── Écran d'accueil ───────────────────────────────────────────────────────────

if st.session_state.demarre is None:
    st.title("Évaluation des thèmes")
    st.markdown("""
Un programme a réparti automatiquement les articles de cent journaux anciens,
de 1819 à 1953, en groupes qui se ressemblent. Chaque groupe est décrit par les
mots qui y reviennent le plus.

Il s'agit de savoir si ces groupes se tiennent, autrement dit si un lecteur y
reconnaît quelque chose. Aucune connaissance particulière n'est nécessaire, et
c'est votre intuition de lecteur qui est mesurée.

**Deux tâches**, une vingtaine de minutes en tout. La seconde s'ouvre lorsque la
première est terminée. Vos réponses sont enregistrées au fur et à mesure, et
vous pouvez fermer la fenêtre puis reprendre où vous en étiez.
""")
    dispo = kits()
    if not dispo:
        st.error("Aucun kit trouvé dans `jugement/`.")
        st.stop()
    kit = st.selectbox("Votre kit", dispo,
                       help="Chaque annotateur emploie le sien.")
    if st.button("Commencer", type="primary"):
        st.session_state.kit = kit
        st.session_state.demarre = time.time()
        st.session_state.debut_item = time.time()
        st.rerun()
    st.stop()

kit = st.session_state.kit
intrusion = charger(kit, "1_intrusion.csv")
denomination = charger(kit, "2_denomination.csv")
faites_1 = deja_repondu(kit, "intrusion")
faites_2 = deja_repondu(kit, "denomination")

# ── Barre latérale ────────────────────────────────────────────────────────────

with st.sidebar:
    st.caption(kit.replace("_", " "))
    ecoule = int(time.time() - st.session_state.demarre)
    st.metric("Temps de séance", f"{ecoule // 60} min {ecoule % 60:02d} s")
    st.progress(len(faites_1) / len(intrusion), text=f"Tâche 1 : {len(faites_1)}/{len(intrusion)}")
    if len(faites_1) >= len(intrusion):
        st.progress(len(faites_2) / len(denomination),
                    text=f"Tâche 2 : {len(faites_2)}/{len(denomination)}")
    st.caption("Les réponses sont enregistrées à chaque validation.")

# ── Tâche 1 : intrusion ───────────────────────────────────────────────────────

restants_1 = [l for l in intrusion if int(l["item"]) not in faites_1]

if restants_1:
    ligne = restants_1[0]
    n = int(ligne["item"])
    st.subheader(f"Tâche 1, l'intrus  ·  item {len(faites_1) + 1} sur {len(intrusion)}")
    st.markdown("Cinq de ces mots viennent d'un même groupe, le sixième vient "
                "d'ailleurs. **Lequel est l'intrus ?**")
    st.caption("Répondez même si vous hésitez. Certains groupes n'auront aucun "
               "sens, et c'est un résultat attendu : répondez au jugé.")

    mots = [ligne[f"mot_{i}"] for i in range(1, 7)]
    cols = st.columns(3)
    choix = None
    for i, mot in enumerate(mots):
        if cols[i % 3].button(mot, key=f"m{n}_{i}", use_container_width=True):
            choix = mot
    st.write("")
    if st.button("Passer cet item", key=f"skip{n}"):
        choix = ""

    if choix is not None:
        enregistrer(kit, "intrusion",
                    {"item": n, "intrus_designe": choix,
                     "secondes": round(time.time() - st.session_state.debut_item, 1),
                     "horodatage": datetime.now().isoformat(timespec="seconds")},
                    ["item", "intrus_designe", "secondes", "horodatage"])
        st.session_state.debut_item = time.time()
        st.rerun()
    st.stop()

# ── Transition ────────────────────────────────────────────────────────────────

if not faites_2 and not st.session_state.get("phase2"):
    st.success("Première tâche terminée. Merci.")
    st.markdown("""
La seconde tâche présente cette fois **les dix mots** de chaque groupe, et
demande si vous sauriez le nommer. Elle est plus rapide.
""")
    if st.button("Continuer", type="primary"):
        st.session_state.phase2 = True
        st.session_state.debut_item = time.time()
        st.rerun()
    st.stop()

# ── Tâche 2 : dénomination ────────────────────────────────────────────────────

restants_2 = [l for l in denomination if int(l["item"]) not in faites_2]

if restants_2:
    ligne = restants_2[0]
    n = int(ligne["item"])
    st.subheader(f"Tâche 2, nommer  ·  item {len(faites_2) + 1} sur {len(denomination)}")
    st.markdown("Les dix mots de ce groupe :")
    st.info(ligne["dix_mots"])
    st.markdown("**Sauriez-vous donner à ce groupe un intitulé que comprendrait "
                "quelqu'un consultant un site de presse ancienne ?**")

    # Sans formulaire : chaque bouton vaut réponse et enregistrement immédiat.
    # Le formulaire précédent échouait en silence lorsque aucune case n'était
    # cochée, le bouton de validation ne produisant alors aucun effet visible.
    nom = st.text_input("L'intitulé, si vous en voyez un",
                        key=f"nom{n}",
                        placeholder="sport, annonces immobilières, politique étrangère…")
    c1, c2, c3 = st.columns(3)
    rep = None
    if c1.button("oui", key=f"o{n}", use_container_width=True,
                 help="un intitulé s'impose"):
        rep = "oui"
    if c2.button("approximativement", key=f"a{n}", use_container_width=True,
                 help="je vois à peu près, sans pouvoir le nommer"):
        rep = "approx"
    if c3.button("non", key=f"n{n}", use_container_width=True,
                 help="rien ne se dégage"):
        rep = "non"

    if rep:
        enregistrer(kit, "denomination",
                    {"item": n, "nommable": rep, "nom_propose": nom.strip(),
                     "secondes": round(time.time() - st.session_state.debut_item, 1),
                     "horodatage": datetime.now().isoformat(timespec="seconds")},
                    ["item", "nommable", "nom_propose", "secondes", "horodatage"])
        st.session_state.debut_item = time.time()
        st.rerun()
    st.caption("Écrivez l'intitulé d'abord si vous en avez un, puis répondez.")
    st.stop()

# ── Fin ───────────────────────────────────────────────────────────────────────

total = int(time.time() - st.session_state.demarre)
st.balloons()
st.title("Terminé")
st.markdown(f"""
Les {len(intrusion)} items des deux tâches sont annotés. Séance de
**{total // 60} minutes**.

Les réponses se trouvent dans `jugement/reponses/`. Rien d'autre n'est à faire.
""")
st.caption("Merci du temps que vous y avez consacré.")
