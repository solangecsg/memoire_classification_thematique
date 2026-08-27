"""
test_lots.py : la constitution des lots et la lecture du référentiel IPTC

POURQUOI CES TESTS

Le groupage des articles est ce qui divise la dépense par neuf, et c'est aussi
ce qui appauvrit la description : la fonction qui compose les lots porte donc à
la fois l'économie et sa contrepartie. Deux propriétés doivent tenir sans quoi
la campagne entière serait à refaire : un article n'est jamais coupé, et aucun
lot ne dépasse le budget de jetons de l'appel.

Les fonctions de lecture du référentiel sont testées avec elles : une erreur sur
l'extraction du code ou du libellé produirait des étiquettes hors taxonomie,
que le schéma de sortie rejetterait sans dire pourquoi.

PAQUETS EMPLOYÉS

  pytest    exécution des tests
"""

import pytest


def articles(*longueurs):
    """Fabrique des articles de longueur voulue, en mots."""
    return [{"id": f"DIV.{i}", "text": " ".join(["mot"] * n)}
            for i, n in enumerate(longueurs, 1)]


# ── la lecture du référentiel ────────────────────────────────────────────────

def test_code_extrait_les_huit_chiffres(lots):
    assert lots._code({"qcode": "medtop:20000002"}) == "20000002"


def test_code_ne_coupe_qu_au_premier_deux_points(lots):
    assert lots._code({"qcode": "medtop:2000:0002"}) == "2000:0002"


def test_label_prefere_le_francais(lots):
    c = {"qcode": "medtop:29999999",
         "prefLabel": {"fr": "Agronomie", "en-GB": "Agronomy"}}
    assert lots._label(c) == "Agronomie"


def test_label_retombe_sur_l_anglais_puis_sur_ce_qui_reste(lots):
    sans_fr = {"qcode": "medtop:29999998", "prefLabel": {"en-GB": "Agronomy"}}
    assert lots._label(sans_fr) == "Agronomy"
    exotique = {"qcode": "medtop:29999997", "prefLabel": {"de": "Agronomie"}}
    assert lots._label(exotique) == "Agronomie"


def test_label_sans_intitule_rend_une_chaine_vide(lots):
    assert lots._label({"qcode": "medtop:29999996", "prefLabel": {}}) == ""


# ── la constitution des lots ─────────────────────────────────────────────────

def test_lots_aucun_article_n_est_perdu(lots):
    art = articles(*([50] * 60))
    rendus = [a for lot in lots.make_batches(art, 8000) for a in lot]
    assert len(rendus) == 60
    assert [a["id"] for a in rendus] == [a["id"] for a in art]


def test_lots_l_ordre_est_conserve(lots):
    """Les identifiants servent à rattacher la réponse du modèle à l'article :
    un lot qui réordonnerait rendrait le rattachement faux."""
    art = articles(*([40] * 30))
    plat = [a["id"] for lot in lots.make_batches(art, 8000) for a in lot]
    assert plat == sorted(plat, key=lambda x: int(x.split(".")[1]))


def test_lots_respectent_la_taille_maximale(lots):
    art = articles(*([5] * 200))             # très courts : seule la taille borne
    for lot in lots.make_batches(art, 8000):
        assert len(lot) <= lots.MAX_BATCH_SIZE


def test_lots_respectent_le_budget_de_jetons(lots):
    art = articles(*([600] * 40))
    for lot in lots.make_batches(art, 8000):
        cout = 8000 + sum(lots.count_tokens(a["text"])
                          + lots.PER_ARTICLE_WRAPPER_TOKENS for a in lot)
        assert len(lot) == 1 or cout <= lots.MAX_BATCH_TOKENS


def test_un_article_trop_gros_part_seul_sans_etre_coupe(lots):
    """La règle qui compte le plus : un article est envoyé entier ou pas du
    tout. Un article qui dépasse à lui seul le budget part dans son propre lot,
    jamais tronqué ni rejeté."""
    enorme = " ".join(["mot"] * 60000)
    art = [{"id": "DIV.1", "text": "court"},
           {"id": "DIV.2", "text": enorme},
           {"id": "DIV.3", "text": "court"}]
    lots_faits = lots.make_batches(art, 8000)
    seul = [b for b in lots_faits if any(a["id"] == "DIV.2" for a in b)]
    assert len(seul) == 1 and len(seul[0]) == 1
    assert seul[0][0]["text"] == enorme      # intact


def test_lots_liste_vide(lots):
    assert lots.make_batches([], 8000) == []


def test_aucun_lot_n_est_vide(lots):
    """Un lot vide partirait en appel sans contenu : la dépense serait payée
    pour rien, et la réponse serait rattachée à aucun article."""
    for cas in (articles(*([5] * 200)),
                articles(*([600] * 40)),
                [{"id": "DIV.1", "text": " ".join(["mot"] * 60000)}],
                [{"id": "DIV.1", "text": " ".join(["mot"] * 60000)},
                 {"id": "DIV.2", "text": "court"}]):
        for lot in lots.make_batches(cas, 8000):
            assert lot, "un lot vide a été produit"


def test_un_article_trop_gros_en_tete_part_aussi_seul(lots):
    """Le cas limite : l'article qui dépasse arrive en premier, quand aucun lot
    n'est encore ouvert."""
    enorme = " ".join(["mot"] * 60000)
    faits = lots.make_batches([{"id": "DIV.1", "text": enorme},
                               {"id": "DIV.2", "text": "court"}], 8000)
    assert all(len(b) >= 1 for b in faits)
    premier = [b for b in faits if any(a["id"] == "DIV.1" for a in b)][0]
    assert len(premier) == 1 and premier[0]["text"] == enorme


def test_le_groupage_reduit_bien_le_nombre_d_appels(lots):
    """L'économie mesurée par le mémoire tient à ce nombre : cent articles
    courts doivent tenir en bien moins de cent appels."""
    art = articles(*([120] * 100))
    assert len(lots.make_batches(art, 8000)) <= 100 / 4


# ── le comptage de jetons ────────────────────────────────────────────────────

def test_count_tokens_croit_avec_le_texte(lots):
    assert lots.count_tokens("mot " * 10) < lots.count_tokens("mot " * 100)


def test_count_tokens_jamais_nul_sur_du_texte(lots):
    assert lots.count_tokens("a") >= 1
