"""
test_verification.py : les fonctions qui fabriquent et dépouillent les épreuves

POURQUOI CES TESTS

Les taux rapportés par le mémoire reposent sur trois fonctions de ce script :
la stratification par bande, qui décide de ce qui est soumis au jugement ;
l'extrait, qui décide de ce que l'annotateur voit ; et l'intervalle de Wilson,
qui décide de ce qu'on peut conclure. Une erreur dans l'une d'elles fausserait
silencieusement toutes les mesures, sans qu'aucune exécution n'échoue.

PAQUETS EMPLOYÉS

  pytest    exécution des tests
  math      bibliothèque standard, pour les bornes attendues
"""

import math

import pytest


# ── la stratification par bande ──────────────────────────────────────────────

@pytest.mark.parametrize("effectif, attendu", [
    (1, "hapax"),          # employée une seule fois
    (2, "rare"),           # première valeur de la bande rare
    (9, "rare"),           # dernière valeur de la bande rare
    (10, "moyenne"),       # la borne appartient à la bande supérieure
    (99, "moyenne"),
    (100, "fréquente"),
    (14126, "fréquente"),  # le total d'attributions de la campagne
])
def test_bande_range_chaque_effectif(verif, effectif, attendu):
    assert verif.bande(effectif) == attendu


def test_bande_couvre_tout_le_domaine_sans_recouvrement(verif):
    """Chaque effectif reçoit une bande, et la bande change exactement aux
    bornes déclarées. Le test interroge la fonction plutôt que de refaire son
    calcul : une réécriture de la comparaison doit le faire échouer."""
    noms = {nom for nom, _, _ in verif.BANDES}
    ruptures = []
    for n in range(1, 500):
        b = verif.bande(n)
        assert b in noms, f"{n} rend une bande inconnue : {b}"
        if n > 1 and b != verif.bande(n - 1):
            ruptures.append(n)
    assert ruptures == [2, 10, 100], f"bornes déplacées : {ruptures}"


def test_bande_zero_retombe_sur_hapax(verif):
    """Un effectif nul n'existe pas dans le relevé, mais la fonction ne doit
    pas lever pour autant."""
    assert verif.bande(0) == "hapax"


# ── l'extrait montré à l'annotateur ──────────────────────────────────────────

def test_extrait_ramene_le_texte_sur_une_ligne(verif):
    assert verif.extrait("un\ntexte\tsur   plusieurs lignes", mots=10) == \
        "un texte sur plusieurs lignes"


def test_extrait_tronque_et_signale_la_troncature(verif):
    texte = " ".join(str(i) for i in range(50))
    court = verif.extrait(texte, mots=10)
    assert court.startswith("0 1 2")
    assert court.endswith("[…]")
    assert len(court.split()) == 11          # dix mots et la marque


def test_extrait_ne_signale_rien_si_rien_n_est_coupe(verif):
    assert "[…]" not in verif.extrait("trois mots seulement", mots=10)


def test_extrait_longueur_constante_quelle_que_soit_la_source(verif):
    """L'annotateur doit voir la même quantité de texte pour un article long
    que pour un bref : sans cela le long recevrait plus d'attention."""
    a = verif.extrait(" ".join(["mot"] * 1000), mots=180)
    b = verif.extrait(" ".join(["mot"] * 500), mots=180)
    assert len(a.split()) == len(b.split()) == 181


# ── l'intervalle de Wilson ───────────────────────────────────────────────────

def test_wilson_encadre_la_proportion_observee(verif):
    bas, haut = verif.wilson(33, 69)         # l'épreuve B du mémoire
    assert bas < 33 / 69 < haut


def test_wilson_reste_dans_zero_un(verif):
    """Contrairement à l'intervalle normal, Wilson ne sort jamais de [0, 1],
    et c'est la raison de son emploi sur des taux extrêmes."""
    for k, n in ((0, 58), (60, 60), (1, 60), (2, 79)):
        bas, haut = verif.wilson(k, n)
        assert 0.0 <= bas <= haut <= 1.0


def test_wilson_se_resserre_quand_l_effectif_croit(verif):
    """Cent fois plus d'observations pour la même proportion doivent donner un
    intervalle nettement plus étroit."""
    etroit = verif.wilson(500, 1000)
    large = verif.wilson(5, 10)
    assert (etroit[1] - etroit[0]) < (large[1] - large[0]) / 5


def test_wilson_effectif_nul_ne_leve_pas(verif):
    assert verif.wilson(0, 0) == (0.0, 0.0)


def test_wilson_valeurs_du_memoire(verif):
    """Les bornes citées dans le mémoire pour l'épreuve E, à un dixième de
    point près : si la fonction changeait, le texte deviendrait faux."""
    bas, haut = verif.wilson(28, 40)         # cascade avec justification
    assert math.isclose(bas * 100, 54.6, abs_tol=0.15)
    assert math.isclose(haut * 100, 81.9, abs_tol=0.15)
