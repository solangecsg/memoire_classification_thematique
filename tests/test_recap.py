"""
test_recap.py : la détection des thèmes occupés par des débris d'océrisation

POURQUOI CES TESTS

Le mémoire rapporte qu'un thème sur dix est occupé par des fragments que la
reconnaissance de caractères a laissés, et que leur part passe de 3 à 11 pour
cent entre 1880 et 1919. Ce compte repose entièrement sur un seuil de
convention : au moins cinq des dix mots de tête comptent deux caractères ou
moins. Le seuil vaut par sa constance plutôt que par sa justesse, ce qui rend un
test d'autant plus utile : il fige la convention.

PAQUETS EMPLOYÉS

  pytest    exécution des tests
"""

import pytest


def test_theme_de_bruit_reconnait_des_debris(recap):
    """Un cas relevé dans la campagne : des fragments de deux caractères."""
    assert recap.theme_de_bruit("de la ce un il en et à num que") is True


def test_theme_francais_ordinaire_n_est_pas_du_bruit(recap):
    assert recap.theme_de_bruit(
        "gouvernement chambre ministre séance député loi projet vote "
        "commission budget") is False


def test_le_seuil_est_bien_de_cinq_sur_dix(recap):
    """Quatre mots courts ne suffisent pas, cinq suffisent : la convention est
    fixée ici, et le compte du mémoire en dépend."""
    quatre = "de la ce un gouvernement chambre ministre séance député loi"
    cinq = "de la ce un il chambre ministre séance député loi"
    assert recap.theme_de_bruit(quatre) is False
    assert recap.theme_de_bruit(cinq) is True


def test_seuls_les_dix_premiers_mots_comptent(recap):
    """Le critère porte sur les mots de tête : ce qui suit le dixième ne doit
    rien changer."""
    tete = "gouvernement chambre ministre séance député loi projet vote " \
           "commission budget"
    assert recap.theme_de_bruit(tete + " de la ce un il en et à") is False


def test_chaine_vide(recap):
    assert recap.theme_de_bruit("") is False


def test_le_resultat_est_un_booleen(recap):
    """La valeur sert à compter : une liste vide ne doit pas s'y glisser."""
    assert isinstance(recap.theme_de_bruit("de la ce un il"), bool)
    assert isinstance(recap.theme_de_bruit(""), bool)
