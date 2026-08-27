"""
conftest.py : charge les scripts du dépôt comme des modules importables

POURQUOI CE FICHIER

Les scripts de ce dépôt sont écrits pour être lancés en ligne de commande, pas
pour être importés : ils vivent dans des dossiers sans `__init__.py` et portent
des noms avec des tirets. Les importer par leur chemin, plutôt que de les
réorganiser en paquet, laisse les scripts inchangés et permet malgré tout de
tester leurs fonctions pures.

L'import se fait à la demande et le résultat est mis en cache pour la session :
`classify_iptc_mistral_batched` télécharge un tokenizer au premier appel, et le
faire une fois par test coûterait cher.

PAQUETS EMPLOYÉS

  importlib, sys, pathlib   bibliothèque standard
  pytest                    fixtures
"""

import importlib.util
import sys
from pathlib import Path

import pytest

RACINE = Path(__file__).resolve().parent.parent
_CACHE = {}


def charger(chemin_relatif):
    """Charge un script du dépôt comme module, une seule fois par session.

    Le dossier du script est ajouté au chemin de recherche : plusieurs scripts
    s'importent entre eux par leur nom de fichier."""
    if chemin_relatif in _CACHE:
        return _CACHE[chemin_relatif]
    chemin = RACINE / chemin_relatif
    if not chemin.exists():
        pytest.skip(f"{chemin_relatif} absent du dépôt")
    dossier = str(chemin.parent)
    if dossier not in sys.path:
        sys.path.insert(0, dossier)
    nom = chemin.stem
    spec = importlib.util.spec_from_file_location(nom, chemin)
    module = importlib.util.module_from_spec(spec)
    sys.modules[nom] = module
    try:
        spec.loader.exec_module(module)
    except Exception as e:                      # dépendance absente de la CI
        pytest.skip(f"{chemin_relatif} non importable : {type(e).__name__} {e}")
    _CACHE[chemin_relatif] = module
    return module


@pytest.fixture(scope="session")
def verif():
    """Le script qui fabrique et dépouille les épreuves de vérification."""
    return charger("classification/verification_etiquettes.py")


@pytest.fixture(scope="session")
def lots():
    """Le script de classification par lots, dont dépend tout le chiffrage."""
    return charger("classification/classify_iptc_mistral_batched.py")


@pytest.fixture(scope="session")
def recap():
    """Le récapitulatif de la campagne de modélisation thématique."""
    return charger("topic-modeling/recap_runs.py")
