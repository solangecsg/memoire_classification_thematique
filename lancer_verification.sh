#!/bin/sh
# Lance l'interface d'annotation dans son propre environnement.
# Le désarmement de VIRTUAL_ENV est nécessaire : l'environnement hérité
# désigne un autre venv du dépôt, que l'interpréteur tente alors de lire.
unset VIRTUAL_ENV PYTHONHOME PYTHONPATH
ICI=$(cd "$(dirname "$0")" && pwd)
exec "$ICI/.venv-verif/bin/python3.14" -m streamlit run \
     "$ICI/classification/app_verification.py" \
     --server.port 8502 --server.headless true --browser.gatherUsageStats false
