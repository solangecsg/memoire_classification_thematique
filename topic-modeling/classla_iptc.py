"""
classla_iptc.py : classification thématique par un encodeur affiné

CE QUE FAIT CE SCRIPT

Applique au sous-corpus le classifieur thématique multilingue diffusé par le
groupe CLASSLA, et enregistre une prédiction par article.

Ce classifieur est obtenu par affinage de xlm-roberta-large sur quinze mille
dépêches en croate, slovène, catalan et grec, dont les étiquettes ont été
produites automatiquement par un modèle génératif plutôt que par des
annotateurs. Ses auteurs rapportent une macro-F1 de 0,746.

Il prédit les dix-sept catégories du premier niveau du référentiel IPTC, là où
la classification par modèle de langue conduite dans ../classification/ emploie
le troisième niveau et ses 567 entrées. L'intérêt de la comparaison tient à ce
décalage de grain : elle mesure ce qu'un modèle disponible sur étagère décrit
d'un corpus de presse ancienne, et ce qu'il laisse indistinct.

Le corpus est constitué par lda_mallet_corpus.constituer, avec les mêmes
paramètres que les chapitres 1 et 2, afin que les trois familles de méthodes
portent sur les mêmes documents.

CE QUE LE SCRIPT ENREGISTRE PAR ARTICLE

  etiquette        catégorie de premier niveau retenue, intitulé anglais
  score            probabilité que le modèle lui attribue
  seconde          deuxième catégorie la plus probable
  score_seconde    sa probabilité. L'écart entre les deux scores dit si la
                   décision était disputée.
  n_jetons         longueur du texte en jetons du modèle, avant troncature
  part_encodee     part du texte effectivement soumise à l'encodeur. Le modèle
                   n'en lit que 512 jetons, contrainte identique à celle des
                   modèles de plongement du chapitre 2.

ENTRÉES

  ../re-ocr/corpus/reocr_mistral/   texte ré-océrisé, source par défaut
  ../re-ocr/corpus/original/        texte hérité, avec --source bnf
  Le modèle est téléchargé depuis Hugging Face au premier lancement.

SORTIES

  resultats/classla_{source}_{granularite}_{n}_{date}/predictions.json
  resultats/classla_{source}_{granularite}_{n}_{date}/meta.json

PAQUETS EMPLOYÉS

  argparse, json, time, collections, datetime, pathlib   bibliothèque standard
  torch          calcul des tenseurs et accélération matérielle. Le script
                 choisit seul entre Metal sur Mac, CUDA sur carte NVIDIA et le
                 processeur.
  transformers   chargement du modèle et de son tokeniseur depuis Hugging Face
  lda_mallet_corpus   constitution du corpus, importée pour que les documents
                      soient identiques à ceux des autres chapitres

USAGE

    python3 classla_iptc.py --limite 20    # essai sur vingt articles
    python3 classla_iptc.py                # tout le sous-corpus ré-océrisé
"""

import argparse
import json
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import lda_mallet_corpus as base

ICI = Path(__file__).resolve().parent
OUTPUT_DIR = ICI / "resultats"
# Identifiant du modèle sur Hugging Face. Le téléchargement a lieu au premier
# lancement et occupe environ 2,2 Go.
MODELE = "classla/multilingual-IPTC-news-topic-classifier"

# Nombre maximal de jetons que l'encodeur accepte. La valeur tient à
# l'architecture du modèle et ne se règle pas : xlm-roberta-large est entraîné
# avec des positions bornées à 512. Un texte plus long est tronqué, et la clé
# part_encodee consigne la proportion effectivement lue.
LONGUEUR = 512


def charger_documents(source: str, granularite: str, min_mots: int,
                      longueur_min: int) -> list[dict]:
    """Constitue le corpus par le module partagé lda_mallet_corpus.

    Les listes d'arrêt et le seuil de longueur sont transmis parce qu'ils
    déterminent quels documents survivent au filtrage, donc la composition du
    corpus. Le texte soumis au modèle reste le texte brut, porté par la clé
    texte, le filtrage ne servant ici qu'à retenir les mêmes documents que les
    autres chapitres.
    """
    conf = base.SOURCES[source]
    suffixe = conf["suffixe"]
    fascicules = sorted(
        d.name[:-len(suffixe)] if suffixe and d.name.endswith(suffixe) else d.name
        for d in conf["racine"].iterdir() if d.is_dir())
    return base.constituer(source, granularite, fascicules, min_mots,
                           base.charger_stopwords(), base.charger_stoplocs(),
                           longueur_min, 0)


def classer(docs: list[dict], lot: int, appareil: str) -> tuple[list[dict], dict]:
    """Classe tous les documents et rend les prédictions avec un bilan.

    Le traitement procède ainsi.

    1. Charger le tokeniseur et le modèle, puis placer celui-ci en mode
       évaluation. Ce mode désactive les mécanismes propres à l'entraînement,
       qui rendraient les prédictions non reproductibles.
    2. Trier les documents par longueur croissante. Le remplissage d'un lot est
       déterminé par son élément le plus long, et le tri économise donc du
       calcul sur les lots de documents courts.
    3. Pour chaque lot, encoder les textes avec troncature à 512 jetons, puis
       les encoder une seconde fois sans troncature afin de connaître leur
       longueur réelle et d'en déduire la part effectivement lue.
    4. Passer le lot dans le modèle sans calcul de gradient, ce que garantit
       torch.no_grad. Le gradient sert à l'entraînement et occuperait ici de la
       mémoire sans usage.
    5. Convertir les scores bruts en probabilités par une fonction softmax, puis
       retenir les deux catégories les plus probables.

    Les prédictions sont rendues dans l'ordre d'entrée, le tri par longueur
    étant défait par l'indexation du tableau sorties.
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODELE)
    modele = AutoModelForSequenceClassification.from_pretrained(MODELE)
    modele.eval().to(appareil)
    id2label = modele.config.id2label
    print(f"  {len(id2label)} étiquettes : {', '.join(sorted(id2label.values()))}")

    # Les documents sont traités par longueur croissante : le remplissage d'un
    # lot est déterminé par son élément le plus long, et le tri l'économise.
    ordre = sorted(range(len(docs)), key=lambda i: len(docs[i]["texte"]))
    sorties: list[dict | None] = [None] * len(docs)
    t0 = time.time()
    tronques = 0

    for debut in range(0, len(ordre), lot):
        idx = ordre[debut:debut + lot]
        textes = [docs[i]["texte"] for i in idx]
        enc = tok(textes, truncation=True, max_length=LONGUEUR,
                  padding=True, return_tensors="pt")
        # Part du texte réellement soumise à l'encodeur, mesurée avant troncature.
        entiers = tok(textes, truncation=False)["input_ids"]
        enc = {k: v.to(appareil) for k, v in enc.items()}
        with torch.no_grad():
            logits = modele(**enc).logits
        probas = torch.softmax(logits.float(), dim=-1).cpu()
        for rang, i in enumerate(idx):
            p = probas[rang]
            j = int(p.argmax())
            n_jetons = len(entiers[rang])
            if n_jetons > LONGUEUR:
                tronques += 1
            sorties[i] = {
                "doc_id": docs[i]["doc_id"],
                "fascicule": docs[i]["fascicule"],
                "unite": docs[i]["unite"],
                "page": docs[i]["page"],
                "n_mots": len(docs[i]["texte"].split()),
                "n_jetons": n_jetons,
                "part_encodee": round(min(1.0, LONGUEUR / n_jetons), 3),
                "etiquette": id2label[j],
                "score": round(float(p[j]), 4),
                # La seconde étiquette dit si la décision était disputée.
                "seconde": id2label[int(p.argsort(descending=True)[1])],
                "score_seconde": round(float(p.sort(descending=True).values[1]), 4),
            }
        fait = debut + len(idx)
        if fait % (lot * 10) < lot or fait == len(ordre):
            ecoule = time.time() - t0
            reste = ecoule / fait * (len(ordre) - fait)
            print(f"    {fait}/{len(ordre)} : {ecoule:.0f}s écoulées, "
                  f"{reste:.0f}s restantes", flush=True)

    info = {"secondes": round(time.time() - t0, 1), "tronques": tronques}
    return [s for s in sorties if s], info


def rapporter(preds: list[dict]) -> None:
    """Imprime la répartition des articles entre les catégories.

    Trois informations accompagnent le décompte. Le score médian par catégorie
    dit avec quelle assurance le modèle l'emploie. Le nombre de décisions sous
    une probabilité de 0,5 mesure la part des cas douteux. Ces deux quantités
    servent au chapitre 3, qui montre que l'accord avec la classification de
    troisième niveau croît avec l'assurance du modèle.
    """
    c = Counter(p["etiquette"] for p in preds)
    print(f"\n  {len(preds)} articles classés, {len(c)} étiquettes employées "
          f"sur 17\n")
    print(f"  {'étiquette':46} {'articles':>9} {'part':>7} {'score méd.':>11}")
    for lab, n in c.most_common():
        scores = sorted(p["score"] for p in preds if p["etiquette"] == lab)
        med = scores[len(scores) // 2]
        print(f"  {lab[:46]:46} {n:9} {n/len(preds):6.1%} {med:11.3f}")
    hesitants = sum(1 for p in preds if p["score"] < 0.5)
    print(f"\n  décisions sous 0,5 de probabilité : {hesitants} "
          f"({hesitants/len(preds):.1%})")


def main() -> None:
    """Constitue le corpus, lance la classification et enregistre les sorties.

    Les valeurs par défaut reprennent la configuration retenue aux chapitres 1
    et 2 : source ré-océrisée, granularité de l'article, seuil de 20 mots et
    longueur minimale de 3. Elles fixent le sous-corpus à 5 361 articles, ce qui
    rend les trois chapitres comparables document par document.

    La taille de lot vaut 16 par défaut. Une valeur plus grande accélère le
    traitement et demande davantage de mémoire vive sur le processeur
    graphique ; 16 tient dans la mémoire d'un ordinateur portable avec un modèle
    de cette taille.
    """
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=sorted(base.SOURCES), default="mistral")
    p.add_argument("--granularite", choices=["bloc", "article"], default="article")
    p.add_argument("--min-mots", type=int, default=None)
    p.add_argument("--longueur-min", type=int, default=3,
                   help="valeur employée par les chapitres 1 et 2, qui fixe "
                        "le sous-corpus à 5 361 articles")
    p.add_argument("--lot", type=int, default=16)
    p.add_argument("--appareil", default=None, help="mps, cuda ou cpu")
    p.add_argument("--limite", type=int, default=None,
                   help="ne traiter que les n premiers documents, pour un essai")
    args = p.parse_args()

    if args.appareil is None:
        import torch
        args.appareil = ("mps" if torch.backends.mps.is_available()
                         else "cuda" if torch.cuda.is_available() else "cpu")
    min_mots = args.min_mots if args.min_mots is not None else (
        5 if args.granularite == "bloc" else 20)

    print(f"constitution du corpus ({args.source}, {args.granularite}, "
          f"seuil {min_mots} mots)")
    docs = charger_documents(args.source, args.granularite, min_mots,
                             args.longueur_min)
    if args.limite:
        docs = docs[:args.limite]
    print(f"  {len(docs)} documents")

    print(f"\nclassification sur {args.appareil}, lots de {args.lot}")
    preds, info = classer(docs, args.lot, args.appareil)
    rapporter(preds)
    print(f"  {info['tronques']} articles tronqués à {LONGUEUR} jetons "
          f"({info['tronques']/len(preds):.1%})")

    nom = (f"classla_{args.source}_{args.granularite}_{len(preds)}_"
           f"{datetime.now():%Y%m%d_%H%M%S}")
    dossier = OUTPUT_DIR / nom
    dossier.mkdir(parents=True, exist_ok=True)
    (dossier / "predictions.json").write_text(
        json.dumps(preds, ensure_ascii=False, indent=1), encoding="utf-8")
    (dossier / "meta.json").write_text(json.dumps({
        "modele": MODELE, "source": args.source, "granularite": args.granularite,
        "min_mots": min_mots, "longueur_min": args.longueur_min,
        "longueur_max": LONGUEUR, "lot": args.lot,
        "appareil": args.appareil, "n_documents": len(preds),
        "date": datetime.now().isoformat(timespec="seconds"), **info},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n  → {dossier}")


if __name__ == "__main__":
    main()
