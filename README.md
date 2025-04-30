# Construction de Thésaurus Hiérarchique Automatique (SKOS)

> Première version des fichiers, ces derniers n'ont pas encore été testés.

Ce projet permet de structurer automatiquement une liste plate de termes en un **thésaurus SKOS hiérarchique enrichi**.

## Fonctionnalités principales

- Analyse linguistique avancée (spaCy)
- Embeddings sémantiques multilingues (Sentence Transformers)
- Clustering hybride (KMeans, Louvain, DBSCAN)
- Génération automatique de hiérarchie broader/narrower/related
- Exportation SKOS au format RDF/Turtle
- Visualisations interactives (t-SNE, distribution de clusters)
- Rapport HTML complet des résultats

## Organisation du projet

```bash
votre-projet-thesaurus/
├── data/         # Fichiers CSV d'entrée
├── outputs/      # Résultats générés automatiquement
├── logs/         # Fichiers de log
├── src/          # Code source Python
├── requirements.txt
├── README.md
└── run_thesaurus.sh / run_thesaurus.bat
```
## Installation

1. Cloner le projet :

```bash
git clone https://github.com/votre-utilisateur/votre-projet-thesaurus.git
cd votre-projet-thesaurus
```

2. Installer les dépendances :

```bash
pip install -r requirements.txt
```
## Utilisation

Lancer un traitement complet :

```bash
python src/main.py --csv data/termes.csv --domain musique_baroque --save-model --generate-report --verbose
```

Ou uniquement exporter un fichier existant

```bash
python src/main.py --csv data/termes.csv --only-export
```
### Options disponibles :

| Option | Description |
|:------:|:------------:|
| `--csv` | Chemin du fichier CSV à traiter |
| `--domain` | Nom du domaine (ex : musique_baroque) |
| `--language` | Langue utilisée (par défaut : `fr`) |
| `--max-depth` | Profondeur maximale de la hiérarchie |
| `--generate-report` | Générer un rapport HTML complet |
| `--only-export` | Exporter uniquement sans recalculer |
| `--verbose` | Activer les logs détaillés |

## Licence

Ce projet est sous licence MIT - libre à vous de l'utiliser, le modifier, le diffuser.