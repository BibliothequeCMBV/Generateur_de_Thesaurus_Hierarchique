"""
Hiérarchisation automatisée et enrichissement de thésaurus SKOS

Ce script permet de structurer une liste plate de termes en un thésaurus hiérarchique SKOS
en utilisant des techniques avancées de traitement du langage naturel, d'apprentissage automatique
et d'analyse de graphes.

Principales fonctionnalités:
- Analyse sémantique des termes avec embeddings vectoriels
- Détection automatique des catégories et sous-catégories
- Identification des relations hiérarchiques (broader/narrower) et associatives (related)
- Détection de communautés dans le graphe de termes
- Évaluation de la qualité de la hiérarchie construite
- Exportation au format SKOS compatible (RDF/Turtle)
- Visualisations interactives de la structure hiérarchique

Amélioration par rapport aux versions précédentes:
- Architecture orientée objet modulaire et extensible
- Combinaison d'approches symboliques et d'apprentissage automatique
- Algorithmes avancés de détection de communautés
- Interface graphique pour visualiser la hiérarchie
- Système de validation automatique
- Support multilingue

Prérequis:
- Python 3.7+
- pandas, numpy, scikit-learn
- spacy avec un modèle linguistique approprié (ex: fr_core_news_md)
- sentence-transformers
- networkx, matplotlib, plotly pour les visualisations
- rdflib pour l'export SKOS
- python-louvain (community-detection) pour la détection de communautés
"""

import pandas as pd
import numpy as np
import re
import os
import sys
import spacy
import matplotlib.pyplot as plt
import networkx as nx
import rdflib
from rdflib import Graph, Literal, URIRef, Namespace
from rdflib.namespace import SKOS, RDF, RDFS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import dendrogram
from sentence_transformers import SentenceTransformer, util
import community as community_louvain  # aussi connu sous le nom de python-louvain
import logging
import warnings
import time
import json
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Union, Optional, Any
import argparse

# Fonction utilitaire pour rendre les objets JSON-sérialisables
def convert_numpy(obj):
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    else:
        raise TypeError(f"Type {type(obj)} not serializable")

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Sortie console
        logging.FileHandler('thesaurus_processing.log')  # Fichier de log
    ]
)
logger = logging.getLogger(__name__)

# Ignorer les avertissements
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class ThesaurusHierarchyBuilder:
    """
    Classe principale pour analyser, structurer et hiérarchiser un thésaurus.
    
    Cette classe encapsule toutes les fonctionnalités nécessaires pour transformer
    une liste plate de termes en une structure hiérarchique enrichie, compatible
    avec le standard SKOS (Simple Knowledge Organization System).
    
    Elle combine:
    - Des approches linguistiques (analyse morphosyntaxique, extraction d'entités)
    - Des techniques d'embedding vectoriel pour représenter la sémantique des termes
    - Des algorithmes de clustering et de détection de communautés
    - Des analyses de graphes pour identifier les relations entre termes
    
    Cette implémentation est conçue pour être modulaire, extensible et adaptable
    à différents domaines et langues.
    """
    
    def __init__(self, 
                 domain: str = "baroque_music",
                 language: str = "fr",
                 language_model: str = "fr_core_news_md",
                 embedding_model: str = "distiluse-base-multilingual-cased-v1",
                 max_depth: int = 3,
                 output_dir: str = "./outputs"):
        """
        Initialise le constructeur de hiérarchie avec les paramètres spécifiés.
        
        Parameters:
            domain (str): Domaine de connaissance du thésaurus (par défaut: "baroque_music")
            language (str): Code de langue à deux lettres, ex: "fr" pour français (par défaut: "fr")
            language_model (str): Nom du modèle spaCy à utiliser (par défaut: fr_core_news_md)
            embedding_model (str): Modèle SentenceTransformer pour les embeddings
            max_depth (int): Profondeur maximale de la hiérarchie à construire
            output_dir (str): Répertoire pour sauvegarder les résultats
        """
        # Métadonnées du thésaurus
        self.domain = domain
        self.language = language
        self.name = f"Thésaurus {domain.replace('_', ' ')}"
        self.description = f"Thésaurus hiérarchique sur {domain.replace('_', ' ')}"
        self.max_depth = max_depth
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialisation du constructeur de thésaurus pour le domaine: {domain}")
        logger.info(f"Langue: {language}, modèle linguistique: {language_model}")
        
        # Modèle linguistique
        try:
            self.nlp = spacy.load(language_model)
            logger.info(f"Modèle spaCy {language_model} chargé avec succès")
        except OSError:
            logger.warning(f"Modèle {language_model} non trouvé. Installation en cours...")
            try:
                os.system(f"python -m spacy download {language_model}")
                self.nlp = spacy.load(language_model)
            except Exception as e:
                logger.error(f"Échec de l'installation du modèle spaCy: {e}")
                # Essayer avec un modèle plus petit comme solution de repli
                fallback_model = f"{language}_core_web_sm"
                logger.warning(f"Tentative avec le modèle de repli: {fallback_model}")
                try:
                    os.system(f"python -m spacy download {fallback_model}")
                    self.nlp = spacy.load(fallback_model)
                except:
                    logger.critical("Impossible de charger un modèle spaCy. L'analyse linguistique sera limitée.")
                    # Créer un pipeline minimal si aucun modèle n'est disponible
                    self.nlp = spacy.blank(language)
        
        # Modèle d'embeddings
        try:
            self.embedding_model = SentenceTransformer(embedding_model)
            logger.info(f"Modèle d'embeddings {embedding_model} chargé avec succès")
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle d'embeddings: {e}")
            logger.info("Tentative avec un modèle d'embeddings alternatif...")
            try:
                # Essayer des alternatives
                alternatives = [
                    'paraphrase-multilingual-MiniLM-L12-v2',
                    'distilbert-base-nli-mean-tokens',
                    'all-MiniLM-L6-v2'
                ]
                for alt_model in alternatives:
                    try:
                        self.embedding_model = SentenceTransformer(alt_model)
                        logger.info(f"Modèle alternatif {alt_model} chargé avec succès")
                        break
                    except:
                        continue
            except Exception as e2:
                logger.critical(f"Impossible de charger un modèle d'embedding: {e2}")
                logger.warning("Utilisation de TF-IDF comme solution de repli pour les embeddings")
                self.embedding_model = None  # Sera remplacé par TF-IDF si nécessaire
        
        # Attributs pour stocker les données
        self.df = None                  # DataFrame des termes
        self.embeddings = None          # Matrice des embeddings
        self.similarity_matrix = None   # Matrice de similarité
        self.hierarchy = None           # Structure hiérarchique construite
        self.graph = None               # Graphe de relations entre termes
        self.metrics = {}               # Métriques d'évaluation
        
        # Configuration des exports
        self.namespace = Namespace(f"http://cmbv.fr/skos/{domain}/")
        
        # Timestamp pour identifier cette session
        self.timestamp = int(time.time())
        logger.info(f"Constructeur de thésaurus initialisé (ID: {self.timestamp})")

    def load_data(self, csv_file: str, term_column: str = "skos:prefLabel", definition_column: str = None) -> pd.DataFrame:
        """
        Charge et prépare les données depuis un fichier CSV, avec support pour les définitions.
        
        Cette fonction effectue:
        1. Le chargement du fichier CSV
        2. La détection des colonnes pertinentes (termes et définitions)
        3. Un prétraitement initial des termes et définitions
        
        Parameters:
            csv_file (str): Chemin vers le fichier CSV contenant les termes
            term_column (str): Nom de la colonne contenant les termes principaux
            definition_column (str, optional): Nom de la colonne contenant les définitions des termes
                
        Returns:
            pd.DataFrame: DataFrame avec les données chargées et prétraitées
                
        Raises:
            FileNotFoundError: Si le fichier CSV n'existe pas
            ValueError: Si la structure du fichier ne permet pas l'extraction des termes
        """
        logger.info(f"Chargement des données depuis {csv_file}")
        
        try:
            # Vérifier si le fichier existe
            if not os.path.exists(csv_file):
                raise FileNotFoundError(f"Le fichier {csv_file} n'existe pas")
            
            # Charger le CSV
            self.df = pd.read_csv(csv_file, sep="\t")
            logger.info(f"Données chargées: {len(self.df)} termes trouvés")
            logger.info(f"Colonnes disponibles: {', '.join(self.df.columns)}")
            
            # Vérifier si la colonne de termes spécifiée existe
            if term_column not in self.df.columns:
                available_cols = self.df.columns.tolist()
                logger.warning(f"Colonne {term_column} non trouvée")
                
                # Essayer de deviner la colonne de termes
                potential_cols = [col for col in available_cols if 
                                any(term in col.lower() for term in ['prefLabel', 'label', 'term', 'concept', 'mot', 'clé'])]
                
                if potential_cols:
                    term_column = potential_cols[0]
                    logger.info(f"Utilisation de la colonne {term_column} à la place")
                else:
                    # Utiliser la première colonne contenant des chaînes de caractères
                    for col in available_cols:
                        if self.df[col].dtype == 'object':
                            term_column = col
                            logger.warning(f"Utilisation de la colonne {term_column} par défaut")
                            break
                    else:
                        raise ValueError("Aucune colonne de texte appropriée trouvée dans le fichier")
            
            # Recherche de la colonne de définitions si non spécifiée
            if definition_column is None:
                potential_def_cols = [col for col in self.df.columns if 
                                    any(term in col.lower() for term in ['definition', 'desc', 'description', 'meaning', 'sens', 'skos:definition'])]
                
                if potential_def_cols:
                    definition_column = potential_def_cols[0]
                    logger.info(f"Colonne de définitions détectée automatiquement: {definition_column}")
            
            # Enregistrer le nom de la colonne de définitions si trouvée
            self.definition_column = definition_column
            
            # Vérifier les valeurs manquantes dans la colonne de termes
            missing = self.df[term_column].isna().sum()
            if missing > 0:
                logger.warning(f"{missing} valeurs manquantes trouvées dans la colonne {term_column}")
                self.df = self.df.dropna(subset=[term_column])
                logger.info(f"Données après suppression des valeurs manquantes: {len(self.df)} termes")
            
            # S'assurer que chaque terme est unique
            if self.df[term_column].duplicated().any():
                logger.warning(f"Termes dupliqués trouvés dans la colonne {term_column}")
                self.df = self.df.drop_duplicates(subset=[term_column])
                logger.info(f"Données après suppression des duplicatas: {len(self.df)} termes")
            
            # Réinitialiser les index
            self.df = self.df.reset_index(drop=True)
            
            # Vérifier et traiter les définitions si la colonne est disponible
            if definition_column and definition_column in self.df.columns:
                logger.info(f"Colonne de définitions trouvée: {definition_column}")
                # Remplacer les valeurs manquantes dans les définitions par une chaîne vide
                self.df[definition_column] = self.df[definition_column].fillna("")
                # Prétraiter les définitions
                self.df = self._preprocess_definitions(self.df, definition_column)
            else:
                logger.warning("Aucune colonne de définitions trouvée ou spécifiée")
                self.definition_column = None
            
            # Prétraitement des termes
            self.df = self._preprocess_terms(self.df, term_column)
            
            return self.df
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement des données: {e}")
            raise

    def _preprocess_terms(self, df: pd.DataFrame, term_column: str) -> pd.DataFrame:
        """
        Nettoie et prépare les termes pour l'analyse.
        
        Cette fonction effectue plusieurs opérations de prétraitement:
        1. Conservation des termes originaux
        2. Nettoyage (minuscules, suppression des caractères spéciaux)
        3. Analyse linguistique (lemmatisation, POS tagging, extraction d'entités)
        4. Identification des caractéristiques linguistiques
        
        Parameters:
            df (pd.DataFrame): DataFrame contenant les termes
            term_column (str): Nom de la colonne contenant les termes
            
        Returns:
            pd.DataFrame: DataFrame enrichi avec les colonnes de prétraitement
        """
        logger.info("Prétraitement linguistique des termes...")
        start_time = time.time()
        
        # Conserver les termes originaux
        df['original_term'] = df[term_column].copy()
        
        # Nettoyage basique
        df['clean_term'] = df[term_column].str.lower()
        df['clean_term'] = df['clean_term'].apply(lambda x: re.sub(r'[^\w\s]', ' ', str(x)) if pd.notna(x) else "")
        df['clean_term'] = df['clean_term'].apply(lambda x: re.sub(r'\s+', ' ', x).strip())
        
        # Générer un ID unique pour chaque terme si non disponible
        if 'conceptID' not in df.columns:
            logger.info("Génération d'identifiants uniques pour les termes")
            df['conceptID'] = df['clean_term'].apply(
                lambda x: re.sub(r'[^\w]', '_', x).lower()[:30] + '_' + str(hash(x) % 10000)
            )
        
        # Analyse linguistique avec spaCy
        logger.info("Analyse linguistique avec spaCy...")
        
        # Utiliser un traitement par lots pour plus d'efficacité
        batch_size = 100
        all_results = []
        
        # Traiter les termes par lots
        for i in range(0, len(df), batch_size):
            batch = df['clean_term'].iloc[i:i+batch_size].tolist()
            # Traiter les termes avec spaCy
            docs = list(self.nlp.pipe(batch))
            
            # Extraire les informations linguistiques
            batch_results = []
            for doc in docs:
                # Extraire les lemmes (formes de base des mots)
                lemmas = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
                
                # Extraire les parties du discours (nom, verbe, adjectif, etc.)
                pos_tags = [token.pos_ for token in doc if not token.is_stop and token.is_alpha]
                
                # Identifier les termes d'entité nommée (personnes, lieux, etc.)
                entities = [(ent.text, ent.label_) for ent in doc.ents]
                
                # Extraire les tokens significatifs (hors stop words)
                tokens = [token.text for token in doc if not token.is_stop and token.is_alpha]
                
                # Analyse syntaxique
                root_word = next((token.text for token in doc if token.dep_ == "ROOT"), None)
                
                # Analyse des modifieurs et compléments
                modifiers = [token.text for token in doc if token.dep_ in ["amod", "advmod", "compound"]]
                
                batch_results.append({
                    "lemma": " ".join(lemmas),
                    "pos": " ".join(pos_tags),
                    "entities": entities,
                    "tokens": tokens,
                    "n_tokens": len(tokens),
                    "root": root_word,
                    "modifiers": modifiers
                })
            
            all_results.extend(batch_results)
        
        # Ajouter les résultats au DataFrame
        df['lemmatized'] = [result["lemma"] for result in all_results]
        df['pos_tags'] = [result["pos"] for result in all_results]
        df['entities'] = [result["entities"] for result in all_results]
        df['tokens'] = [result["tokens"] for result in all_results]
        df['n_tokens'] = [result["n_tokens"] for result in all_results]
        df['root_word'] = [result["root"] for result in all_results]
        df['modifiers'] = [result["modifiers"] for result in all_results]
        
        # Détecter les caractéristiques linguistiques
        df['is_proper_noun'] = df['pos_tags'].apply(lambda x: 'PROPN' in x)
        df['is_person'] = df['entities'].apply(lambda x: any(label == 'PER' for _, label in x))
        df['is_compound'] = df['n_tokens'].apply(lambda x: x > 1)
        
        # Identifier les termes par longueur (simples vs complexes)
        df['term_complexity'] = pd.cut(
            df['n_tokens'], 
            bins=[0, 1, 2, 3, float('inf')], 
            labels=['simple', 'composé', 'expression', 'complexe'],
            include_lowest=True
        )
        
        elapsed_time = time.time() - start_time
        logger.info(f"Prétraitement terminé en {elapsed_time:.2f} secondes")
        
        return df
    
    def _preprocess_definitions(self, df: pd.DataFrame, definition_column: str) -> pd.DataFrame:
        """
        Nettoie et prépare les définitions pour l'analyse.
        
        Cette fonction effectue plusieurs opérations de prétraitement sur les définitions:
        1. Nettoyage (minuscules, suppression des caractères spéciaux)
        2. Analyse linguistique (lemmatisation, extraction de mots-clés)
        
        Parameters:
            df (pd.DataFrame): DataFrame contenant les définitions
            definition_column (str): Nom de la colonne contenant les définitions
            
        Returns:
            pd.DataFrame: DataFrame enrichi avec les colonnes de prétraitement des définitions
        """
        logger.info("Prétraitement linguistique des définitions...")
        start_time = time.time()
        
        # Conserver les définitions originales
        df['original_definition'] = df[definition_column].copy()
        
        # Nettoyage basique des définitions
        df['clean_definition'] = df[definition_column].str.lower() if pd.api.types.is_string_dtype(df[definition_column]) else df[definition_column].astype(str).str.lower()
        df['clean_definition'] = df['clean_definition'].apply(lambda x: re.sub(r'[^\w\s]', ' ', str(x)) if pd.notna(x) else "")
        df['clean_definition'] = df['clean_definition'].apply(lambda x: re.sub(r'\s+', ' ', x).strip())
        
        # Analyse linguistique avec spaCy pour les définitions
        logger.info("Analyse linguistique des définitions avec spaCy...")
        
        # Utiliser un traitement par lots pour plus d'efficacité
        batch_size = 100
        all_results = []
        
        # Traiter les définitions par lots
        for i in range(0, len(df), batch_size):
            batch = df['clean_definition'].iloc[i:i+batch_size].tolist()
            # Traiter les définitions avec spaCy
            docs = list(self.nlp.pipe(batch))
            
            # Extraire les informations linguistiques
            batch_results = []
            for doc in docs:
                # Extraire les lemmes (formes de base des mots)
                lemmas = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
                
                # Extraire les mots-clés (noms, adjectifs, verbes significatifs)
                keywords = [token.text for token in doc if token.pos_ in ["NOUN", "ADJ", "VERB"] and not token.is_stop]
                
                # Identifier les termes d'entité nommée spécifiques au domaine musical
                music_entities = [token.text for token in doc if any(music_term in token.text.lower() 
                                                                for music_term in ["baroque", "musique", "instrument", 
                                                                                "note", "composition", "harmonie", 
                                                                                "mélodie", "rythme", "accord"])]
                
                batch_results.append({
                    "def_lemma": " ".join(lemmas),
                    "def_keywords": keywords,
                    "def_music_entities": music_entities,
                    "def_length": len(doc)
                })
            
            all_results.extend(batch_results)
        
        # Ajouter les résultats au DataFrame
        df['def_lemmatized'] = [result["def_lemma"] for result in all_results]
        df['def_keywords'] = [result["def_keywords"] for result in all_results]
        df['def_music_entities'] = [result["def_music_entities"] for result in all_results]
        df['def_length'] = [result["def_length"] for result in all_results]
        
        elapsed_time = time.time() - start_time
        logger.info(f"Prétraitement des définitions terminé en {elapsed_time:.2f} secondes")
        
        return df

    def generate_embeddings(self) -> np.ndarray:
        """
        Génère des représentations vectorielles (embeddings) pour tous les termes,
        en incorporant les définitions lorsque disponibles.
        
        Cette fonction utilise un modèle SentenceTransformer ou TF-IDF pour créer des vecteurs
        représentant la sémantique des termes et des définitions. Les embeddings sont combinés
        avec une pondération configurable pour refléter l'importance relative des termes et des définitions.
        
        Returns:
            np.ndarray: Matrice des embeddings combinés (n_terms × embedding_dim)
                
        Raises:
            ValueError: Si aucune donnée n'a été chargée auparavant
        """
        logger.info("Génération des embeddings vectoriels...")
        start_time = time.time()
        
        if self.df is None:
            raise ValueError("Aucune donnée chargée. Utilisez d'abord la méthode load_data().")
        
        # Utiliser les termes nettoyés pour les embeddings
        terms = self.df['clean_term'].tolist()
        
        # Facteur de pondération pour combiner les embeddings des termes et des définitions
        # Valeur entre 0 et 1, où 1 signifie "uniquement les termes" et 0 "uniquement les définitions"
        term_weight = 0.6  # 60% terme, 40% définition
        
        try:
            # Générer les embeddings pour les termes
            if self.embedding_model is not None:
                logger.info("Utilisation du modèle SentenceTransformer pour les embeddings")
                term_embeddings = self.embedding_model.encode(
                    terms, 
                    show_progress_bar=True,
                    batch_size=32,
                    convert_to_numpy=True
                )
                logger.info(f"Embeddings des termes générés. Dimension: {term_embeddings.shape}")
                
                # Si des définitions sont disponibles, générer leurs embeddings
                if self.definition_column is not None:
                    logger.info("Génération des embeddings pour les définitions...")
                    definitions = self.df['clean_definition'].tolist()
                    
                    definition_embeddings = self.embedding_model.encode(
                        definitions,
                        show_progress_bar=True,
                        batch_size=32,
                        convert_to_numpy=True
                    )
                    logger.info(f"Embeddings des définitions générés. Dimension: {definition_embeddings.shape}")
                    
                    # Combiner les embeddings avec la pondération spécifiée
                    logger.info(f"Combinaison des embeddings (pondération: {term_weight:.2f} terme, {1-term_weight:.2f} définition)")
                    self.embeddings = term_weight * term_embeddings + (1 - term_weight) * definition_embeddings
                    logger.info(f"Embeddings combinés générés. Dimension: {self.embeddings.shape}")
                else:
                    # Utiliser uniquement les embeddings des termes si pas de définitions
                    self.embeddings = term_embeddings
                    
            else:
                # Solution de repli: utiliser TF-IDF
                logger.info("Utilisation de TF-IDF pour les embeddings (solution de repli)")
                
                if self.definition_column is not None:
                    # Combiner les termes et les définitions pour TF-IDF
                    combined_texts = [
                        f"{term} {definition}" 
                        if definition else term
                        for term, definition in zip(terms, self.df['clean_definition'].tolist())
                    ]
                    
                    vectorizer = TfidfVectorizer(
                        max_features=300,
                        ngram_range=(1, 2),
                        min_df=2,
                        max_df=0.9
                    )
                    self.embeddings = vectorizer.fit_transform(combined_texts).toarray()
                else:
                    # TF-IDF uniquement sur les termes
                    vectorizer = TfidfVectorizer(
                        max_features=300,
                        ngram_range=(1, 2),
                        min_df=2,
                        max_df=0.9
                    )
                    self.embeddings = vectorizer.fit_transform(terms).toarray()
                    
                logger.info(f"Embeddings TF-IDF générés. Dimension: {self.embeddings.shape}")
            
            # Calculer la matrice de similarité entre tous les termes
            logger.info("Calcul de la matrice de similarité...")
            self.similarity_matrix = cosine_similarity(self.embeddings)
            
            # Créer une visualisation des embeddings
            self._visualize_embeddings()
            
            elapsed_time = time.time() - start_time
            logger.info(f"Génération des embeddings terminée en {elapsed_time:.2f} secondes")
            
            return self.embeddings
                
        except Exception as e:
            logger.error(f"Erreur lors de la génération des embeddings: {e}")
            
            # Solution de secours: utiliser TF-IDF si tout échoue
            logger.info("Utilisation de TF-IDF comme alternative après erreur")
            
            if self.definition_column is not None:
                # Combiner les termes et définitions
                combined_texts = [
                    f"{term} {definition}" 
                    if definition else term
                    for term, definition in zip(terms, self.df['clean_definition'].tolist())
                ]
                vectorizer = TfidfVectorizer(max_features=100)
                self.embeddings = vectorizer.fit_transform(combined_texts).toarray()
            else:
                vectorizer = TfidfVectorizer(max_features=100)
                self.embeddings = vectorizer.fit_transform(terms).toarray()
                
            logger.info(f"Embeddings TF-IDF générés comme solution de secours. Dimension: {self.embeddings.shape}")
            
            # Matrice de similarité de secours
            self.similarity_matrix = cosine_similarity(self.embeddings)
            
            return self.embeddings

    def _visualize_embeddings(self) -> None:
        """
        Crée et sauvegarde une visualisation 2D des embeddings des termes.
        
        Cette fonction:
        1. Réduit la dimensionnalité des embeddings à 2D avec t-SNE
        2. Crée un nuage de points interactif avec Plotly
        3. Sauvegarde la visualisation en HTML et PNG
        
        La visualisation aide à comprendre comment les termes sont organisés
        dans l'espace sémantique et peut révéler des clusters naturels.
        """
        logger.info("Visualisation des embeddings avec t-SNE...")
        
        # Réduire la dimensionnalité avec t-SNE pour la visualisation
        try:
            # Adapter la perplexité à la taille du jeu de données
            perplexity = min(30, max(5, len(self.df) // 10))
            
            tsne = TSNE(
                n_components=2,
                random_state=42,
                perplexity=perplexity,
                learning_rate='auto',
                init='pca',
                n_jobs=-1  # Utiliser tous les processeurs disponibles
            )
            embeddings_2d = tsne.fit_transform(self.embeddings)
            
            # Créer un DataFrame pour la visualisation
            viz_df = pd.DataFrame(embeddings_2d, columns=['x', 'y'])
            viz_df['term'] = self.df['original_term']
            viz_df['category'] = self.df['main_category'] if 'main_category' in self.df.columns else 'Non classifié'
            
            # Création d'une figure interactive avec Plotly
            fig = px.scatter(
                viz_df, x='x', y='y', 
                hover_name='term',
                color='category' if 'main_category' in self.df.columns else None,
                title=f"Visualisation des embeddings des termes ({self.domain})",
                labels={'x': 'Dimension 1', 'y': 'Dimension 2'},
                opacity=0.7
            )
            
            # Ajuster la mise en page
            fig.update_layout(
                legend_title_text='Catégorie',
                autosize=True,
                width=900,
                height=700,
                template="plotly_white",
                hovermode='closest'
            )
            
            # Sauvegarder la visualisation
            fig.write_html(self.output_dir / "embeddings_visualization.html")
            fig.write_image(self.output_dir / "embeddings_visualization.png")
            
            logger.info(f"Visualisation des embeddings enregistrée dans {self.output_dir}")
            
        except Exception as e:
            logger.warning(f"Erreur lors de la création de la visualisation des embeddings: {e}")
            
            # Créer une visualisation de secours avec matplotlib
            try:
                plt.figure(figsize=(10, 8))
                plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.5)
                plt.title("Visualisation des embeddings (t-SNE)")
                plt.xlabel("Dimension 1")
                plt.ylabel("Dimension 2")
                plt.tight_layout()
                plt.savefig(self.output_dir / "embeddings_visualization_fallback.png")
                logger.info("Visualisation de secours créée avec matplotlib")
            except:
                logger.warning("Impossible de créer une visualisation des embeddings")

    def classify_terms(self, n_clusters: Optional[int] = None) -> pd.DataFrame:
        """
        Classifie les termes en utilisant une approche hybride combinant:
        1. Classification basée sur les embeddings (apprentissage non supervisé)
        2. Classification basée sur des règles linguistiques
        3. Utilisation des catégories prédéfinies pour les termes spécifiques au domaine
        
        La fonction détermine automatiquement le nombre optimal de clusters si non spécifié.
        
        Parameters:
            n_clusters (int, optional): Nombre de clusters à former. Si None, déterminé automatiquement.
            
        Returns:
            pd.DataFrame: DataFrame enrichi avec les colonnes de classification
            
        Raises:
            ValueError: Si les embeddings n'ont pas été générés auparavant
        """
        logger.info("Classification des termes...")
        start_time = time.time()
        
        if self.embeddings is None:
            raise ValueError("Les embeddings n'ont pas été générés. Utilisez d'abord la méthode generate_embeddings().")
        
        # 1. Détermination du nombre optimal de clusters si non spécifié
        if n_clusters is None:
            n_clusters = self._determine_optimal_clusters()
            logger.info(f"Nombre optimal de clusters déterminé: {n_clusters}")
        
        # 2. Classification par clustering
        self._apply_clustering(n_clusters)
        
        # 3. Classification par analyse linguistique
        self._classify_by_linguistic_features()
        
        # 4. Fusion des classifications pour obtenir une classification finale
        self._merge_classifications()
        
        # 5. Analyse des résultats de classification
        self._analyze_classification_results()
        
        elapsed_time = time.time() - start_time
        logger.info(f"Classification terminée en {elapsed_time:.2f} secondes")
        
        return self.df

    def _determine_optimal_clusters(self, min_clusters: int = 3, max_clusters: int = 20) -> int:
        """
        Détermine le nombre optimal de clusters en utilisant plusieurs métriques.
        
        Cette méthode teste différentes valeurs de clusters et sélectionne la meilleure
        en se basant sur des métriques comme le score de silhouette ou l'indice de Calinski-Harabasz.
        
        Parameters:
            min_clusters (int): Nombre minimum de clusters à tester
            max_clusters (int): Nombre maximum de clusters à tester
            
        Returns:
            int: Nombre optimal de clusters
        """
        logger.info(f"Détermination du nombre optimal de clusters de {min_clusters} à {max_clusters}...")
        
        # Ajuster les valeurs min/max en fonction du nombre de termes
        n_terms = len(self.df)
        max_clusters = min(max_clusters, n_terms // 5)  # Éviter trop de clusters pour peu de termes
        min_clusters = min(min_clusters, max_clusters - 1)  # S'assurer que min < max
        
        if min_clusters <= 1:
            min_clusters = 2  # Au moins 2 clusters
        
        # Initialiser les métriques
        silhouette_scores = []
        ch_scores = []
        
        # Tester différentes valeurs de clusters
        for n_clusters in range(min_clusters, max_clusters + 1):
            try:
                # Utiliser k-means pour le clustering
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(self.embeddings)
                
                # Calculer le score de silhouette
                silhouette_avg = silhouette_score(self.embeddings, cluster_labels)
                silhouette_scores.append(silhouette_avg)
                
                # Calculer l'indice de Calinski-Harabasz
                ch_score = calinski_harabasz_score(self.embeddings, cluster_labels)
                ch_scores.append(ch_score)
                
                logger.debug(f"Clusters: {n_clusters}, Silhouette: {silhouette_avg:.4f}, CH: {ch_score:.1f}")
                
            except Exception as e:
                logger.warning(f"Erreur pour {n_clusters} clusters: {e}")
                silhouette_scores.append(-1)
                ch_scores.append(-1)
        
        # Normaliser les scores pour combiner les métriques
        if silhouette_scores and max(silhouette_scores) > 0:
            norm_silhouette = [s / max(silhouette_scores) for s in silhouette_scores]
        else:
            norm_silhouette = [0] * len(silhouette_scores)
        
        if ch_scores and max(ch_scores) > 0:
            norm_ch = [s / max(ch_scores) for s in ch_scores]
        else:
            norm_ch = [0] * len(ch_scores)
        
        # Combiner les métriques (moyenne pondérée)
        combined_scores = [0.7 * s + 0.3 * c for s, c in zip(norm_silhouette, norm_ch)]
        
        # Déterminer le nombre optimal de clusters
        best_index = combined_scores.index(max(combined_scores))
        optimal_clusters = best_index + min_clusters
        
        logger.info(f"Nombre optimal de clusters: {optimal_clusters}")
        
        # Sauvegarder un graphique des scores
        try:
            plt.figure(figsize=(10, 6))
            plt.plot(range(min_clusters, max_clusters + 1), silhouette_scores, 'o-', label='Silhouette')
            plt.plot(range(min_clusters, max_clusters + 1), norm_ch, 's-', label='Calinski-Harabasz (norm.)')
            plt.plot(range(min_clusters, max_clusters + 1), combined_scores, 'x-', label='Score combiné')
            plt.axvline(x=optimal_clusters, color='r', linestyle='--', label=f'Optimal: {optimal_clusters}')
            plt.xlabel('Nombre de clusters')
            plt.ylabel('Score')
            plt.title('Détermination du nombre optimal de clusters')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.savefig(self.output_dir / "optimal_clusters.png")
        except Exception as e:
            logger.warning(f"Erreur lors de la création du graphique des scores: {e}")
        
        return optimal_clusters

    def _apply_clustering(self, n_clusters: int) -> None:
        """
        Applique plusieurs algorithmes de clustering et combine leurs résultats.
        
        Cette méthode utilise:
        1. K-means pour un clustering basé sur la distance
        2. Clustering agglomératif hiérarchique pour capturer les relations hiérarchiques
        3. DBSCAN pour identifier des clusters de densité variable
        
        Les résultats sont ensuite combinés pour une classification plus robuste.
        
        Parameters:
            n_clusters (int): Nombre de clusters à former
        """
        logger.info(f"Application des algorithmes de clustering avec {n_clusters} clusters...")
        
        # 1. K-means
        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            kmeans_labels = kmeans.fit_predict(self.embeddings)
            self.df['kmeans_cluster'] = kmeans_labels
            logger.info(f"Clustering K-means appliqué: {n_clusters} clusters formés")
            
            # Ajouter la distance au centre du cluster (indicateur de représentativité)
            cluster_centers = kmeans.cluster_centers_
            self.df['kmeans_center_dist'] = [
                np.linalg.norm(self.embeddings[i] - cluster_centers[label])
                for i, label in enumerate(kmeans_labels)
            ]
        except Exception as e:
            logger.error(f"Erreur lors du clustering K-means: {e}")
            self.df['kmeans_cluster'] = -1
            self.df['kmeans_center_dist'] = np.nan
        
        # 2. Clustering hiérarchique
        try:
            agg_clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
                metric='euclidean',
                linkage='ward'
            )
            agg_labels = agg_clustering.fit_predict(self.embeddings)
            self.df['hierarchical_cluster'] = agg_labels
            logger.info("Clustering hiérarchique appliqué")
            
            # Visualisation du dendrogramme pour un sous-ensemble de données
            try:
                if len(self.df) <= 100:  # Limitez la visualisation à un nombre raisonnable de termes
                    plt.figure(figsize=(12, 8))
                    from scipy.cluster.hierarchy import linkage, dendrogram
                    Z = linkage(self.embeddings, method='ward')
                    dendrogram(Z, labels=self.df['original_term'].tolist(), leaf_font_size=8)
                    plt.title("Dendrogramme du clustering hiérarchique")
                    plt.tight_layout()
                    plt.savefig(self.output_dir / "hierarchical_clustering.png")
                    logger.info("Dendrogramme du clustering hiérarchique sauvegardé")
            except Exception as e:
                logger.warning(f"Impossible de générer le dendrogramme: {e}")
        except Exception as e:
            logger.error(f"Erreur lors du clustering hiérarchique: {e}")
            self.df['hierarchical_cluster'] = -1
        
        # 3. DBSCAN pour détecter des clusters de densité variable
        try:
            # Estimer epsilon (distance maximale entre deux points dans un même cluster)
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=min(10, len(self.df) - 1))
            nn.fit(self.embeddings)
            distances, _ = nn.kneighbors(self.embeddings)
            epsilon = np.percentile(distances[:, -1], 75)  # 75e percentile des distances
            
            dbscan = DBSCAN(eps=epsilon, min_samples=3)
            dbscan_labels = dbscan.fit_predict(self.embeddings)
            self.df['dbscan_cluster'] = dbscan_labels
            
            # Compter le nombre de clusters formés (en excluant le bruit = -1)
            n_dbscan_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
            logger.info(f"DBSCAN appliqué: {n_dbscan_clusters} clusters formés")
            
            # Identifier les termes considérés comme du bruit (outliers)
            n_outliers = sum(1 for l in dbscan_labels if l == -1)
            self.df['is_outlier'] = dbscan_labels == -1
            logger.info(f"DBSCAN a identifié {n_outliers} termes comme outliers ({n_outliers/len(self.df)*100:.1f}%)")
        except Exception as e:
            logger.error(f"Erreur lors du clustering DBSCAN: {e}")
            self.df['dbscan_cluster'] = -1
            self.df['is_outlier'] = False
        
        # 4. Détection de communautés dans le graphe de similarité
        try:
            # Créer un graphe de similarité où les arêtes sont pondérées par la similarité
            similarity_threshold = 0.5  # Ne conserver que les similarités significatives
            G = nx.Graph()
            
            # Ajouter les nœuds
            for i in range(len(self.df)):
                G.add_node(i, label=self.df.iloc[i]['original_term'])
            
            # Ajouter les arêtes pondérées par similarité
            for i in range(len(self.df)):
                for j in range(i+1, len(self.df)):
                    sim = self.similarity_matrix[i, j]
                    if sim > similarity_threshold:
                        G.add_edge(i, j, weight=sim)
            
            # Appliquer l'algorithme de Louvain pour la détection de communautés
            communities = community_louvain.best_partition(G)
            self.df['community'] = [communities.get(i, -1) for i in range(len(self.df))]
            
            n_communities = len(set(communities.values()))
            logger.info(f"Détection de communautés: {n_communities} communautés identifiées")
            
            # Visualiser le graphe de communautés
            try:
                plt.figure(figsize=(12, 12))
                pos = nx.spring_layout(G, k=0.3, iterations=50)
                
                # Couleurs pour les communautés
                cmap = plt.cm.get_cmap("tab20", n_communities)
                
                # Dessiner les nœuds colorés par communauté
                for comm_id in set(communities.values()):
                    list_nodes = [node for node in G.nodes() if communities[node] == comm_id]
                    nx.draw_networkx_nodes(
                        G, pos, nodelist=list_nodes,
                        node_color=[cmap(comm_id)],
                        node_size=100, alpha=0.8
                    )
                
                # Dessiner les arêtes avec une transparence basée sur le poids
                edge_colors = [G[u][v]['weight'] for u, v in G.edges()]
                nx.draw_networkx_edges(G, pos, alpha=0.2, width=1, edge_color=edge_colors, edge_cmap=plt.cm.Blues)
                
                # Ajouter des étiquettes pour les nœuds principaux (limiter pour éviter l'encombrement)
                # Trouver les nœuds les plus centraux dans chaque communauté
                central_nodes = {}
                for comm_id in set(communities.values()):
                    comm_nodes = [node for node in G.nodes() if communities[node] == comm_id]
                    if comm_nodes:
                        # Utiliser la centralité de degré comme mesure
                        node_centrality = {node: nx.degree_centrality(G)[node] for node in comm_nodes}
                        most_central = max(node_centrality, key=node_centrality.get)
                        central_nodes[most_central] = G.nodes[most_central]['label']
                
                nx.draw_networkx_labels(G, pos, labels=central_nodes, font_size=8)
                
                plt.title("Graphe des communautés de termes")
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(self.output_dir / "community_graph.png", dpi=150)
                logger.info("Graphe des communautés sauvegardé")
            except Exception as e:
                logger.warning(f"Impossible de générer le graphe des communautés: {e}")
        except Exception as e:
            logger.error(f"Erreur lors de la détection de communautés: {e}")
            self.df['community'] = -1

    def _classify_by_linguistic_features(self) -> None:
        """
        Classifie les termes en utilisant des caractéristiques linguistiques extraites.
        
        Cette méthode analyse la structure grammaticale, les parties du discours,
        et d'autres caractéristiques linguistiques pour enrichir la classification.
        """
        logger.info("Classification basée sur l'analyse linguistique...")
        
        # 1. Détecter les catégories grammaticales dominantes
        pos_categories = {
            'entité': ['PROPN', 'PERSON', 'ORG', 'LOC'],
            'objet': ['NOUN'],
            'action': ['VERB'],
            'qualité': ['ADJ'],
            'concept': ['NOUN', 'ADJ']
        }
        
        # Initialiser la colonne pour la catégorie linguistique
        self.df['linguistic_category'] = None
        
        # Appliquer la classification linguistique
        for idx, row in self.df.iterrows():
            pos_tags = row['pos_tags'].split()
            
            # Détecter la catégorie en fonction des POS tags
            for category, pos_list in pos_categories.items():
                if any(pos in pos_tags for pos in pos_list):
                    self.df.at[idx, 'linguistic_category'] = category
                    break
            
            # Classification par défaut
            if self.df.at[idx, 'linguistic_category'] is None:
                self.df.at[idx, 'linguistic_category'] = 'concept'
        
        # 2. Identifier les relations potentielles basées sur la structure grammaticale
        # Par exemple, "Sonate pour violon" -> relation entre "sonate" et "violon"
        self.df['related_terms'] = None
        
        for idx, row in self.df.iterrows():
            # Analyser la structure pour identifier les relations
            term_doc = self.nlp(row['clean_term'])
            related = []
            
            for token in term_doc:
                # Chercher des modifieurs ou compléments
                if token.dep_ in ['prep', 'pobj', 'nmod', 'amod']:
                    # Trouver le terme principal auquel ce token est relié
                    if token.head.text != token.text:
                        related.append((token.head.text, token.text, token.dep_))
            
            if related:
                self.df.at[idx, 'related_terms'] = related
        
        logger.info("Classification linguistique terminée")

    def _merge_classifications(self) -> None:
        """
        Fusionne les différentes classifications pour obtenir une classification finale.
        
        Cette méthode combine:
        - La classification basée sur le domaine
        - Les résultats des différents algorithmes de clustering
        - La classification linguistique
        
        pour déterminer la catégorie principale et secondaire de chaque terme.
        """
        logger.info("Fusion des différentes classifications...")
        
        # 1. Création de la colonne pour la classification finale
        self.df['main_category'] = None
        self.df['secondary_category'] = None
        self.df['category_confidence'] = 0.0
        
        # 2. Utiliser kmeans puis la communauté
        for idx, row in self.df.iterrows():
            if pd.isna(row['main_category']):
                if 'community' in self.df.columns and row['community'] != -1:
                    community_id = row['community']
                    
                    # Trouver le nom de la catégorie le plus fréquent dans cette communauté
                    community_terms = self.df[self.df['community'] == community_id]
                    
                    # Si pas de nom de catégorie disponible, utiliser "Groupe X"
                    self.df.at[idx, 'main_category'] = f"Groupe {community_id}"
                    self.df.at[idx, 'category_confidence'] = 0.4
                
                # Utiliser kmeans si communauté non disponible
                elif 'kmeans_cluster' in self.df.columns and row['kmeans_cluster'] != -1:
                    cluster_id = row['kmeans_cluster']
                    
                    # Trouver le nom de la catégorie le plus fréquent dans ce cluster
                    cluster_terms = self.df[self.df['kmeans_cluster'] == cluster_id]
                    
                    self.df.at[idx, 'main_category'] = f"Cluster {cluster_id}"
                    self.df.at[idx, 'category_confidence'] = 0.3
                
                # En dernier recours, utiliser la catégorie linguistique
                else:
                    self.df.at[idx, 'main_category'] = row['linguistic_category'].capitalize() if pd.notna(row['linguistic_category']) else "Non classifié"
                    self.df.at[idx, 'category_confidence'] = 0.2
        
        # 3. Déterminer les catégories secondaires
        for idx, row in self.df.iterrows():
            # Sinon, utiliser la catégorie linguistique si différente
            if pd.notna(row['linguistic_category']) and row['linguistic_category'].capitalize() != row['main_category']:
                self.df.at[idx, 'secondary_category'] = row['linguistic_category'].capitalize()
        
        # 4. Calculer des statistiques sur la classification finale
        cat_counts = self.df['main_category'].value_counts()
        main_categories = cat_counts.index.tolist()
        n_categories = len(main_categories)
        
        logger.info(f"Classification finale: {n_categories} catégories principales identifiées")
        logger.info(f"Top 5 catégories: {', '.join(main_categories[:5] if len(main_categories) >= 5 else main_categories)}")
        
        # 5. Créer une visualisation des catégories principales
        try:
            plt.figure(figsize=(12, 8))
            cat_counts_visual = cat_counts.head(15)  # Limiter aux 15 plus grandes catégories
            cat_counts_visual.plot(kind='bar')
            plt.title("Distribution des catégories principales")
            plt.xlabel("Catégorie")
            plt.ylabel("Nombre de termes")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(self.output_dir / "main_categories_distribution.png")
        except Exception as e:
            logger.warning(f"Impossible de créer la visualisation des catégories: {e}")

    def _analyze_classification_results(self) -> None:
        """
        Analyse les résultats de la classification pour évaluer sa qualité.
        
        Cette méthode calcule diverses métriques de qualité et les enregistre
        pour aider à évaluer la performance de la classification.
        """
        logger.info("Analyse des résultats de la classification...")
        
        metrics = {}
        
        # 1. Calculer la distribution des tailles de catégories
        cat_counts = self.df['main_category'].value_counts()
        metrics['n_categories'] = len(cat_counts)
        metrics['max_category_size'] = cat_counts.max()
        metrics['min_category_size'] = cat_counts.min()
        metrics['avg_category_size'] = cat_counts.mean()
        metrics['category_size_std'] = cat_counts.std()
        
        # 2. Calculer la cohérence intra-catégorie
        intra_category_similarity = {}
        
        for category in self.df['main_category'].unique():
            # Indices des termes dans cette catégorie
            indices = self.df[self.df['main_category'] == category].index.tolist()
            
            if len(indices) >= 2:  # Besoin d'au moins 2 termes pour calculer la similarité
                # Extraire les paires de similarité pour cette catégorie
                similarities = []
                for i in range(len(indices)):
                    for j in range(i+1, len(indices)):
                        similarities.append(self.similarity_matrix[indices[i], indices[j]])
                
                intra_category_similarity[category] = np.mean(similarities)
        
        # Moyenne de la cohérence pour toutes les catégories
        if intra_category_similarity:
            metrics['avg_intra_category_similarity'] = np.mean(list(intra_category_similarity.values()))
            metrics['max_intra_category_similarity'] = np.max(list(intra_category_similarity.values()))
            metrics['min_intra_category_similarity'] = np.min(list(intra_category_similarity.values()))
        
        # 3. Calculer la séparation inter-catégories
        inter_category_similarity = []
        
        categories = self.df['main_category'].unique()
        for i in range(len(categories)):
            cat1 = categories[i]
            indices1 = self.df[self.df['main_category'] == cat1].index.tolist()
            
            for j in range(i+1, len(categories)):
                cat2 = categories[j]
                indices2 = self.df[self.df['main_category'] == cat2].index.tolist()
                
                # Calculer toutes les similarités entre les termes des deux catégories
                similarities = []
                for idx1 in indices1:
                    for idx2 in indices2:
                        similarities.append(self.similarity_matrix[idx1, idx2])
                
                if similarities:
                    inter_category_similarity.append(np.mean(similarities))
        
        if inter_category_similarity:
            metrics['avg_inter_category_similarity'] = np.mean(inter_category_similarity)
        
        # 4. Calculer une mesure de qualité de classification (ratio intra/inter)
        if 'avg_intra_category_similarity' in metrics and 'avg_inter_category_similarity' in metrics:
            metrics['classification_quality'] = metrics['avg_intra_category_similarity'] / metrics['avg_inter_category_similarity']
        
        # 5. Distribution de la confiance de classification
        metrics['avg_confidence'] = self.df['category_confidence'].mean()
        metrics['high_confidence'] = (self.df['category_confidence'] > 0.7).mean()
        
        # 6. Calculer le pourcentage de termes avec catégorie secondaire
        metrics['secondary_category_ratio'] = self.df['secondary_category'].notna().mean()
        
        # Enregistrer les métriques
        self.metrics['classification'] = metrics
        
        # Journaliser les principales métriques
        logger.info(f"Nombre de catégories: {metrics['n_categories']}")
        logger.info(f"Taille moyenne des catégories: {metrics['avg_category_size']:.2f} termes")
        if 'classification_quality' in metrics:
            logger.info(f"Qualité de la classification (ratio intra/inter): {metrics['classification_quality']:.2f}")
        logger.info(f"Confiance moyenne de classification: {metrics['avg_confidence']:.2f}")
        
        # Sauvegarder les métriques
        self.metrics['classification'] = metrics
        
        # Journaliser les principales métriques
        logger.info(f"Nombre de catégories: {metrics['n_categories']}")
        logger.info(f"Taille moyenne des catégories: {metrics['avg_category_size']:.2f} termes")
        if 'classification_quality' in metrics:
            logger.info(f"Qualité de la classification (ratio intra/inter): {metrics['classification_quality']:.2f}")
        logger.info(f"Confiance moyenne de classification: {metrics['avg_confidence']:.2f}")
        
        # Sauvegarder les métriques dans un fichier JSON
        try:
            metrics_path = self.output_dir / "classification_metrics.json"
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=4, ensure_ascii=False, default=convert_numpy)
            logger.info(f"Métriques de classification sauvegardées dans {metrics_path}")
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde des métriques: {e}")

    def export_skos(self) -> None:
        """
        Exporte la hiérarchie construite au format SKOS enrichi (RDF/Turtle et RDF/XML).
        
        Cette méthode utilise toutes les colonnes disponibles du CSV original :
        prefLabel, altLabel, hiddenLabel, definition, scopeNote, changeNote, ref:type.
        """
        logger.info("Exportation enrichie du thésaurus au format SKOS...")
        
        try:
            # Créer un graphe RDF
            g = Graph()
            g.bind("skos", SKOS)
            g.bind("rdfs", RDFS)
            g.bind("cmbv", self.namespace)
            
            # Créer un ConceptScheme
            scheme_uri = self.namespace[f"{self.domain}_scheme"]
            g.add((scheme_uri, RDF.type, SKOS.ConceptScheme))
            g.add((scheme_uri, RDFS.label, Literal(self.name, lang=self.language)))
            g.add((scheme_uri, RDFS.comment, Literal(self.description, lang=self.language)))
            
            for idx, row in self.df.iterrows():
                concept_uri = self.namespace[row['conceptID']]
                
                g.add((concept_uri, RDF.type, SKOS.Concept))
                g.add((concept_uri, SKOS.prefLabel, Literal(row.get('skos:prefLabel', row['original_term']), lang=self.language)))
                g.add((concept_uri, SKOS.inScheme, scheme_uri))
                
                # Ajouter les altLabels si présents
                if pd.notna(row.get('skos:altLabel')):
                    alt_labels = row['skos:altLabel']
                    if isinstance(alt_labels, str):
                        for alt in re.split(r'##', alt_labels):
                            alt = alt.strip()
                            if alt:
                                g.add((concept_uri, SKOS.altLabel, Literal(alt, lang=self.language)))
                
                # Ajouter les hiddenLabels si présents
                if pd.notna(row.get('skos:hiddenLabel')):
                    hidden_labels = row['skos:hiddenLabel']
                    if isinstance(hidden_labels, str):
                        for hidden in re.split(r'##', hidden_labels):
                            hidden = hidden.strip()
                            if hidden:
                                g.add((concept_uri, SKOS.hiddenLabel, Literal(hidden, lang=self.language)))
                
                # Ajouter la définition si présente
                if pd.notna(row.get('skos:definition')):
                    g.add((concept_uri, SKOS.definition, Literal(row['skos:definition'], lang=self.language)))
                
                # Ajouter la scopeNote si présente
                if pd.notna(row.get('skos:scopeNote')):
                    g.add((concept_uri, SKOS.scopeNote, Literal(row['skos:scopeNote'], lang=self.language)))
                
                # Ajouter la changeNote si présente
                if pd.notna(row.get('skos:changeNote')):
                    g.add((concept_uri, SKOS.changeNote, Literal(row['skos:changeNote'], lang=self.language)))
                
                # Ajouter ref:type comme type spécifique si souhaité (facultatif)
                if pd.notna(row.get('ref:type')):
                    type_uri = self.namespace[re.sub(r'[^\w]', '_', row['ref:type'].lower())]
                    g.add((concept_uri, RDF.type, type_uri))
                
                # Ajouter les relations broader vers la catégorie principale
                if pd.notna(row.get('main_category')):
                    broader_concept = self.namespace[
                        re.sub(r'[^\w]', '_', row['main_category'].lower())[:30]
                    ]
                    g.add((concept_uri, SKOS.broader, broader_concept))
                    
                    # Définir aussi le concept de la catégorie s'il n'existe pas encore
                    g.add((broader_concept, RDF.type, SKOS.Concept))
                    g.add((broader_concept, SKOS.prefLabel, Literal(row['main_category'], lang=self.language)))
                    g.add((broader_concept, SKOS.inScheme, scheme_uri))
            
            # Sauvegarder en Turtle (.ttl)
            ttl_path = self.output_dir / "thesaurus_export.ttl"
            g.serialize(destination=str(ttl_path), format='turtle')
            logger.info(f"Thésaurus exporté au format Turtle: {ttl_path}")
            
            # Sauvegarder aussi en RDF/XML (.rdf)
            rdf_path = self.output_dir / "thesaurus_export.rdf"
            g.serialize(destination=str(rdf_path), format='xml')
            logger.info(f"Thésaurus exporté au format RDF/XML: {rdf_path}")
        
        except Exception as e:
            logger.error(f"Erreur lors de l'exportation enrichie SKOS: {e}")

    def generate_report(self) -> None:
        """
        Génère un rapport HTML interactif contenant:
        - Statistiques de classification
        - Graphes de distribution
        - Résumé des métriques
        - Liens vers les visualisations générées
        """
        try:
            report_path = self.output_dir / "report.html"
            with open(report_path, "w", encoding="utf-8") as f:
                f.write("<html><head><meta charset='utf-8'><title>Rapport Thésaurus</title></head><body>")
                f.write(f"<h1>Rapport du thésaurus : {self.name}</h1>")
                f.write(f"<h2>Domaine : {self.domain}</h2>")
                f.write(f"<h3>Langue : {self.language}</h3>")
                f.write("<hr>")
                
                # Ajouter les métriques de classification
                if self.metrics.get('classification'):
                    f.write("<h2>Métriques de classification</h2>")
                    f.write("<ul>")
                    for key, value in self.metrics['classification'].items():
                        f.write(f"<li><b>{key}</b> : {value}</li>")
                    f.write("</ul>")
                
                # Ajouter les visualisations
                f.write("<h2>Visualisations</h2>")
                if (self.output_dir / "embeddings_visualization.html").exists():
                    f.write(f"<p><a href='embeddings_visualization.html'>Visualisation des embeddings</a></p>")
                if (self.output_dir / "optimal_clusters.png").exists():
                    f.write(f"<p><img src='optimal_clusters.png' width='600'></p>")
                if (self.output_dir / "main_categories_distribution.png").exists():
                    f.write(f"<p><img src='main_categories_distribution.png' width='600'></p>")
                if (self.output_dir / "community_graph.png").exists():
                    f.write(f"<p><img src='community_graph.png' width='600'></p>")
                
                f.write("</body></html>")
            
            logger.info(f"Rapport HTML généré : {report_path}")
        except Exception as e:
            logger.error(f"Erreur lors de la génération du rapport : {e}")