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
        2. Classification basée sur des règles linguistiques des termes ET de leurs définitions
        3. Détection de hiérarchie par analyse de relations sémantiques
        4. Utilisation des catégories prédéfinies pour les termes musicaux baroques
        
        La fonction détermine automatiquement le nombre optimal de clusters si non spécifié.
        
        Parameters:
            n_clusters (int, optional): Nombre de clusters à former. Si None, déterminé automatiquement.
            
        Returns:
            pd.DataFrame: DataFrame enrichi avec les colonnes de classification
            
        Raises:
            ValueError: Si les embeddings n'ont pas été générés auparavant
        """
        logger.info("Classification des termes avec intégration des définitions...")
        start_time = time.time()
        
        if self.embeddings is None:
            raise ValueError("Les embeddings n'ont pas été générés. Utilisez d'abord la méthode generate_embeddings().")
        
        # 1. Détermination du nombre optimal de clusters si non spécifié
        if n_clusters is None:
            n_clusters = self._determine_optimal_clusters()
            logger.info(f"Nombre optimal de clusters déterminé: {n_clusters}")
        
        # 2. Classification par clustering avancé
        self._apply_clustering(n_clusters)
        
        # 3. Classification par analyse linguistique des termes et définitions
        self._classify_by_linguistic_features()
        
        # 4. Détection des relations hiérarchiques
        self._detect_hierarchical_relations()
        
        # 5. Fusion des classifications pour obtenir une classification finale
        self._merge_classifications()
        
        # 6. Analyse des résultats de classification
        self._analyze_classification_results()
        
        elapsed_time = time.time() - start_time
        logger.info(f"Classification terminée en {elapsed_time:.2f} secondes")
        
        return self.df

    def _determine_optimal_clusters(self, min_clusters: int = 3, max_clusters: int = 25) -> int:
        """
        Détermine le nombre optimal de clusters en utilisant plusieurs métriques.
        
        Cette méthode teste différentes valeurs de clusters et sélectionne la meilleure
        en se basant sur des métriques comme le score de silhouette, l'indice de Calinski-Harabasz,
        et l'indice de Davies-Bouldin.
        
        Parameters:
            min_clusters (int): Nombre minimum de clusters à tester
            max_clusters (int): Nombre maximum de clusters à tester
            
        Returns:
            int: Nombre optimal de clusters
        """
        logger.info(f"Détermination du nombre optimal de clusters de {min_clusters} à {max_clusters}...")
        
        # Ajuster les valeurs min/max en fonction du nombre de termes
        n_terms = len(self.df)
        max_clusters = min(max_clusters, n_terms // 4)  # Éviter trop de clusters pour peu de termes
        min_clusters = min(min_clusters, max_clusters - 1)  # S'assurer que min < max
        
        if min_clusters <= 1:
            min_clusters = 2  # Au moins 2 clusters
        
        # Initialiser les métriques
        silhouette_scores = []
        ch_scores = []
        db_scores = []  # Davies-Bouldin index (plus petit = meilleur)
        
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
                
                # Calculer l'indice Davies-Bouldin
                db_score = davies_bouldin_score(self.embeddings, cluster_labels)
                db_scores.append(db_score)
                
                logger.debug(f"Clusters: {n_clusters}, Silhouette: {silhouette_avg:.4f}, CH: {ch_score:.1f}, DB: {db_score:.4f}")
                
            except Exception as e:
                logger.warning(f"Erreur pour {n_clusters} clusters: {e}")
                silhouette_scores.append(-1)
                ch_scores.append(-1)
                db_scores.append(float('inf'))
        
        # Normaliser les scores pour combiner les métriques
        if silhouette_scores and max(silhouette_scores) > 0:
            norm_silhouette = [s / max(silhouette_scores) for s in silhouette_scores]
        else:
            norm_silhouette = [0] * len(silhouette_scores)
        
        if ch_scores and max(ch_scores) > 0:
            norm_ch = [s / max(ch_scores) for s in ch_scores]
        else:
            norm_ch = [0] * len(ch_scores)
        
        # Pour Davies-Bouldin, plus petit est meilleur, donc normaliser différemment
        if db_scores and min(db_scores) > 0:
            # Inverser pour que plus grand = meilleur
            norm_db = [1 - (s / max(db_scores)) if s != float('inf') else 0 for s in db_scores]
        else:
            norm_db = [0] * len(db_scores)
        
        # Combiner les métriques (moyenne pondérée)
        combined_scores = [0.5 * s + 0.3 * c + 0.2 * d for s, c, d in zip(norm_silhouette, norm_ch, norm_db)]
        
        # Déterminer le nombre optimal de clusters
        best_index = combined_scores.index(max(combined_scores))
        optimal_clusters = best_index + min_clusters
        
        logger.info(f"Nombre optimal de clusters: {optimal_clusters}")
        
        # Sauvegarder un graphique des scores
        try:
            plt.figure(figsize=(12, 8))
            
            # Subplot pour Silhouette et CH
            plt.subplot(2, 1, 1)
            plt.plot(range(min_clusters, max_clusters + 1), silhouette_scores, 'o-', label='Silhouette')
            plt.plot(range(min_clusters, max_clusters + 1), norm_ch, 's-', label='Calinski-Harabasz (norm.)')
            plt.axvline(x=optimal_clusters, color='r', linestyle='--', label=f'Optimal: {optimal_clusters}')
            plt.xlabel('Nombre de clusters')
            plt.ylabel('Score')
            plt.title('Scores de silhouette et Calinski-Harabasz')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Subplot pour Davies-Bouldin et score combiné
            plt.subplot(2, 1, 2)
            plt.plot(range(min_clusters, max_clusters + 1), norm_db, 'd-', label='Davies-Bouldin (norm. inversée)')
            plt.plot(range(min_clusters, max_clusters + 1), combined_scores, 'x-', label='Score combiné')
            plt.axvline(x=optimal_clusters, color='r', linestyle='--', label=f'Optimal: {optimal_clusters}')
            plt.xlabel('Nombre de clusters')
            plt.ylabel('Score')
            plt.title('Score Davies-Bouldin et score combiné')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()
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
        4. Détection de communautés dans le graphe de similarité
        5. HDBSCAN pour une meilleure détection hiérarchique
        
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
            
            # Identifier les termes les plus représentatifs de chaque cluster (les plus proches du centre)
            representative_terms = {}
            for cluster_id in range(n_clusters):
                cluster_indices = np.where(kmeans_labels == cluster_id)[0]
                if len(cluster_indices) > 0:
                    distances = [self.df.iloc[i]['kmeans_center_dist'] for i in cluster_indices]
                    closest_idx = cluster_indices[np.argmin(distances)]
                    representative_terms[cluster_id] = self.df.iloc[closest_idx]['original_term']
            
            # Ajouter une colonne avec le terme représentatif du cluster
            self.df['kmeans_representative'] = self.df['kmeans_cluster'].map(representative_terms)
            
        except Exception as e:
            logger.error(f"Erreur lors du clustering K-means: {e}")
            self.df['kmeans_cluster'] = -1
            self.df['kmeans_center_dist'] = np.nan
            self.df['kmeans_representative'] = None
        
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
            
            # Tenter d'obtenir les distances de fusion pour identifier la hiérarchie
            try:
                from scipy.cluster.hierarchy import linkage
                Z = linkage(self.embeddings, method='ward')
                
                # Sauvegarder la matrice de distances pour l'analyse hiérarchique
                self.hierarchical_linkage = Z
                
                # Extraire des niveaux hiérarchiques approximatifs (1 à 3 niveaux)
                # Plus la distance de fusion est grande, plus le niveau est élevé
                heights = Z[:, 2]  # hauteurs de fusion
                thresholds = np.percentile(heights, [33, 66, 100])
                
                # Appliquer les seuils pour obtenir différents niveaux hiérarchiques
                hierarchy_levels = []
                
                for i in range(len(self.df)):
                    # Analyser à quel niveau hiérarchique ce terme appartient
                    # en utilisant les distances de fusion
                    if 'hierarchical_cluster' in self.df.columns:
                        cluster_id = self.df.iloc[i]['hierarchical_cluster']
                        
                        # Logique simplifiée pour déterminer le niveau hiérarchique
                        # dans un contexte musical (à affiner selon domaine)
                        term_complexity = self.df.iloc[i]['term_complexity'] if 'term_complexity' in self.df.columns else None
                        n_tokens = self.df.iloc[i]['n_tokens'] if 'n_tokens' in self.df.columns else 0
                        
                        # Utiliser la complexité du terme comme indicateur du niveau hiérarchique
                        if term_complexity == 'simple' or n_tokens == 1:
                            hierarchy_level = 1  # Termes simples = concepts de haut niveau
                        elif term_complexity in ['composé', 'expression'] or n_tokens == 2:
                            hierarchy_level = 2  # Termes composés = niveau intermédiaire
                        else:
                            hierarchy_level = 3  # Termes complexes = concepts spécifiques
                        
                        hierarchy_levels.append(hierarchy_level)
                    else:
                        hierarchy_levels.append(0)
                
                self.df['hierarchy_level'] = hierarchy_levels
                logger.info("Niveaux hiérarchiques estimés à partir du clustering")
                
            except Exception as e:
                logger.warning(f"Impossible de déterminer la hiérarchie à partir du clustering: {e}")
            
            # Visualisation du dendrogramme pour un sous-ensemble de données
            try:
                if len(self.df) <= 100:  # Limiter la visualisation à un nombre raisonnable de termes
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
            
            # Analyser les outliers : ils peuvent être soit très spécifiques, soit très génériques
            if 'def_length' in self.df.columns:
                outliers_def_lengths = self.df[self.df['is_outlier']]['def_length'].mean()
                non_outliers_def_lengths = self.df[~self.df['is_outlier']]['def_length'].mean()
                
                logger.info(f"Longueur moyenne des définitions: outliers={outliers_def_lengths:.1f}, non-outliers={non_outliers_def_lengths:.1f}")
                
                # Les outliers avec des définitions longues sont probablement des termes très spécifiques
                if outliers_def_lengths > non_outliers_def_lengths * 1.5:
                    logger.info("Les outliers semblent être des termes très spécifiques (définitions longues)")
                elif outliers_def_lengths < non_outliers_def_lengths * 0.7:
                    logger.info("Les outliers semblent être des termes très génériques (définitions courtes)")
        except Exception as e:
            logger.error(f"Erreur lors du clustering DBSCAN: {e}")
            self.df['dbscan_cluster'] = -1
            self.df['is_outlier'] = False
        
        # 4. HDBSCAN pour une meilleure détection hiérarchique
        try:
            import hdbscan
            
            # Ajuster les paramètres pour le domaine musical baroque
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=3,
                min_samples=2,
                cluster_selection_epsilon=0.5,
                alpha=1.0,
                cluster_selection_method='eom'  # 'eom' est souvent meilleur pour la hiérarchie
            )
            
            hdbscan_labels = clusterer.fit_predict(self.embeddings)
            self.df['hdbscan_cluster'] = hdbscan_labels
            
            # Compter le nombre de clusters formés (en excluant le bruit = -1)
            n_hdbscan_clusters = len(set(hdbscan_labels)) - (1 if -1 in hdbscan_labels else 0)
            logger.info(f"HDBSCAN appliqué: {n_hdbscan_clusters} clusters formés")
            
            # Ajouter des informations de probabilité d'appartenance
            if hasattr(clusterer, 'probabilities_'):
                self.df['hdbscan_probability'] = clusterer.probabilities_
                
            # Utiliser l'arbre de clustering pour la hiérarchie
            if hasattr(clusterer, 'condensed_tree_'):
                # Extraire les niveaux hiérarchiques à partir de l'arbre condensé
                self.hdbscan_tree = clusterer.condensed_tree_
                
                # Tenter de visualiser la hiérarchie HDBSCAN
                try:
                    plt.figure(figsize=(12, 8))
                    clusterer.condensed_tree_.plot(select_clusters=True)
                    plt.title("Arbre hiérarchique HDBSCAN")
                    plt.savefig(self.output_dir / "hdbscan_hierarchy.png")
                    logger.info("Arbre hiérarchique HDBSCAN sauvegardé")
                except Exception as e:
                    logger.warning(f"Impossible de visualiser l'arbre HDBSCAN: {e}")
        except Exception as e:
            logger.error(f"Erreur lors de l'application de HDBSCAN: {e}")
            self.df['hdbscan_cluster'] = -1
        
        # 5. Détection de communautés dans le graphe de similarité
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
            
            # Calculer la centralité des termes dans chaque communauté
            # Cette centralité peut indiquer l'importance ou la représentativité du terme
            centrality = nx.degree_centrality(G)
            self.df['centrality'] = [centrality.get(i, 0) for i in range(len(self.df))]
            
            # Termes représentatifs des communautés (les plus centraux)
            community_representatives = {}
            for comm_id in set(communities.values()):
                comm_nodes = [node for node in range(len(self.df)) if communities.get(node, -1) == comm_id]
                if comm_nodes:
                    # Prendre le terme le plus central
                    most_central = max(comm_nodes, key=lambda n: centrality.get(n, 0))
                    community_representatives[comm_id] = self.df.iloc[most_central]['original_term']
                    
            # Ajouter le terme représentatif de la communauté
            self.df['community_representative'] = self.df['community'].map(community_representatives)
            
            # Visualiser le graphe de communautés
            try:
                plt.figure(figsize=(14, 14))
                pos = nx.spring_layout(G, k=0.3, iterations=50, seed=42)
                
                # Couleurs pour les communautés
                cmap = plt.cm.get_cmap("tab20", n_communities)
                
                # Taille des nœuds proportionnelle à leur centralité
                node_size = [centrality.get(node, 0) * 2000 + 100 for node in G.nodes()]
                
                # Dessiner les nœuds colorés par communauté
                for comm_id in set(communities.values()):
                    list_nodes = [node for node in G.nodes() if communities[node] == comm_id]
                    nx.draw_networkx_nodes(
                        G, pos, nodelist=list_nodes,
                        node_color=[cmap(comm_id)],
                        node_size=[centrality.get(node, 0) * 2000 + 100 for node in list_nodes], 
                        alpha=0.8
                    )
                
                # Dessiner les arêtes avec une transparence basée sur le poids
                edge_colors = [G[u][v]['weight'] for u, v in G.edges()]
                nx.draw_networkx_edges(G, pos, alpha=0.2, width=1, edge_color=edge_colors, edge_cmap=plt.cm.Blues)
                
                # Ajouter des étiquettes pour les termes les plus centraux
                # Limiter aux 20-30 termes les plus centraux pour éviter l'encombrement
                top_central_nodes = sorted(G.nodes(), key=lambda n: centrality.get(n, 0), reverse=True)[:25]
                central_labels = {node: G.nodes[node]['label'] for node in top_central_nodes}
                nx.draw_networkx_labels(G, pos, labels=central_labels, font_size=8)
                
                plt.title("Graphe des communautés de termes musicaux baroques")
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(self.output_dir / "community_graph.png", dpi=150)
                logger.info("Graphe des communautés sauvegardé")
            except Exception as e:
                logger.warning(f"Impossible de générer le graphe des communautés: {e}")
        except Exception as e:
            logger.error(f"Erreur lors de la détection de communautés: {e}")
            self.df['community'] = -1
            self.df['centrality'] = 0.0
            self.df['community_representative'] = None

    def _classify_by_linguistic_features(self) -> None:
        """
        Classifie les termes en utilisant des caractéristiques linguistiques extraites
        à partir des termes ET de leurs définitions.
        
        Cette méthode analyse la structure grammaticale, les parties du discours,
        et d'autres caractéristiques linguistiques pour enrichir la classification.
        Elle intègre également les informations extraites des définitions quand disponibles.
        """
        logger.info("Classification basée sur l'analyse linguistique avec intégration des définitions...")
        
        # 1. Détecter les catégories grammaticales dominantes
        pos_categories = {
            'entité': ['PROPN', 'PERSON', 'ORG', 'LOC'],
            'instrument': ['NOUN'],
            'technique': ['NOUN', 'VERB'],
            'forme_musicale': ['NOUN'],
            'compositeur': ['PROPN', 'PERSON'],
            'théorie': ['NOUN', 'ADJ'],
            'qualité': ['ADJ'],
            'concept': ['NOUN', 'ADJ']
        }
        
        # Initialiser la colonne pour la catégorie linguistique
        self.df['linguistic_category'] = None
        
        # Catégories spécifiques à la musique baroque pour une meilleure classification
        musical_categories = {
            'instrument': ['violon', 'flûte', 'clavecin', 'orgue', 'luth', 'viole', 'hautbois', 'trompette', 'instrument'],
            'forme_musicale': ['sonate', 'fugue', 'suite', 'cantate', 'concerto', 'oratorio', 'opéra', 'aria', 'prélude', 'toccata', 'partita'],
            'technique': ['ornementation', 'contrepoint', 'improvisation', 'basse', 'continuo', 'cadence', 'gamme', 'interval', 'accord'],
            'compositeur': ['bach', 'handel', 'vivaldi', 'telemann', 'rameau', 'couperin', 'scarlatti', 'monteverdi', 'purcell'],
            'théorie': ['tempérament', 'tonalité', 'mode', 'harmonie', 'contrepoint', 'rhétorique', 'affect', 'doctrine', 'système']
        }
        
        # Appliquer la classification linguistique en prenant compte des termes et définitions
        for idx, row in self.df.iterrows():
            term = row['clean_term'].lower() if pd.notna(row['clean_term']) else ""
            
            # Identifier si le terme appartient à une catégorie musicale spécifique
            category_found = False
            for category, keywords in musical_categories.items():
                if any(keyword in term for keyword in keywords):
                    self.df.at[idx, 'linguistic_category'] = category
                    category_found = True
                    break
            
            # Si pas de correspondance directe, utiliser les POS tags
            if not category_found:
                pos_tags = row['pos_tags'].split() if pd.notna(row['pos_tags']) else []
                
                # Détecter la catégorie en fonction des POS tags
                for category, pos_list in pos_categories.items():
                    if any(pos in pos_tags for pos in pos_list):
                        self.df.at[idx, 'linguistic_category'] = category
                        category_found = True
                        break
            
            # Utiliser les définitions si disponibles et si toujours pas de catégorie
            if not category_found and 'def_keywords' in self.df.columns and isinstance(row['def_keywords'], list) and len(row['def_keywords']) > 0:
                def_keywords = row['def_keywords']
                
                # Vérifier si les mots-clés de la définition correspondent à une catégorie
                for category, keywords in musical_categories.items():
                    # Vérifier si un des mots-clés de la définition correspond à une catégorie musicale
                    if isinstance(def_keywords, list) and any(any(keyword in def_kw.lower() for def_kw in def_keywords) for keyword in keywords):
                        self.df.at[idx, 'linguistic_category'] = category
                        category_found = True
                        break
            
            # Classification par défaut
            if not category_found:
                self.df.at[idx, 'linguistic_category'] = 'concept'
        
        # 2. Identifier les relations potentielles basées sur la structure grammaticale et les définitions
        self.df['related_terms'] = None
        self.df['related_by_definition'] = None
        
        # Créer un dictionnaire pour stocker toutes les relations identifiées
        all_relations = defaultdict(list)
        
        for idx, row in self.df.iterrows():
            # Analyser la structure pour identifier les relations dans le terme
            term_doc = self.nlp(row['clean_term'] if pd.notna(row['clean_term']) else "")
            term_relations = []
            
            for token in term_doc:
                # Chercher des modifieurs ou compléments
                if token.dep_ in ['prep', 'pobj', 'nmod', 'amod']:
                    # Trouver le terme principal auquel ce token est relié
                    head = token.head.text
                    if head and head != token.text:
                        term_relations.append(head)
            
            # Analyser la définition pour identifier des relations sémantiques si disponible
            def_relations = []
            if 'definition_clean' in self.df.columns and pd.notna(row['definition_clean']):
                def_doc = self.nlp(row['definition_clean'])
                
                # Extraire des termes potentiellement liés à partir de la définition
                for chunk in def_doc.noun_chunks:
                    if len(chunk.text.split()) <= 3:  # Limiter la taille des expressions
                        possible_related = chunk.text.lower()
                        # Vérifier si ce terme existe dans notre liste de termes
                        if possible_related != row['clean_term'].lower() and any(
                            possible_related in other_term.lower() 
                            for other_term in self.df['clean_term'] if pd.notna(other_term)
                        ):
                            def_relations.append(possible_related)
            
            # Enregistrer les relations trouvées
            if term_relations:
                self.df.at[idx, 'related_terms'] = ','.join(term_relations)
                # Ajouter aux relations globales
                for rel in term_relations:
                    all_relations[row['original_term']].append(rel)
            
            if def_relations:
                self.df.at[idx, 'related_by_definition'] = ','.join(def_relations)
                # Ajouter aux relations globales
                for rel in def_relations:
                    all_relations[row['original_term']].append(rel)
        
        # Stocker les relations pour une utilisation ultérieure
        self.term_relations = all_relations
        
        logger.info(f"Classification linguistique terminée: {self.df['linguistic_category'].value_counts().to_dict()}")

    def _detect_hierarchical_relations(self) -> None:
        """
        Détecte les relations hiérarchiques entre les termes en analysant:
        1. Les inclusions lexicales (terme A contenu dans terme B)
        2. Les relations sémantiques basées sur les embeddings
        3. Les relations identifiées dans les définitions
        4. Les patterns hiérarchiques spécifiques à la musique baroque
        
        Crée une structure hiérarchique des termes musicaux baroques.
        """
        logger.info("Détection des relations hiérarchiques entre termes...")
        
        # Initialiser les colonnes pour les relations hiérarchiques
        self.df['hypernyms'] = None  # Termes plus génériques
        self.df['hyponyms'] = None   # Termes plus spécifiques
        self.df['part_of'] = None    # Relation "fait partie de"
        
        # Dictionnaires pour stocker les relations
        hypernyms = defaultdict(list)  # Terme -> ses hypernyms
        hyponyms = defaultdict(list)   # Terme -> ses hyponyms
        part_of = defaultdict(list)    # Terme -> ce dont il fait partie
        
        # 1. Détection par inclusion lexicale
        for i, row_i in self.df.iterrows():
            term_i = row_i['clean_term'].lower() if pd.notna(row_i['clean_term']) else ""
            if not term_i:
                continue
                
            # Tokens du terme i
            tokens_i = set(term_i.split())
            
            for j, row_j in self.df.iterrows():
                if i == j:
                    continue
                    
                term_j = row_j['clean_term'].lower() if pd.notna(row_j['clean_term']) else ""
                if not term_j:
                    continue
                    
                tokens_j = set(term_j.split())
                
                # Vérifier l'inclusion lexicale
                if tokens_i.issubset(tokens_j) and len(tokens_i) < len(tokens_j):
                    # term_i est plus générique, term_j est plus spécifique
                    hypernyms[term_j].append(term_i)
                    hyponyms[term_i].append(term_j)
                elif tokens_j.issubset(tokens_i) and len(tokens_j) < len(tokens_i):
                    # term_j est plus générique, term_i est plus spécifique
                    hypernyms[term_i].append(term_j)
                    hyponyms[term_j].append(term_i)
        
        # 2. Détection par proximité sémantique (embeddings)
        for i, row_i in self.df.iterrows():
            term_i = row_i['original_term']
            emb_i = self.embeddings[i]
            
            # Trouver les termes les plus similaires
            similarities = [(j, cosine_similarity([emb_i], [self.embeddings[j]])[0][0]) 
                            for j in range(len(self.df)) if j != i]
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Prendre les 5 termes les plus similaires
            top_similar = similarities[:5]
            
            for j, sim in top_similar:
                if sim > 0.8:  # Seuil de similarité élevé
                    term_j = self.df.iloc[j]['original_term']
                    
                    # Vérifier la différence de complexité pour inférer la relation hiérarchique
                    complexity_i = len(term_i.split())
                    complexity_j = len(term_j.split())
                    
                    if complexity_i < complexity_j:
                        # term_i est probablement plus générique
                        if term_j not in hyponyms[term_i]:
                            hyponyms[term_i].append(term_j)
                        if term_i not in hypernyms[term_j]:
                            hypernyms[term_j].append(term_i)
                    elif complexity_i > complexity_j:
                        # term_j est probablement plus générique
                        if term_i not in hyponyms[term_j]:
                            hyponyms[term_j].append(term_i)
                        if term_j not in hypernyms[term_i]:
                            hypernyms[term_i].append(term_j)
        
        # 3. Détecter les relations "partie de" à partir des définitions
        part_of_patterns = [
            "partie de", "élément de", "composante de", "section de", 
            "segment de", "appartient à", "caractéristique de", "propre à"
        ]
        
        if 'definition_clean' in self.df.columns:
            for idx, row in self.df.iterrows():
                term = row['original_term']
                
                if pd.notna(row['definition_clean']):
                    definition = row['definition_clean'].lower()
                    
                    # Chercher des patterns "partie de"
                    for pattern in part_of_patterns:
                        if pattern in definition:
                            # Trouver ce dont ce terme fait partie
                            words_after = definition.split(pattern)[1].strip().split()[:5]
                            potential_whole = ' '.join(words_after)
                            
                            # Chercher un terme existant correspondant
                            for other_idx, other_row in self.df.iterrows():
                                if idx != other_idx:
                                    other_term = other_row['original_term'].lower()
                                    if other_term in potential_whole:
                                        part_of[term].append(other_row['original_term'])
                                        break
        
        # 4. Détecter les relations basées sur des patterns spécifiques à la musique baroque
        baroque_hierarchies = {
            'forme_musicale': ['opéra', 'oratorio', 'cantate', 'suite', 'sonate', 'concerto'],
            'technique': ['ornementation', 'contrepoint', 'basse continue', 'improvisation'],
            'instrument': ['cordes', 'vents', 'claviers', 'percussion']
        }
        
        # Établir des relations hiérarchiques basées sur les catégories musicales prédéfinies
        if 'linguistic_category' in self.df.columns:
            for category, hierarchy in baroque_hierarchies.items():
                category_terms = self.df[self.df['linguistic_category'] == category]['original_term'].tolist()
                
                # Pour chaque terme dans cette catégorie
                for term in category_terms:
                    # Vérifier si ce terme appartient à une hiérarchie spécifique
                    for i, hier_term in enumerate(hierarchy):
                        if hier_term.lower() in term.lower():
                            # Les termes de niveau supérieur dans la hiérarchie sont hypernyms
                            for j in range(i):
                                if hierarchy[j] not in hypernyms[term] and hierarchy[j] != term:
                                    hypernyms[term].append(hierarchy[j])
                            
                            # Les termes de niveau inférieur sont hyponyms
                            for j in range(i+1, len(hierarchy)):
                                if hierarchy[j] not in hyponyms[term] and hierarchy[j] != term:
                                    hyponyms[term].append(hierarchy[j])
        
        # Enregistrer les relations hiérarchiques dans le DataFrame
        for idx, row in self.df.iterrows():
            term = row['original_term']
            clean_term = row['clean_term'].lower() if pd.notna(row['clean_term']) else ""
            
            if clean_term in hypernyms:
                self.df.at[idx, 'hypernyms'] = ','.join(hypernyms[clean_term])
            
            if clean_term in hyponyms:
                self.df.at[idx, 'hyponyms'] = ','.join(hyponyms[clean_term])
            
            if term in part_of:
                self.df.at[idx, 'part_of'] = ','.join(part_of[term])
        
        # Sauvegarder les relations hiérarchiques pour une utilisation ultérieure
        self.hypernyms = {k: v for k, v in hypernyms.items() if v}
        self.hyponyms = {k: v for k, v in hyponyms.items() if v}
        self.part_of = {k: v for k, v in part_of.items() if v}
        
        logger.info(f"Relations hiérarchiques détectées: {sum(len(v) for v in hypernyms.values())} hypernyms, "
                    f"{sum(len(v) for v in hyponyms.values())} hyponyms, "
                    f"{sum(len(v) for v in part_of.values())} relations partie-de")
        
        # Visualiser les relations hiérarchiques principales (les plus fréquentes)
        try:
            # Créer un graphe dirigé pour visualiser la hiérarchie
            G = nx.DiGraph()
            
            # Ajouter les nœuds (termes)
            for term in self.df['original_term']:
                G.add_node(term)
            
            # Ajouter les arêtes (relations hiérarchiques)
            for term, hypers in hypernyms.items():
                term_idx = self.df[self.df['clean_term'].str.lower() == term.lower()].index
                if len(term_idx) > 0:
                    term_original = self.df.iloc[term_idx[0]]['original_term']
                    for hyper in hypers:
                        hyper_idx = self.df[self.df['clean_term'].str.lower() == hyper.lower()].index
                        if len(hyper_idx) > 0:
                            hyper_original = self.df.iloc[hyper_idx[0]]['original_term']
                            G.add_edge(hyper_original, term_original)
            
            # Limiter la visualisation aux composants principaux
            if len(G) > 0:
                # Identifier les composantes connexes
                components = list(nx.weakly_connected_components(G))
                components.sort(key=len, reverse=True)
                
                # Prendre la plus grande composante
                if components:
                    main_component = G.subgraph(components[0])
                    
                    # Visualiser seulement si le nombre de nœuds est raisonnable
                    if len(main_component) <= 50:
                        plt.figure(figsize=(12, 10))
                        pos = nx.spring_layout(main_component, k=0.8, iterations=100, seed=42)
                        
                        nx.draw_networkx_nodes(main_component, pos, node_size=80, alpha=0.8, 
                                            node_color="lightblue")
                        nx.draw_networkx_edges(main_component, pos, width=1, alpha=0.5, 
                                            edge_color="gray", arrows=True)
                        nx.draw_networkx_labels(main_component, pos, font_size=8)
                        
                        plt.title("Relations hiérarchiques principales")
                        plt.axis('off')
                        plt.tight_layout()
                        plt.savefig(self.output_dir / "hierarchical_relations.png")
                        logger.info("Graphe des relations hiérarchiques sauvegardé")
        except Exception as e:
            logger.warning(f"Impossible de visualiser les relations hiérarchiques: {e}")

    def _merge_classifications(self) -> None:
        """
        Fusionne les différentes classifications pour obtenir une classification finale.
        
        Cette méthode intègre:
        1. Les résultats des différents algorithmes de clustering
        2. La classification linguistique
        3. Les relations hiérarchiques détectées
        4. L'analyse de la centralité des termes
        
        Génère une classification consensuelle optimisée pour le domaine musical baroque.
        """
        logger.info("Fusion des classifications pour obtenir une classification finale...")
        
        # Initialiser les colonnes pour la classification finale
        self.df['final_category'] = None
        self.df['category_confidence'] = 0.0
        self.df['taxo_level'] = None  # Niveau taxonomique (1 = général, 2 = intermédiaire, 3 = spécifique)
        
        # Catégories de termes musicaux baroques (adaptées au domaine)
        baroque_categories = {
            'forme_musicale': ['sonate', 'fugue', 'suite', 'concerto', 'cantate', 'oratorio', 'opéra', 'aria', 'partita'],
            'technique': ['ornementation', 'contrepoint', 'basse', 'continuo', 'cadence', 'improvisation', 'trille'],
            'instrument': ['violon', 'clavecin', 'orgue', 'flûte', 'hautbois', 'viole', 'luth', 'théorbe'],
            'théorie': ['harmonie', 'mode', 'tempérament', 'gamme', 'tonalité', 'accord', 'intervalle'],
            'esthétique': ['affect', 'passion', 'grâce', 'goût', 'sentiment', 'caractère', 'style'],
            'notation': ['clé', 'partition', 'tablature', 'mesure', 'rythme', 'note'],
            'interprétation': ['articulation', 'phrasé', 'nuance', 'tempo', 'dynamique', 'expression']
        }
        
        # 1. Intégrer les résultats de clustering avec la classification linguistique
        for idx, row in self.df.iterrows():
            confidence_scores = {}
            
            # Partir de la classification linguistique comme base
            if pd.notna(row['linguistic_category']):
                base_category = row['linguistic_category']
                confidence_scores[base_category] = 0.5  # Confiance de base
            else:
                base_category = 'concept'
                confidence_scores[base_category] = 0.3
            
            # Ajouter des indices basés sur les résultats des clusterings
            if pd.notna(row['kmeans_representative']):
                rep_term = row['kmeans_representative'].lower()
                for category, keywords in baroque_categories.items():
                    if any(keyword in rep_term for keyword in keywords):
                        confidence_scores[category] = confidence_scores.get(category, 0) + 0.2
            
            # Utiliser la détection de communauté
            if pd.notna(row['community_representative']):
                rep_term = row['community_representative'].lower()
                for category, keywords in baroque_categories.items():
                    if any(keyword in rep_term for keyword in keywords):
                        confidence_scores[category] = confidence_scores.get(category, 0) + 0.3
            
            # Analyse lexicale du terme lui-même
            term = row['clean_term'].lower() if pd.notna(row['clean_term']) else ""
            for category, keywords in baroque_categories.items():
                if any(keyword in term for keyword in keywords):
                    confidence_scores[category] = confidence_scores.get(category, 0) + 0.4
            
            # Intégrer les définitions si disponibles
            if 'definition_clean' in self.df.columns and pd.notna(row['definition_clean']):
                definition = row['definition_clean'].lower()
                for category, keywords in baroque_categories.items():
                    if any(keyword in definition for keyword in keywords):
                        confidence_scores[category] = confidence_scores.get(category, 0) + 0.3
            
            # Utiliser la catégorie avec le score de confiance le plus élevé
            if confidence_scores:
                final_category = max(confidence_scores.items(), key=lambda x: x[1])[0]
                confidence = confidence_scores[final_category]
            else:
                final_category = base_category
                confidence = 0.3
            
            # Déterminer le niveau taxonomique
            taxo_level = 2  # Niveau par défaut (intermédiaire)
            
            # Les termes courts et généraux sont de niveau 1
            if len(term.split()) == 1:
                taxo_level = 1
            # Les termes longs ou techniques sont de niveau 3
            elif len(term.split()) >= 3 or ('hypernyms' in self.df.columns and pd.notna(row['hypernyms'])):
                taxo_level = 3
            
            # Ajuster en fonction des relations hiérarchiques
            if 'hypernyms' in self.df.columns and pd.notna(row['hypernyms']) and len(row['hypernyms'].split(',')) >= 2:
                taxo_level = min(taxo_level + 1, 3)  # Augmenter le niveau (plus spécifique)
            if 'hyponyms' in self.df.columns and pd.notna(row['hyponyms']) and len(row['hyponyms'].split(',')) >= 3:
                taxo_level = max(taxo_level - 1, 1)  # Réduire le niveau (plus général)
            
            # Enregistrer la classification finale
            self.df.at[idx, 'final_category'] = final_category
            self.df.at[idx, 'category_confidence'] = confidence
            self.df.at[idx, 'taxo_level'] = taxo_level
        
        # Statistiques sur la classification finale
        category_counts = self.df['final_category'].value_counts()
        logger.info(f"Classification finale: {category_counts.to_dict()}")
        
        # Répartition des niveaux taxonomiques
        level_counts = self.df['taxo_level'].value_counts()
        logger.info(f"Niveaux taxonomiques: {level_counts.to_dict()}")
        
        # Visualisation de la classification finale
        try:
            plt.figure(figsize=(10, 6))
            ax = category_counts.plot(kind='bar', color='skyblue')
            plt.title("Répartition des catégories de termes musicaux baroques")
            plt.xlabel("Catégorie")
            plt.ylabel("Nombre de termes")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(self.output_dir / "final_categories.png")
            logger.info("Graphique des catégories finales sauvegardé")
        except Exception as e:
            logger.warning(f"Impossible de visualiser les catégories finales: {e}")

    def _analyze_classification_results(self) -> None:
        """
        Analyse les résultats de classification pour extraire des insights et valider la qualité.
        
        Cette méthode:
        1. Évalue la cohérence interne des clusters
        2. Identifie les termes les plus représentatifs de chaque catégorie
        3. Détecte les outliers potentiels
        4. Génère des visualisations pour faciliter l'interprétation
        5. Produit des statistiques sur la taxonomie résultante
        """
        logger.info("Analyse des résultats de classification...")
        
        # 1. Cohérence interne des clusters
        category_embeddings = {}
        category_coherence = {}
        
        for category in self.df['final_category'].unique():
            if pd.isna(category):
                continue
                
            # Calculer les embeddings moyens pour chaque catégorie
            cat_indices = self.df[self.df['final_category'] == category].index
            cat_embeddings = self.embeddings[cat_indices]
            
            if len(cat_embeddings) > 0:
                category_embeddings[category] = np.mean(cat_embeddings, axis=0)
                
                # Calculer la cohérence (distance moyenne entre les termes et le centre)
                distances = [
                    cosine_similarity([category_embeddings[category]], [self.embeddings[idx]])[0][0]
                    for idx in cat_indices
                ]
                category_coherence[category] = np.mean(distances) if distances else 0
        
        # Trier les catégories par cohérence
        sorted_coherence = {k: v for k, v in sorted(category_coherence.items(), key=lambda x: x[1], reverse=True)}
        
        logger.info("Cohérence interne des catégories:")
        for category, coherence in sorted_coherence.items():
            logger.info(f"  - {category}: {coherence:.4f}")
        
        # 2. Termes les plus représentatifs de chaque catégorie
        representative_terms = {}
        
        for category in self.df['final_category'].unique():
            if pd.isna(category):
                continue
                
            cat_indices = self.df[self.df['final_category'] == category].index
            
            if not len(cat_indices) or category not in category_embeddings:
                continue
                
            # Calculer la similarité de chaque terme avec le centre de la catégorie
            similarities = [
                (idx, cosine_similarity([category_embeddings[category]], [self.embeddings[idx]])[0][0])
                for idx in cat_indices
            ]
            
            # Trier par similarité
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Prendre les 5 termes les plus représentatifs
            top_indices = [idx for idx, _ in similarities[:5]]
            representative_terms[category] = self.df.iloc[top_indices]['original_term'].tolist()
        
        # Ajouter les termes représentatifs au DataFrame
        self.df['is_representative'] = False
        for category, terms in representative_terms.items():
            self.df.loc[self.df['original_term'].isin(terms), 'is_representative'] = True
        
        logger.info("Termes représentatifs par catégorie:")
        for category, terms in representative_terms.items():
            logger.info(f"  - {category}: {', '.join(terms[:3])}...")
        
        # 3. Détection des outliers par catégorie
        self.df['is_category_outlier'] = False
        outlier_threshold = 0.3  # Seuil pour considérer comme outlier
        
        for category in self.df['final_category'].unique():
            if pd.isna(category) or category not in category_embeddings:
                continue
                
            cat_indices = self.df[self.df['final_category'] == category].index
            
            for idx in cat_indices:
                similarity = cosine_similarity([category_embeddings[category]], [self.embeddings[idx]])[0][0]
                
                # Si la similarité est faible, marquer comme outlier
                if similarity < outlier_threshold:
                    self.df.at[idx, 'is_category_outlier'] = True
        
        n_outliers = sum(self.df['is_category_outlier'])
        logger.info(f"Nombre d'outliers détectés: {n_outliers} ({n_outliers/len(self.df)*100:.1f}%)")
        
        # 4. Visualisation de la classification par réduction dimensionnelle
        try:
            # Utiliser UMAP pour la visualisation (meilleure préservation de la structure globale)
            import umap
            reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
            embedding_2d = reducer.fit_transform(self.embeddings)
            
            # Ajouter les coordonnées réduites au DataFrame
            self.df['x_coord'] = embedding_2d[:, 0]
            self.df['y_coord'] = embedding_2d[:, 1]
            
            # Créer une visualisation interactive avec Plotly
            try:
                import plotly.express as px
                import plotly.graph_objects as go
                
                # Créer un DataFrame pour la visualisation
                plot_df = self.df[['original_term', 'final_category', 'taxo_level', 'x_coord', 'y_coord', 'is_representative', 'is_category_outlier']]
                
                # Créer la figure
                fig = px.scatter(
                    plot_df, 
                    x='x_coord', 
                    y='y_coord',
                    color='final_category',
                    symbol='taxo_level',
                    hover_name='original_term',
                    size=[8 if r else 5 for r in plot_df['is_representative']],
                    opacity=[0.5 if o else 0.8 for o in plot_df['is_category_outlier']],
                    title="Visualisation des termes musicaux baroques par catégorie et niveau taxonomique"
                )
                
                # Ajuster la mise en page
                fig.update_layout(
                    width=1000,
                    height=800,
                    legend_title_text='Catégorie',
                    xaxis_title="",
                    yaxis_title="",
                    xaxis_showticklabels=False,
                    yaxis_showticklabels=False
                )
                
                # Sauvegarder au format HTML pour interactivité
                fig.write_html(str(self.output_dir / "terms_visualization.html"))
                
                # Sauvegarder aussi une version statique
                fig.write_image(str(self.output_dir / "terms_visualization.png"))
                
                logger.info("Visualisation des termes sauvegardée au format HTML et PNG")
                
            except Exception as e:
                logger.warning(f"Impossible de créer la visualisation Plotly: {e}")
                # Fallback sur Matplotlib
                plt.figure(figsize=(12, 10))
                categories = self.df['final_category'].unique()
                
                for i, cat in enumerate(categories):
                    if pd.isna(cat):
                        continue
                    
                    cat_data = self.df[self.df['final_category'] == cat]
                    plt.scatter(
                        cat_data['x_coord'], 
                        cat_data['y_coord'],
                        label=cat,
                        alpha=0.7,
                        s=[50 if r else 20 for r in cat_data['is_representative']]
                    )
                
                plt.title("Visualisation 2D des termes musicaux baroques")
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(self.output_dir / "terms_visualization.png")
                logger.info("Visualisation des termes sauvegardée (version simplifiée)")
        
        except Exception as e:
            logger.error(f"Erreur lors de la visualisation des résultats: {e}")
        
        # 5. Statistiques sur la taxonomie
        taxo_stats = {
            'n_categories': len(self.df['final_category'].unique()),
            'n_level1': sum(self.df['taxo_level'] == 1),
            'n_level2': sum(self.df['taxo_level'] == 2),
            'n_level3': sum(self.df['taxo_level'] == 3),
            'n_hierarchical_relations': sum(pd.notna(self.df['hypernyms'])) + sum(pd.notna(self.df['hyponyms'])),
            'n_representative_terms': sum(self.df['is_representative']),
            'n_outliers': sum(self.df['is_category_outlier']),
            'avg_confidence': self.df['category_confidence'].mean()
        }
        
        logger.info(f"Statistiques de la taxonomie générée:")
        for stat, value in taxo_stats.items():
            logger.info(f"  - {stat}: {value}")
        
        # Sauvegarder les statistiques
        with open(self.output_dir / "taxonomy_stats.json", 'w') as f:
            json.dump(taxo_stats, f, indent=2)
        
        # Générer un rapport textuel des résultats
        self._generate_classification_report()

    def _generate_classification_report(self) -> None:
        """
        Génère un rapport détaillé de la classification des termes et de la structure taxonomique.
        
        Ce rapport inclut:
        1. Une synthèse globale des résultats
        2. L'analyse par catégorie avec les termes représentatifs
        3. Les statistiques de cohérence et de couverture
        4. Les recommandations pour l'amélioration de la taxonomie
        5. Une section dédiée aux cas particuliers et outliers
        """
        logger.info("Génération du rapport de classification...")
        
        report_path = self.output_dir / "classification_report.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            # Titre et introduction
            f.write("# Rapport de classification des termes musicaux baroques\n\n")
            f.write(f"*Généré le {datetime.now().strftime('%d/%m/%Y à %H:%M')}*\n\n")
            f.write("## Synthèse globale\n\n")
            
            # Statistiques générales
            n_terms = len(self.df)
            n_categories = len(self.df['final_category'].dropna().unique())
            n_hierarchical = sum(pd.notna(self.df['hypernyms']) | pd.notna(self.df['hyponyms']))
            
            f.write(f"- **Nombre total de termes analysés**: {n_terms}\n")
            f.write(f"- **Nombre de catégories identifiées**: {n_categories}\n")
            f.write(f"- **Termes avec relations hiérarchiques**: {n_hierarchical} ({n_hierarchical/n_terms*100:.1f}%)\n")
            f.write(f"- **Niveaux taxonomiques**: {self.df['taxo_level'].value_counts().to_dict()}\n")
            f.write(f"- **Confiance moyenne de classification**: {self.df['category_confidence'].mean():.2f}\n\n")
            
            # Graphique de répartition des catégories
            f.write("![Répartition des catégories](final_categories.png)\n\n")
            
            # Analyse par catégorie
            f.write("## Analyse détaillée par catégorie\n\n")
            
            for category in sorted(self.df['final_category'].dropna().unique()):
                cat_df = self.df[self.df['final_category'] == category]
                f.write(f"### {category.capitalize()}\n\n")
                
                # Statistiques de la catégorie
                f.write(f"- **Nombre de termes**: {len(cat_df)}\n")
                f.write(f"- **Répartition par niveau**: {cat_df['taxo_level'].value_counts().to_dict()}\n")
                f.write(f"- **Confiance moyenne**: {cat_df['category_confidence'].mean():.2f}\n\n")
                
                # Termes représentatifs
                rep_terms = cat_df[cat_df['is_representative']]['original_term'].tolist()
                if rep_terms:
                    f.write("**Termes représentatifs**:\n\n")
                    for term in rep_terms:
                        f.write(f"- {term}\n")
                    f.write("\n")
                
                # Outliers potentiels
                outliers = cat_df[cat_df['is_category_outlier']]['original_term'].tolist()
                if outliers:
                    f.write("**Outliers potentiels**:\n\n")
                    for term in outliers[:5]:  # Limiter aux 5 premiers
                        f.write(f"- {term}\n")
                    if len(outliers) > 5:
                        f.write(f"- ... et {len(outliers)-5} autres\n")
                    f.write("\n")
            
            # Visualisation de la structure
            f.write("## Visualisation de la structure taxonomique\n\n")
            f.write("La visualisation interactive des termes est disponible dans le fichier `terms_visualization.html`.\n\n")
            f.write("![Visualisation 2D des termes](terms_visualization.png)\n\n")
            
            # Recommandations
            f.write("## Recommandations d'amélioration\n\n")
            
            # Catégories avec faible cohérence
            low_coherence = [cat for cat, coh in category_coherence.items() if coh < 0.6]
            if low_coherence:
                f.write("### Catégories à réviser\n\n")
                f.write("Les catégories suivantes présentent une faible cohérence interne et pourraient bénéficier d'une révision:\n\n")
                for cat in low_coherence:
                    f.write(f"- **{cat}**: envisager de subdiviser ou de reclassifier certains termes\n")
                f.write("\n")
            
            # Termes sans relations
            n_isolated = sum((~pd.notna(self.df['hypernyms'])) & (~pd.notna(self.df['hyponyms'])))
            if n_isolated > 0:
                f.write("### Termes isolés\n\n")
                f.write(f"**{n_isolated}** termes ({n_isolated/n_terms*100:.1f}%) n'ont aucune relation hiérarchique définie.\n")
                f.write("Il est recommandé d'enrichir ces termes avec des relations broader/narrower pour améliorer la structure du thésaurus.\n\n")
            
            # Prochaines étapes
            f.write("## Prochaines étapes\n\n")
            f.write("1. **Validation manuelle** des catégories et relations proposées\n")
            f.write("2. **Enrichissement** des termes avec des définitions et contextes d'usage\n")
            f.write("3. **Extension** de la couverture à d'autres périodes musicales\n")
            f.write("4. **Intégration** avec des ressources externes (MIMO, Grove Dictionary, etc.)\n")
            f.write("5. **Publication** au format SKOS/RDF pour interopérabilité\n")
        
        logger.info(f"Rapport de classification généré: {report_path}")

    def generate_skos_structure(self) -> None:
        """
        Génère la structure SKOS à partir de la classification et des relations établies.
        
        Cette méthode:
        1. Crée le graphe RDF principal
        2. Définit les concepts et les schémas de concepts
        3. Établit les relations hiérarchiques (broader/narrower)
        4. Ajoute les relations associatives (related)
        5. Intègre les métadonnées descriptives (labels, définitions, notes)
        6. Organise les concepts en collections thématiques
        """
        logger.info("Génération de la structure SKOS...")
        
        # Initialiser le graphe RDF avec rdflib
        self.graph = rdflib.Graph()
        
        # Définir les espaces de noms (namespaces)
        self.graph.bind('skos', SKOS)
        self.graph.bind('dcterms', DCTERMS)
        self.graph.bind('owl', OWL)
        self.graph.bind('rdfs', RDFS)
        
        # Définir l'espace de nom pour notre thésaurus
        BAROQUE = rdflib.Namespace("http://cmbv.com/thesaurus/")
        self.graph.bind('baroque', BAROQUE)
        
        # Créer le ConceptScheme principal
        scheme_uri = BAROQUE.ThesaurusMusicalBaroque
        self.graph.add((scheme_uri, RDF.type, SKOS.ConceptScheme))
        self.graph.add((scheme_uri, DCTERMS.title, rdflib.Literal("Thésaurus de la musique baroque", lang="fr")))
        self.graph.add((scheme_uri, DCTERMS.created, rdflib.Literal(datetime.now().date().isoformat(), datatype=XSD.date)))
        self.graph.add((scheme_uri, DCTERMS.description, rdflib.Literal(
            "Thésaurus hiérarchisé des termes de la musique baroque généré par analyse automatique", lang="fr")))
        
        # Créer les Collections pour chaque catégorie
        collections = {}
        for category in sorted(self.df['final_category'].dropna().unique()):
            collection_uri = BAROQUE[f"collection_{category.lower().replace(' ', '_')}"]
            collections[category] = collection_uri
            self.graph.add((collection_uri, RDF.type, SKOS.Collection))
            self.graph.add((collection_uri, SKOS.prefLabel, rdflib.Literal(category.capitalize(), lang="fr")))
            self.graph.add((scheme_uri, DCTERMS.hasPart, collection_uri))
        
        # Fonction pour normaliser les termes en URI
        def term_to_uri(term):
            # Nettoyer et normaliser le terme pour l'URI
            clean = term.lower().replace(' ', '_').replace("'", '_').replace('-', '_')
            clean = ''.join(c for c in clean if c.isalnum() or c == '_')
            return BAROQUE[clean]
        
        # Traiter chaque terme et créer les concepts SKOS
        term_uris = {}  # Stocker les URI pour chaque terme
        
        for idx, row in self.df.iterrows():
            if pd.isna(row['original_term']):
                continue
                
            # Créer l'URI du concept
            term_uri = term_to_uri(row['original_term'])
            term_uris[row['original_term']] = term_uri
            
            # Ajouter le concept au graphe
            self.graph.add((term_uri, RDF.type, SKOS.Concept))
            self.graph.add((term_uri, SKOS.prefLabel, rdflib.Literal(row['original_term'], lang="fr")))
            
            # Lier au ConceptScheme
            self.graph.add((term_uri, SKOS.inScheme, scheme_uri))
            
            # Ajouter aux collections appropriées
            if pd.notna(row['final_category']) and row['final_category'] in collections:
                self.graph.add((collections[row['final_category']], SKOS.member, term_uri))
            
            # Ajouter les définitions si disponibles
            if 'definition_clean' in self.df.columns and pd.notna(row['definition_clean']):
                self.graph.add((term_uri, SKOS.definition, rdflib.Literal(row['definition_clean'], lang="fr")))
            
            # Ajouter les métadonnées de niveau taxonomique
            if pd.notna(row['taxo_level']):
                self.graph.add((term_uri, BAROQUE.taxonomicLevel, rdflib.Literal(int(row['taxo_level']))))
            
            # Marquer les termes représentatifs
            if pd.notna(row['is_representative']) and row['is_representative']:
                self.graph.add((term_uri, BAROQUE.isRepresentative, rdflib.Literal(True)))
        
        # Établir les relations hiérarchiques
        for idx, row in self.df.iterrows():
            if pd.isna(row['original_term']):
                continue
                
            term_uri = term_uris[row['original_term']]
            
            # Relations broader/narrower
            if 'hypernyms' in self.df.columns and pd.notna(row['hypernyms']):
                for hypernym in row['hypernyms'].split(','):
                    hypernym = hypernym.strip()
                    if hypernym in term_uris:
                        self.graph.add((term_uri, SKOS.broader, term_uris[hypernym]))
                        self.graph.add((term_uris[hypernym], SKOS.narrower, term_uri))
            
            if 'hyponyms' in self.df.columns and pd.notna(row['hyponyms']):
                for hyponym in row['hyponyms'].split(','):
                    hyponym = hyponym.strip()
                    if hyponym in term_uris:
                        self.graph.add((term_uri, SKOS.narrower, term_uris[hyponym]))
                        self.graph.add((term_uris[hyponym], SKOS.broader, term_uri))
            
            # Relations associatives basées sur la similarité sémantique
            if 'top_similar_terms' in self.df.columns and pd.notna(row['top_similar_terms']):
                similar_terms = row['top_similar_terms'].split(',')
                # Limiter aux 3 termes les plus similaires pour éviter la surcharge
                for sim_term in similar_terms[:3]:
                    sim_term = sim_term.strip()
                    if sim_term in term_uris and sim_term != row['original_term']:
                        # Éviter les relations duplicatives
                        if (term_uri, SKOS.related, term_uris[sim_term]) not in self.graph:
                            self.graph.add((term_uri, SKOS.related, term_uris[sim_term]))
                            self.graph.add((term_uris[sim_term], SKOS.related, term_uri))
        
        # Créer les TopConcepts (concepts de niveau 1)
        for idx, row in self.df[self.df['taxo_level'] == 1].iterrows():
            if pd.isna(row['original_term']):
                continue
                
            term_uri = term_uris[row['original_term']]
            self.graph.add((scheme_uri, SKOS.hasTopConcept, term_uri))
            self.graph.add((term_uri, SKOS.topConceptOf, scheme_uri))
        
        # Statistiques sur le graphe SKOS généré
        n_concepts = len([s for s, p, o in self.graph.triples((None, RDF.type, SKOS.Concept))])
        n_broader = len([s for s, p, o in self.graph.triples((None, SKOS.broader, None))])
        n_related = len([s for s, p, o in self.graph.triples((None, SKOS.related, None))])
        n_collections = len([s for s, p, o in self.graph.triples((None, RDF.type, SKOS.Collection))])
        
        logger.info(f"Structure SKOS générée:")
        logger.info(f"  - Concepts: {n_concepts}")
        logger.info(f"  - Relations hiérarchiques: {n_broader}")
        logger.info(f"  - Relations associatives: {n_related}")
        logger.info(f"  - Collections: {n_collections}")
        
        # Mémoriser le nombre de triplets pour les statistiques
        self.skos_stats = {
            'n_concepts': n_concepts,
            'n_broader': n_broader,
            'n_related': n_related,
            'n_collections': n_collections,
            'n_triplets': len(self.graph)
        }

    def validate_skos_structure(self) -> tuple:
        """
        Valide la qualité et la cohérence de la structure SKOS générée.
        
        Cette méthode vérifie:
        1. L'absence de cycles dans les relations hiérarchiques
        2. La cohérence des relations broader/narrower
        3. La connectivité du graphe
        4. La présence de propriétés requises pour chaque concept
        5. La conformité aux bonnes pratiques SKOS
        
        Returns:
            tuple: (valid, issues) où valid est un booléen et issues est un dictionnaire
                des problèmes détectés
        """
        logger.info("Validation de la structure SKOS...")
        
        issues = {
            'cycles': [],
            'inconsistent_relations': [],
            'orphan_concepts': [],
            'missing_properties': [],
            'other_issues': []
        }
        
        # 1. Vérifier l'absence de cycles dans les relations hiérarchiques
        try:
            # Construire un graphe orienté pour la détection de cycles
            import networkx as nx
            g = nx.DiGraph()
            
            # Ajouter les arêtes pour les relations broader
            for s, p, o in self.graph.triples((None, SKOS.broader, None)):
                s_str = str(s)
                o_str = str(o)
                g.add_edge(s_str, o_str)
            
            # Détecter les cycles
            cycles = list(nx.simple_cycles(g))
            
            if cycles:
                for cycle in cycles:
                    # Convertir les URIs en chaînes pour une meilleure lisibilité
                    readable_cycle = []
                    for uri in cycle:
                        # Extraire le terme à partir de l'URI
                        term = uri.split('/')[-1].replace('_', ' ')
                        readable_cycle.append(term)
                    
                    issues['cycles'].append(' -> '.join(readable_cycle))
                
                logger.warning(f"Cycles détectés dans les relations hiérarchiques: {len(cycles)}")
        except Exception as e:
            logger.error(f"Erreur lors de la détection de cycles: {e}")
            issues['other_issues'].append(f"Erreur de validation des cycles: {str(e)}")
        
        # 2. Vérifier la cohérence des relations broader/narrower
        for s, _, o in self.graph.triples((None, SKOS.broader, None)):
            # Vérifier que la relation inverse existe
            if (o, SKOS.narrower, s) not in self.graph:
                s_label = self.graph.value(s, SKOS.prefLabel)
                o_label = self.graph.value(o, SKOS.prefLabel)
                issue = f"Relation incohérente: {s_label} broader {o_label}, mais la relation narrower inverse manque"
                issues['inconsistent_relations'].append(issue)
        
        logger.info(f"Relations incohérentes: {len(issues['inconsistent_relations'])}")
        
        # 3. Vérifier les concepts orphelins (non connectés à un topConcept)
        concepts = set([s for s, p, o in self.graph.triples((None, RDF.type, SKOS.Concept))])
        connected_concepts = set()
        
        # Trouver tous les topConcepts
        top_concepts = set([o for s, p, o in self.graph.triples((None, SKOS.hasTopConcept, None))])
        
        # Fonction récursive pour explorer le graphe
        def explore_from(concept):
            connected_concepts.add(concept)
            # Explorer les concepts plus spécifiques
            for _, _, narrower in self.graph.triples((concept, SKOS.narrower, None)):
                if narrower not in connected_concepts:
                    explore_from(narrower)
        
        # Explorer à partir de chaque topConcept
        for top in top_concepts:
            explore_from(top)
        
        # Identifier les concepts orphelins
        orphans = concepts - connected_concepts
        
        if orphans:
            for orphan in orphans:
                label = self.graph.value(orphan, SKOS.prefLabel)
                issues['orphan_concepts'].append(str(label))
            
            logger.warning(f"Concepts orphelins détectés: {len(orphans)}")
        
        # 4. Vérifier les propriétés requises pour chaque concept
        required_props = [SKOS.prefLabel, SKOS.inScheme]
        
        for concept in concepts:
            missing = []
            for prop in required_props:
                if (concept, prop, None) not in self.graph:
                    missing.append(str(prop).split('#')[-1])
            
            if missing:
                label = self.graph.value(concept, SKOS.prefLabel) or concept
                issues['missing_properties'].append(f"{label}: {', '.join(missing)}")
        
        # 5. Statistiques de validation
        valid = (
            len(issues['cycles']) == 0 and
            len(issues['inconsistent_relations']) == 0 and
            len(issues['missing_properties']) == 0
        )
        
        # Nombre total de problèmes
        total_issues = sum(len(v) for v in issues.values())
        
        logger.info(f"Validation SKOS terminée: {'Valide' if valid else 'Non valide'}")
        logger.info(f"Total des problèmes: {total_issues}")
        
        # Sauvegarder le rapport de validation
        validation_path = self.output_dir / "skos_validation.json"
        with open(validation_path, 'w') as f:
            json.dump({
                'valid': valid,
                'total_issues': total_issues,
                'issues': issues
            }, f, indent=2)
        
        return valid, issues

    def export_skos(self) -> None:
        """
        Exporte le thésaurus SKOS dans différents formats.
        
        Cette méthode génère:
        1. Le fichier RDF/Turtle complet
        2. Une version JSON-LD pour l'interopérabilité web
        3. Une visualisation graphique de la structure
        4. Un export HTML pour consultation facile
        5. Un fichier de métadonnées décrivant le thésaurus
        """
        logger.info("Exportation du thésaurus SKOS...")
        
        # 1. Export au format Turtle (TTL)
        ttl_path = self.output_dir / "thesaurus_baroque.ttl"
        self.graph.serialize(destination=str(ttl_path), format="turtle")
        logger.info(f"Export Turtle généré: {ttl_path}")
        
        # 2. Export au format JSON-LD
        try:
            jsonld_path = self.output_dir / "thesaurus_baroque.jsonld"
            self.graph.serialize(destination=str(jsonld_path), format="json-ld")
            logger.info(f"Export JSON-LD généré: {jsonld_path}")
        except Exception as e:
            logger.warning(f"Impossible de générer l'export JSON-LD: {e}")
        
        # 3. Visualisation graphique de la structure SKOS
        try:
            import networkx as nx
            import matplotlib.pyplot as plt
            
            # Créer un graphe NetworkX
            g = nx.Graph()
            
            # Ajouter les nœuds (concepts)
            for s, p, o in self.graph.triples((None, RDF.type, SKOS.Concept)):
                label = str(self.graph.value(s, SKOS.prefLabel))
                g.add_node(str(s), label=label)
            
            # Ajouter les relations hiérarchiques
            for s, p, o in self.graph.triples((None, SKOS.broader, None)):
                g.add_edge(str(s), str(o), type='hierarchical')
            
            # Ajouter un sous-ensemble des relations associatives pour éviter la surcharge
            related_edges = [(str(s), str(o)) for s, p, o in self.graph.triples((None, SKOS.related, None))]
            # Limiter à un nombre raisonnable
            if len(related_edges) > 100:
                import random
                related_edges = random.sample(related_edges, 100)
            
            for s, o in related_edges:
                g.add_edge(s, o, type='associative')
            
            # Dessiner le graphe
            plt.figure(figsize=(12, 12))
            
            # Utiliser l'algorithme spring_layout pour une meilleure visualisation
            pos = nx.spring_layout(g, k=0.3, iterations=50)
            
            # Dessiner les nœuds
            nx.draw_networkx_nodes(g, pos, node_size=300, alpha=0.6)
            
            # Dessiner les arêtes hiérarchiques
            hierarchical_edges = [(u, v) for u, v, d in g.edges(data=True) if d.get('type') == 'hierarchical']
            nx.draw_networkx_edges(g, pos, edgelist=hierarchical_edges, width=1.0, alpha=0.5)
            
            # Dessiner les arêtes associatives
            associative_edges = [(u, v) for u, v, d in g.edges(data=True) if d.get('type') == 'associative']
            nx.draw_networkx_edges(g, pos, edgelist=associative_edges, width=0.5, alpha=0.3, style='dashed')
            
            # Dessiner les labels des nœuds (limités pour éviter la surcharge)
            # Sélectionner un sous-ensemble de nœuds à afficher
            nodes_to_label = list(g.nodes())[:50]  # Limiter à 50 nœuds
            labels = {n: g.nodes[n]['label'] for n in nodes_to_label}
            nx.draw_networkx_labels(g, pos, labels=labels, font_size=8)
            
            plt.title("Visualisation du thésaurus SKOS")
            plt.axis('off')
            plt.tight_layout()
            
            # Sauvegarder la visualisation
            vis_path = self.output_dir / "thesaurus_graph.png"
            plt.savefig(vis_path, dpi=300)
            plt.close()
            
            logger.info(f"Visualisation du graphe générée: {vis_path}")
        except Exception as e:
            logger.warning(f"Impossible de générer la visualisation du graphe: {e}")
        
        # 4. Export HTML pour consultation facile
        try:
            html_path = self.output_dir / "thesaurus_browser.html"
            
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write("""<!DOCTYPE html>
    <html lang="fr">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Thésaurus de la musique baroque</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 20px; color: #333; }
            h1 { color: #2c3e50; }
            .collection { margin-bottom: 30px; }
            .collection h2 { color: #3498db; border-bottom: 1px solid #eee; padding-bottom: 5px; }
            .concept { margin: 10px 0; padding: 10px; border-left: 3px solid #ddd; }
            .concept:hover { background-color: #f9f9f9; }
            .concept h3 { margin-top: 0; color: #2c3e50; }
            .concept p { margin: 5px 0; }
            .concept .relations { margin-top: 10px; font-size: 0.9em; }
            .concept .relations span { margin-right: 15px; }
            .level-1 { border-left-color: #e74c3c; }
            .level-2 { border-left-color: #f39c12; }
            .level-3 { border-left-color: #27ae60; }
            .search { margin-bottom: 20px; }
            .search input { padding: 8px; width: 300px; }
            .search button { padding: 8px 15px; background: #3498db; color: white; border: none; cursor: pointer; }
            .stats { margin-bottom: 20px; padding: 10px; background: #f8f9fa; border-radius: 5px; }
        </style>
    </head>
    <body>
        <h1>Thésaurus de la musique baroque</h1>
        
        <div class="stats">
            <p><strong>Statistiques:</strong> """)
                
                # Ajouter les statistiques
                f.write(f"<span>{self.skos_stats['n_concepts']} concepts</span> | ")
                f.write(f"<span>{self.skos_stats['n_broader']} relations hiérarchiques</span> | ")
                f.write(f"<span>{self.skos_stats['n_related']} relations associatives</span> | ")
                f.write(f"<span>{self.skos_stats['n_collections']} collections</span>")
                f.write("""</p>
        </div>
        
        <div class="search">
            <input type="text" id="searchInput" placeholder="Rechercher un terme...">
            <button onclick="searchConcepts()">Rechercher</button>
        </div>
        """)
                
                # Générer le contenu HTML pour chaque collection
                collections = [(s, self.graph.value(s, SKOS.prefLabel)) 
                            for s, p, o in self.graph.triples((None, RDF.type, SKOS.Collection))]
                
                for collection_uri, collection_label in collections:
                    f.write(f'<div class="collection">\n')
                    f.write(f'<h2>{collection_label}</h2>\n')
                    
                    # Trouver les concepts de cette collection
                    members = [o for s, p, o in self.graph.triples((collection_uri, SKOS.member, None))]
                    
                    for concept_uri in members:
                        # Extraire les informations du concept
                        label = self.graph.value(concept_uri, SKOS.prefLabel) or "Sans label"
                        definition = self.graph.value(concept_uri, SKOS.definition) or ""
                        
                        # Niveau taxonomique
                        taxo_level = self.graph.value(concept_uri, rdflib.URIRef("http://cmbv.com/thesaurus/taxonomicLevel"))
                        level_class = f"level-{taxo_level}" if taxo_level else ""
                        
                        # Relations
                        broader = [self.graph.value(o, SKOS.prefLabel) for o in self.graph.objects(concept_uri, SKOS.broader)]
                        narrower = [self.graph.value(o, SKOS.prefLabel) for o in self.graph.objects(concept_uri, SKOS.narrower)]
                        related = [self.graph.value(o, SKOS.prefLabel) for o in self.graph.objects(concept_uri, SKOS.related)]
                        
                        # Générer le HTML pour ce concept
                        f.write(f'<div class="concept {level_class}" data-term="{label}">\n')
                        f.write(f'<h3>{label}</h3>\n')
                        
                        if definition:
                            f.write(f'<p><em>{definition}</em></p>\n')
                        
                        f.write('<div class="relations">\n')
                        
                        if broader:
                            f.write('<span><strong>Termes plus généraux:</strong> ')
                            f.write(', '.join(str(b) for b in broader))
                            f.write('</span>\n')
                        
                        if narrower:
                            f.write('<span><strong>Termes plus spécifiques:</strong> ')
                            f.write(', '.join(str(n) for n in narrower))
                            f.write('</span>\n')
                        
                        if related:
                            f.write('<span><strong>Termes associés:</strong> ')
                            f.write(', '.join(str(r) for r in related))
                            f.write('</span>\n')
                        
                        f.write('</div>\n')  # Fin des relations
                        f.write('</div>\n')  # Fin du concept
                    
                    f.write('</div>\n')  # Fin de la collection
                
                # Ajouter le JavaScript pour la recherche
                f.write("""
        <script>
        function searchConcepts() {
            const searchTerm = document.getElementById('searchInput').value.toLowerCase();
            const concepts = document.querySelectorAll('.concept');
            
            concepts.forEach(concept => {
                const term = concept.getAttribute('data-term').toLowerCase();
                if (term.includes(searchTerm)) {
                    concept.style.display = 'block';
                    // Assurons-nous que la collection parente est visible
                    concept.closest('.collection').style.display = 'block';
                } else {
                    concept.style.display = 'none';
                }
            });
            
            // Cacher les collections vides
            document.querySelectorAll('.collection').forEach(collection => {
                const visibleConcepts = collection.querySelectorAll('.concept[style="display: block"]');
                if (visibleConcepts.length === 0) {
                    collection.style.display = 'none';
                }
            });
            
            // Si la recherche est vide, tout afficher
            if (searchTerm === '') {
                concepts.forEach(concept => {
                    concept.style.display = 'block';
                });
                document.querySelectorAll('.collection').forEach(collection => {
                    collection.style.display = 'block';
                });
            }
        }
        </script>
    </body>
    </html>""")
            
            logger.info(f"Export HTML généré: {html_path}")
        
        except Exception as e:
            logger.warning(f"Impossible de générer l'export HTML: {e}")
        
        # 5. Fichier de métadonnées
        metadata = {
            "title": "Thésaurus de la musique baroque",
            "description": "Thésaurus hiérarchisé des termes de la musique baroque généré par analyse automatique",
            "date": datetime.now().isoformat(),
            "version": "1.0",
            "generator": "BaroqueTermTaxonomyBuilder",
            "statistics": self.skos_stats,
            "exports": {
                "ttl": str(ttl_path.name),
                "jsonld": str(jsonld_path.name) if 'jsonld_path' in locals() else None,
                "html": str(html_path.name) if 'html_path' in locals() else None,
                "graph": str(vis_path.name) if 'vis_path' in locals() else None
            }
        }
        
        metadata_path = self.output_dir / "thesaurus_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Métadonnées générées: {metadata_path}")

    def generate_visualization(self) -> None:
        """
        Génère des visualisations avancées du thésaurus pour explorer la structure.
        
        Cette méthode crée:
        1. Une visualisation interactive D3.js du graphe de concepts
        2. Une heatmap des similarités entre catégories
        3. Un dendrogramme hiérarchique basé sur les relations
        4. Une carte de réseau 3D pour une exploration immersive
        """
        logger.info("Génération des visualisations avancées...")
        
        # 1. Visualisation interactive D3.js
        try:
            # Préparer les données pour D3.js (format JSON)
            d3_data = {
                "nodes": [],
                "links": []
            }
            
            # Ajouter les nœuds (concepts)
            node_ids = {}  # Mapper les URIs aux IDs numériques pour D3
            i = 0
            
            for s, p, o in self.graph.triples((None, RDF.type, SKOS.Concept)):
                label = str(self.graph.value(s, SKOS.prefLabel))
                category = None
                
                # Chercher à quelle collection ce concept appartient
                for coll, _, concept in self.graph.triples((None, SKOS.member, s)):
                    category = str(self.graph.value(coll, SKOS.prefLabel))
                    break
                
                # Déterminer le niveau taxonomique
                taxo_level = self.graph.value(s, rdflib.URIRef("http://cmbv.com/thesaurus/taxonomicLevel"))
                level = int(taxo_level) if taxo_level else 2
                
                # Stocker l'ID du nœud
                node_ids[str(s)] = i
                
                # Ajouter le nœud
                d3_data["nodes"].append({
                    "id": i,
                    "name": label,
                    "category": category or "Non classé",
                    "level": level
                })
                
                i += 1
            
            # Ajouter les liens (relations)
            for s, p, o in self.graph.triples((None, SKOS.broader, None)):
                if str(s) in node_ids and str(o) in node_ids:
                    d3_data["links"].append({
                        "source": node_ids[str(s)],
                        "target": node_ids[str(o)],
                        "type": "hierarchical"
                    })
            
            for s, p, o in self.graph.triples((None, SKOS.related, None)):
                if str(s) in node_ids and str(o) in node_ids:
                    d3_data["links"].append({
                        "source": node_ids[str(s)],
                        "target": node_ids[str(o)],
                        "type": "associative"
                    })
            
            # Sauvegarder les données pour D3.js
            d3_data_path = self.output_dir / "thesaurus_d3_data.json"
            with open(d3_data_path, 'w') as f:
                json.dump(d3_data, f)
            
            # Créer la page HTML avec la visualisation D3.js
            d3_html_path = self.output_dir / "thesaurus_interactive.html"
            
            with open(d3_html_path, 'w', encoding='utf-8') as f:
                f.write("""<!DOCTYPE html>
    <html lang="fr">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Visualisation interactive du thésaurus de la musique baroque</title>
        <script src="https://d3js.org/d3.v7.min.js"></script>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                overflow: hidden;
            }
            #container {
                width: 100vw;
                height: 100vh;
                position: relative;
            }
            .node {
                stroke: #fff;
                stroke-width: 1.5px;
            }
            .link {
                stroke-opacity: 0.6;
            }
            .hierarchical {
                stroke: #666;
            }
            .associative {
                stroke: #999;
                stroke-dasharray: 5,5;
            }
            .tooltip {
                position: absolute;
                background-color: white;
                padding: 10px;
                border-radius: 5px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
                pointer-events: none;
                opacity: 0;
                transition: opacity 0.3s;
            }
            .controls {
                position: absolute;
                top: 10px;
                left: 10px;
                z-index: 100;
                background: rgba(255,255,255,0.8);
                padding: 10px;
                border-radius: 5px;
            }
        </style>
    </head>
    <body>
        <div id="container">
            <div class="controls">
                <h3>Thésaurus de la musique baroque</h3>
                <div>
                    <label for="categoryFilter">Filtrer par catégorie:</label>
                    <select id="categoryFilter">
                        <option value="all">Toutes les catégories</option>
                    </select>
                </div>
                <div>
                    <label for="levelFilter">Niveau taxonomique:</label>
                    <select id="levelFilter">
                        <option value="all">Tous les niveaux</option>
                        <option value="1">Niveau 1 (Général)</option>
                        <option value="2">Niveau 2 (Intermédiaire)</option>
                        <option value="3">Niveau 3 (Spécifique)</option>
                    </select>
                </div>
                <div>
                    <input type="checkbox" id="showLabels" checked>
                    <label for="showLabels">Afficher les labels</label>
                </div>
                <div>
                    <label>Zoom:</label>
                    <button id="zoomIn">+</button>
                    <button id="zoomOut">-</button>
                    <button id="resetZoom">Reset</button>
                </div>
            </div>
            <div id="tooltip" class="tooltip"></div>
        </div>

        <script>
        // Charger les données
        d3.json("thesaurus_d3_data.json").then(function(data) {
            // Configuration
            const width = window.innerWidth;
            const height = window.innerHeight;
            const nodeRadius = d => d.level === 1 ? 8 : d.level === 2 ? 6 : 4;
            const colorScale = d3.scaleOrdinal(d3.schemeCategory10);
            
            // Créer le conteneur SVG
            const svg = d3.select("#container")
                .append("svg")
                .attr("width", width)
                .attr("height", height);
                
            // Créer un groupe pour la transformation
            const g = svg.append("g");
            
            // Zoom behavior
            const zoom = d3.zoom()
                .scaleExtent([0.1, 10])
                .on("zoom", (event) => {
                    g.attr("transform", event.transform);
                });
                
            svg.call(zoom);
            
            // Créer la simulation de force
            const simulation = d3.forceSimulation(data.nodes)
                .force("link", d3.forceLink(data.links).id(d => d.id).distance(80))
                .force("charge", d3.forceManyBody().strength(-100))
                .force("center", d3.forceCenter(width / 2, height / 2))
                .force("x", d3.forceX())
                .force("y", d3.forceY());
                
            // Créer les liens
            const link = g.append("g")
                .selectAll("line")
                .data(data.links)
                .enter().append("line")
                .attr("class", d => `link ${d.type}`)
                .attr("stroke-width", 1);
                
            // Créer les nœuds
            const node = g.append("g")
                .selectAll("circle")
                .data(data.nodes)
                .enter().append("circle")
                .attr("class", "node")
                .attr("r", nodeRadius)
                .attr("fill", d => colorScale(d.category))
                .call(d3.drag()
                    .on("start", dragstarted)
                    .on("drag", dragged)
                    .on("end", dragended));
                    
            // Créer les labels
            const label = g.append("g")
                .selectAll("text")
                .data(data.nodes)
                .enter().append("text")
                .attr("dx", 12)
                .attr("dy", ".35em")
                .text(d => d.name)
                .style("font-size", d => d.level === 1 ? "12px" : "10px")
                .style("opacity", 0.8);
                
            // Tooltip
            const tooltip = d3.select("#tooltip");
            
            node.on("mouseover", function(event, d) {
                tooltip.style("opacity", 1)
                    .html(`<strong>${d.name}</strong><br>Catégorie: ${d.category}<br>Niveau: ${d.level}`)
                    .style("left", (event.pageX + 10) + "px")
                    .style("top", (event.pageY - 10) + "px");
            })
            .on("mouseout", function() {
                tooltip.style("opacity", 0);
            });
            
            // Mise à jour des positions
            simulation.on("tick", () => {
                link
                    .attr("x1", d => d.source.x)
                    .attr("y1", d => d.source.y)
                    .attr("x2", d => d.target.x)
                    .attr("y2", d => d.target.y);
                    
                node
                    .attr("cx", d => d.x)
                    .attr("cy", d => d.y);
                    
                label
                    .attr("x", d => d.x)
                    .attr("y", d => d.y);
            });
            
            // Fonctions de drag
            function dragstarted(event, d) {
                if (!event.active) simulation.alphaTarget(0.3).restart();
                d.fx = d.x;
                d.fy = d.y;
            }
            
            function dragged(event, d) {
                d.fx = event.x;
                d.fy = event.y;
            }
            
            function dragended(event, d) {
                if (!event.active) simulation.alphaTarget(0);
                d.fx = null;
                d.fy = null;
            }
            
            // Filtrage
            const categories = [...new Set(data.nodes.map(d => d.category))];
            const categorySelect = d3.select("#categoryFilter");
            
            categories.forEach(category => {
                categorySelect.append("option")
                    .attr("value", category)
                    .text(category);
            });
            
            categorySelect.on("change", filterGraph);
            d3.select("#levelFilter").on("change", filterGraph);
            
            function filterGraph() {
                const selectedCategory = categorySelect.property("value");
                const selectedLevel = d3.select("#levelFilter").property("value");
                
                node.style("display", d => {
                    if (selectedCategory !== "all" && d.category !== selectedCategory) return "none";
                    if (selectedLevel !== "all" && d.level !== parseInt(selectedLevel)) return "none";
                    return null;
                });
                
                const visibleNodes = new Set(data.nodes
                    .filter(d => {
                        if (selectedCategory !== "all" && d.category !== selectedCategory) return false;
                        if (selectedLevel !== "all" && d.level !== parseInt(selectedLevel)) return false;
                        return true;
                    })
                    .map(d => d.id));
                    
                link.style("display", d => {
                    if (visibleNodes.has(d.source.id) && visibleNodes.has(d.target.id)) return null;
                    return "none";
                });
                
                label.style("display", d => {
                    if (selectedCategory !== "all" && d.category !== selectedCategory) return "none";
                    if (selectedLevel !== "all" && d.level !== parseInt(selectedLevel)) return "none";
                    return null;
                });
            }
            
            // Gestion des labels
            d3.select("#showLabels").on("change", function() {
                const showLabels = this.checked;
                label.style("display", showLabels ? null : "none");
            });
            
            // Zoom controls
            d3.select("#zoomIn").on("click", () => {
                svg.transition().duration(500).call(zoom.scaleBy, 1.5);
            });
            
            d3.select("#zoomOut").on("click", () => {
                svg.transition().duration(500).call(zoom.scaleBy, 0.67);
            });
            
            d3.select("#resetZoom").on("click", () => {
                svg.transition().duration(500).call(zoom.transform, d3.zoomIdentity);
            });
        });
        </script>
    </body>
    </html>""")
            
            logger.info(f"Visualisation D3.js interactive générée: {d3_html_path}")
            
        except Exception as e:
            logger.warning(f"Impossible de générer la visualisation D3.js: {e}")
        
        # 2. Heatmap des similarités entre catégories (si disponible)
        try:
            if hasattr(self, 'category_embeddings'):
                import matplotlib.pyplot as plt
                import seaborn as sns
                import numpy as np
                
                # Calculer la matrice de similarité entre catégories
                categories = list(self.category_embeddings.keys())
                n_categories = len(categories)
                similarity_matrix = np.zeros((n_categories, n_categories))
                
                for i, cat1 in enumerate(categories):
                    for j, cat2 in enumerate(categories):
                        embed1 = self.category_embeddings[cat1]
                        embed2 = self.category_embeddings[cat2]
                        similarity = cosine_similarity([embed1], [embed2])[0][0]
                        similarity_matrix[i, j] = similarity
                
                # Créer la heatmap
                plt.figure(figsize=(10, 8))
                sns.heatmap(similarity_matrix, annot=True, cmap="YlGnBu", 
                            xticklabels=categories, yticklabels=categories)
                plt.title("Similarité sémantique entre catégories")
                plt.tight_layout()
                
                heatmap_path = self.output_dir / "category_similarity_heatmap.png"
                plt.savefig(heatmap_path)
                plt.close()
                
                logger.info(f"Heatmap des similarités générée: {heatmap_path}")
        except Exception as e:
            logger.warning(f"Impossible de générer la heatmap des similarités: {e}")
        
        # 3. Dendrogramme hiérarchique
        try:
            import matplotlib.pyplot as plt
            import networkx as nx
            from scipy.cluster.hierarchy import dendrogram, linkage
            
            # Créer un graphe pour représenter les relations hiérarchiques
            g = nx.DiGraph()
            
            # Ajouter les nœuds et les relations
            for s, p, o in self.graph.triples((None, SKOS.broader, None)):
                s_label = str(self.graph.value(s, SKOS.prefLabel))
                o_label = str(self.graph.value(o, SKOS.prefLabel))
                g.add_edge(s_label, o_label)
            
            # Extraire les composantes connexes
            components = list(nx.weakly_connected_components(g))
            
            # Créer un dendrogramme pour chaque composante suffisamment grande
            plt.figure(figsize=(15, 10))
            
            for i, component in enumerate(components[:3]):  # Limiter à 3 composantes principales
                if len(component) < 5:  # Ignorer les petites composantes
                    continue
                    
                # Extraire le sous-graphe
                subgraph = g.subgraph(component)
                
                # Calculer les distances entre les nœuds (basées sur les chemins)
                dist_matrix = []
                nodes = list(subgraph.nodes())
                
                if len(nodes) > 50:  # Limiter aux 50 nœuds principaux si trop grand
                    nodes = sorted(nodes, key=lambda n: subgraph.degree(n), reverse=True)[:50]
                
                # Créer une matrice de distance basée sur la longueur des chemins
                for n1 in nodes:
                    row = []
                    for n2 in nodes:
                        if n1 == n2:
                            row.append(0)
                        else:
                            try:
                                # Utiliser la longueur du chemin comme distance
                                path_len = nx.shortest_path_length(subgraph, n1, n2)
                                row.append(path_len)
                            except nx.NetworkXNoPath:
                                # Pas de chemin, utiliser une valeur maximale
                                row.append(len(nodes))
                    dist_matrix.append(row)
                
                # Convertir en array numpy
                dist_array = np.array(dist_matrix)
                
                # Calculer le linkage pour le dendrogramme
                Z = linkage(dist_array, method='ward')
                
                # Créer un subplot
                plt.subplot(1, min(3, len(components)), i+1)
                dendrogram(Z, labels=nodes, orientation='right', leaf_font_size=8)
                plt.title(f"Composante {i+1}")
                
            plt.suptitle("Dendrogrammes des relations hiérarchiques")
            plt.tight_layout()
            
            dendro_path = self.output_dir / "hierarchy_dendrogram.png"
            plt.savefig(dendro_path)
            plt.close()
            
            logger.info(f"Dendrogramme hiérarchique généré: {dendro_path}")
        except Exception as e:
            logger.warning(f"Impossible de générer le dendrogramme: {e}")

    def generate_report(self) -> None:
        """
        Génère un rapport complet sur le thésaurus SKOS construit.
        
        Ce rapport inclut:
        1. Un résumé des résultats et statistiques 
        2. Une évaluation qualitative de la taxonomie
        3. Les visualisations générées
        4. Les recommandations pour l'amélioration
        5. La documentation technique du processus
        """
        logger.info("Génération du rapport complet...")
        
        report_path = self.output_dir / "thesaurus_final_report.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            # En-tête et introduction
            f.write("# Rapport de construction du thésaurus SKOS de la musique baroque\n\n")
            f.write(f"*Généré le {datetime.now().strftime('%d/%m/%Y à %H:%M')}*\n\n")
            
            f.write("## Résumé exécutif\n\n")
            f.write("Ce document présente les résultats de la construction automatique d'un thésaurus hiérarchisé ")
            f.write("des termes de la musique baroque. Le processus a combiné des techniques de traitement du langage naturel, ")
            f.write("d'apprentissage automatique et d'analyse de graphes pour structurer une liste de termes en un thésaurus ")
            f.write("SKOS normalisé prêt à l'emploi.\n\n")
            
            # Statistiques globales
            if hasattr(self, 'skos_stats'):
                f.write("### Statistiques principales\n\n")
                f.write(f"- **Concepts SKOS**: {self.skos_stats['n_concepts']}\n")
                f.write(f"- **Relations hiérarchiques**: {self.skos_stats['n_broader']}\n")
                f.write(f"- **Relations associatives**: {self.skos_stats['n_related']}\n")
                f.write(f"- **Collections thématiques**: {self.skos_stats['n_collections']}\n")
                f.write(f"- **Triplets RDF totaux**: {self.skos_stats['n_triplets']}\n\n")
            
            # Visualisations
            f.write("## Visualisations\n\n")
            
            # Ajouter les différentes visualisations si elles existent
            vis_files = {
                "Répartition des catégories": "final_categories.png",
                "Visualisation 2D des termes": "terms_visualization.png",
                "Structure du graphe de thésaurus": "thesaurus_graph.png",
                "Similarité entre catégories": "category_similarity_heatmap.png",
                "Dendrogramme hiérarchique": "hierarchy_dendrogram.png"
            }
            
            for desc, filename in vis_files.items():
                file_path = self.output_dir / filename
                if file_path.exists():
                    f.write(f"### {desc}\n\n")
                    f.write(f"![{desc}]({filename})\n\n")
            
            # Structure taxonomique
            f.write("## Structure taxonomique\n\n")
            
            # Afficher les catégories et leur distribution
            category_counts = self.df['final_category'].value_counts().to_dict()
            f.write("### Catégories principales\n\n")
            
            for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
                if pd.isna(category):
                    continue
                percentage = count / len(self.df) * 100
                f.write(f"- **{category}**: {count} termes ({percentage:.1f}%)\n")
            
            f.write("\n### Distribution des niveaux taxonomiques\n\n")
            
            level_counts = self.df['taxo_level'].value_counts().sort_index().to_dict()
            level_names = {1: "Général", 2: "Intermédiaire", 3: "Spécifique"}
            
            for level, count in level_counts.items():
                percentage = count / len(self.df) * 100
                level_name = level_names.get(level, f"Niveau {level}")
                f.write(f"- **{level_name}**: {count} termes ({percentage:.1f}%)\n")
            
            # Évaluation de la qualité
            f.write("\n## Évaluation de la qualité\n\n")
            
            # Si la validation a été effectuée
            if hasattr(self, 'validation_results'):
                valid, issues = self.validation_results
                
                if valid:
                    f.write("✅ **Validation SKOS réussie**: La structure du thésaurus est cohérente et respecte les normes SKOS.\n\n")
                else:
                    f.write("⚠️ **Problèmes de validation SKOS détectés**: Des améliorations sont recommandées.\n\n")
                    
                    if issues['cycles']:
                        f.write("### Cycles détectés\n\n")
                        for i, cycle in enumerate(issues['cycles'][:5]):  # Limiter à 5 exemples
                            f.write(f"{i+1}. {cycle}\n")
                        if len(issues['cycles']) > 5:
                            f.write(f"... et {len(issues['cycles'])-5} autres cycles\n")
                        f.write("\n")
                    
                    if issues['orphan_concepts']:
                        f.write("### Concepts orphelins\n\n")
                        f.write(f"**{len(issues['orphan_concepts'])}** concepts n'ont pas de chemin vers un concept de niveau supérieur.\n\n")
            
            # Cohérence sémantique
            f.write("### Cohérence sémantique\n\n")
            
            if hasattr(self, 'category_coherence'):
                sorted_coherence = {k: v for k, v in sorted(self.category_coherence.items(), key=lambda x: x[1], reverse=True)}
                
                f.write("Score de cohérence par catégorie (1.0 = parfaite cohérence):\n\n")
                for category, coherence in sorted_coherence.items():
                    quality = "Excellente" if coherence > 0.8 else "Bonne" if coherence > 0.6 else "Moyenne" if coherence > 0.4 else "Faible"
                    f.write(f"- **{category}**: {coherence:.2f} ({quality})\n")
                
                avg_coherence = sum(sorted_coherence.values()) / len(sorted_coherence)
                f.write(f"\nCohérence globale moyenne: **{avg_coherence:.2f}**\n\n")
            
    def export_skos(self, output_file: str, namespace: Optional[str] = None) -> None:
        """
        Exporte la hiérarchie des termes au format SKOS (RDF/Turtle).

        Parameters:
            output_file (str): Chemin du fichier de sortie (.ttl)
            namespace (str, optional): Base URI pour les concepts SKOS
        """
        logger.info(f"Export SKOS vers {output_file}")

        ns = Namespace(namespace or f"http://cmbv.fr/skos/{self.domain}/")

        g = Graph()
        g.bind("skos", SKOS)
        g.bind("rdfs", RDFS)

        for idx, row in self.df.iterrows():
            uri = URIRef(ns + row['conceptID'])
            g.add((uri, RDF.type, SKOS.Concept))
            g.add((uri, SKOS.prefLabel, Literal(row['original_term'], lang=self.language)))

            if self.definition_column and pd.notna(row.get('original_definition', '')):
                g.add((uri, SKOS.definition, Literal(row['original_definition'], lang=self.language)))

            if pd.notna(row.get('final_category')):
                g.add((uri, SKOS.note, Literal(f"Category: {row['final_category']}", lang=self.language)))

            # Ajouter les relations hiérarchiques
            if pd.notna(row.get('hypernyms')):
                for hyper in row['hypernyms'].split(','):
                    target_uri = URIRef(ns + self._find_concept_id(hyper))
                    g.add((uri, SKOS.broader, target_uri))

            if pd.notna(row.get('hyponyms')):
                for hypo in row['hyponyms'].split(','):
                    target_uri = URIRef(ns + self._find_concept_id(hypo))
                    g.add((uri, SKOS.narrower, target_uri))

        g.serialize(destination=output_file, format="turtle")
        logger.info(f"Export SKOS terminé : {output_file}")

    def _find_concept_id(self, term: str) -> str:
        """
        Trouve le conceptID correspondant à un terme donné.
        """
        row = self.df[self.df['original_term'].str.lower() == term.lower()]
        if not row.empty:
            return row.iloc[0]['conceptID']
        else:
            # Retourne un conceptID générique (au cas où)
            return re.sub(r'[^\w]', '_', term.lower())[:30] + '_' + str(abs(hash(term)) % 10000)

    def export_json(self, output_file: str) -> None:
        """
        Exporte toute la hiérarchie avec métadonnées en JSON.

        Parameters:
            output_file (str): Chemin du fichier de sortie (.json)
        """
        logger.info(f"Export JSON vers {output_file}")
        
        data = self.df.to_dict(orient="records")
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4, default=convert_numpy)

        logger.info(f"Export JSON terminé : {output_file}")