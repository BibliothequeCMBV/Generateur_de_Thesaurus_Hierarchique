#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import argparse

def joindre_tsv(fichier_tsv1, fichier_tsv2, fichier_sortie):
    """
    Joindre deux fichiers TSV en remplaçant les valeurs dans "skos:definition" 
    du premier fichier par les valeurs de "Définition" du deuxième fichier 
    lorsqu'il y a correspondance entre "skos:prefLabel" du premier fichier et 
    "Concept" du deuxième.
    
    Args:
        fichier_tsv1: Chemin vers le premier fichier TSV
        fichier_tsv2: Chemin vers le deuxième fichier TSV
        fichier_sortie: Chemin vers le fichier de sortie
    """
    print(f"Lecture du fichier 1: {fichier_tsv1}")
    tsv1 = pd.read_csv(fichier_tsv1, sep='\t', encoding='utf-8')
    
    print(f"Lecture du fichier 2: {fichier_tsv2}")
    tsv2 = pd.read_csv(fichier_tsv2, sep='\t', encoding='utf-8')
    
    print("Colonnes du fichier 1:", tsv1.columns.tolist())
    print("Colonnes du fichier 2:", tsv2.columns.tolist())
    
    # Vérifie que les colonnes nécessaires existent
    colonnes_requises_1 = ["skos:prefLabel", "skos:definition"]
    colonnes_requises_2 = ["Concept", "Définition"]
    
    for col in colonnes_requises_1:
        if col not in tsv1.columns:
            raise ValueError(f"Colonne '{col}' non trouvée dans le premier fichier")
    
    for col in colonnes_requises_2:
        if col not in tsv2.columns:
            raise ValueError(f"Colonne '{col}' non trouvée dans le deuxième fichier")
    
    # Création d'un dictionnaire de correspondance (Concept -> Définition) à partir du deuxième fichier
    dict_definitions = dict(zip(tsv2["Concept"], tsv2["Définition"]))
    
    # Fonction pour remplacer la définition si une correspondance est trouvée
    def remplacer_definition(row):
        if row["skos:prefLabel"] in dict_definitions:
            return dict_definitions[row["skos:prefLabel"]]
        return row["skos:definition"]
    
    # Appliquer le remplacement
    print("Application des remplacements de définitions...")
    tsv1["skos:definition"] = tsv1.apply(remplacer_definition, axis=1)
    
    # Enregistrer le résultat
    print(f"Enregistrement du résultat dans: {fichier_sortie}")
    tsv1.to_csv(fichier_sortie, sep='\t', index=False, encoding='utf-8')
    
    # Compte le nombre de remplacements effectués
    nb_correspondances = sum(1 for label in tsv1["skos:prefLabel"] if label in dict_definitions)
    print(f"Terminé! {nb_correspondances} définitions ont été remplacées sur {len(tsv1)} entrées.")

def main():
    parser = argparse.ArgumentParser(description='Joindre deux fichiers TSV avec remplacement de définitions')
    parser.add_argument('fichier_tsv1', help='Chemin vers le premier fichier TSV (contenant skos:prefLabel et skos:definition)')
    parser.add_argument('fichier_tsv2', help='Chemin vers le deuxième fichier TSV (contenant Concept et Définition)')
    parser.add_argument('fichier_sortie', help='Chemin vers le fichier de sortie')
    
    args = parser.parse_args()
    
    joindre_tsv(args.fichier_tsv1, args.fichier_tsv2, args.fichier_sortie)

if __name__ == "__main__":
    main()