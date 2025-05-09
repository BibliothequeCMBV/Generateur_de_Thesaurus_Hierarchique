import argparse
import logging
from enhanced_thesaurus_hierarchy_v2 import ThesaurusHierarchyBuilder

def main():
    parser = argparse.ArgumentParser(
        description="Construction automatique d'un thésaurus hiérarchique enrichi (SKOS)"
    )
    
    # Options principales
    parser.add_argument('--csv', type=str, required=True, help="Chemin vers le fichier CSV des termes")
    parser.add_argument('--domain', type=str, default="baroque_music", help="Domaine du thésaurus (ex: baroque_music)")
    parser.add_argument('--language', type=str, default="fr", help="Langue (ex: fr, en)")
    parser.add_argument('--max-depth', type=int, default=3, help="Profondeur maximale de la hiérarchie")
    parser.add_argument('--output-dir', type=str, default="./outputs", help="Répertoire pour les résultats")
    
    # Comportements optionnels
    parser.add_argument('--only-export', action='store_true', help="Uniquement exporter SKOS sans recalculer")
    parser.add_argument('--generate-report', action='store_true', help="Générer un rapport HTML complet des résultats")
    
    # Debug
    parser.add_argument('--verbose', action='store_true', help="Activer les logs détaillés")

    args = parser.parse_args()

    # Configuration du logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Instanciation du moteur
    builder = ThesaurusHierarchyBuilder(
        domain=args.domain,
        language=args.language,
        max_depth=args.max_depth,
        output_dir=args.output_dir
    )

    if not args.only_export:
        # Traitement complet
        builder.load_data(csv_file=args.csv)
        builder.generate_embeddings()
        builder.classify_terms()
    
    # Export SKOS final
    builder.export_skos()

    # Génération optionnelle du rapport
    if args.generate_report:
        builder.generate_report()

if __name__ == "__main__":
    main()
