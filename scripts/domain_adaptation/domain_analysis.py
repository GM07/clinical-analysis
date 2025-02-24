from argparse import ArgumentParser
import logging

from src.domain_adaptation.domain_analyser import DomainAnalyser
from src.ontology.helper import OntologyHelper
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s',
    force=True # This ensures we override any existing logger configuration
)

parser = ArgumentParser(description='Runs inference on a dataset partition.')

parser.add_argument('--mimic_raw', type=str, required=True, help='Path to MIMIC-III raw data (noteevents.csv)')
parser.add_argument('--processed_mimic', type=str, required=False, help='Path to processed MIMIC-III data')
parser.add_argument('--snomed', type=str, help='Path to snomed ontology file (.owx)')
parser.add_argument('--snomed_cache', type=str, help='Path to snomed cache file')
parser.add_argument('--medcat', type=str, help='Path to medcat annotator checkpoint')
parser.add_argument('--out', type=str, help='Path to output file (or folder if save_separate_files is True)')
parser.add_argument('--save_separate_files', type=bool, default=False,help='Whether to save the class frequencies for each domain in separate files')
parser.add_argument('--limit', type=int, default=1000, help='Limit the number of notes to process')
parser.add_argument('--concept_limit', type=int, default=1000, help='Limit the number of concepts to process')
parser.add_argument('--domains', type=str, help='Comma-separated list of domains to process')

def main():

    args = parser.parse_args()

    print('Called with arguments : ', args)

    analyzer = DomainAnalyser(args.mimic_raw, args.processed_mimic, domains=args.domains.split(','))

    snomed, annotator = OntologyHelper.load_ontology_and_medcat_annotator(args.snomed, args.snomed_cache, args.medcat)
    analyzer.generate_domain_class_frequencies(snomed, annotator, limit=1000)
    if not args.save_separate_files:
        analyzer.save(args.out)
    else:
        os.mkdir(args.out)
        for domain in analyzer.domain_class_frequencies:
            analyzer.domain_class_frequencies[domain].save(args.out + f'/{domain}.dcf')

if __name__ == '__main__':
    main()
