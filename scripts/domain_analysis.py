from argparse import ArgumentParser
import logging

from src.domain_adaptation.domain_analyser import DomainAnalyser
from src.ontology.helper import OntologyHelper


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
parser.add_argument('--out', type=str, help='Path to output file')

def main():

    args = parser.parse_args()

    print('Called with arguments : ', args)

    analyzer = DomainAnalyser(args.mimic_raw, args.processed_mimic)

    snomed, annotator = OntologyHelper.load_ontology_and_medcat_annotator(args.snomed, args.snomed_cache, args.medcat)
    analyzer.generate_domain_class_frequencies(snomed, annotator, limit=1000)
    analyzer.save(args.out)

if __name__ == '__main__':
    main()
