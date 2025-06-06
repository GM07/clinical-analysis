import ast
from collections import defaultdict
import itertools
import logging
import os
import re
import uuid
from typing import Dict, List

from datasets import Dataset as HuggingFaceDataset, concatenate_datasets
import pandas as pd
from tqdm import tqdm

from src.data.dataset import Dataset, PrunedConceptDataset, VerbalizedExtractionDataset
from src.domain_adaptation.domain_class_frequency import DomainClassFrequency
from src.ontology.snomed import Snomed

logger = logging.getLogger(__name__)

standard_domain_mapping = {
    'ecg': 'ECG',
    'ECG': 'ECG',
    'physician_': 'Physician',
    'Nursing': 'Nursing',
    'nursing': 'Nursing',
    'nursing_other': 'Nursing',
    'radiology': 'Radiology',
    'Radiology': 'Radiology'
}


class HumanEvaluationDataset(Dataset):
    """
    Dataset needed to perform a human evaluation (which will generate the right folder structure for human evaluation)


    The columns should be, for each domain and method combinaison :
        - hadm_id : Admission id in MIMIC
        - notes : Clinical notes associated to the admission
        - {method}_{domain}_structured : Extractions pruned according to domain (as a dictionary)
        - {method}_{domain}_summary : Final summary generated by this method for this domain
    """

    def __init__(self, dataset_path = None, data = None):
        """
        Args:
            domains: List of domains to evaluate (only names according to `standard_domain_mapping` at top of the file)
            methods: List of methods to consider (constrained, beam, normal)
            dataset_path: Path to dataset (see `Dataset`)
            data: DataFramce (see `Dataset`)
        """
        super().__init__(dataset_path, data)

        self.verify()

    def verify(self):
        for column in ['id', 'notes', 'admission_id', 'structured_summary', 'summary', 'method', 'domain']:
            assert column in self.data.columns, f'Column {column} should be in the dataset'
                     
class HumanEvaluation:

    """
    In order to perform human evaluation, we need to have a dataset with the following columns:

    - id: Unique identifier for the sample generated
    - hadm_id: Unique identifier for the admission in the MIMIC-III dataset
    - clinical_notes : Clinical notes that were used to generate the summary
    - summary: Summary that was generated by the model
    - expected_domain : The domain that the summary is expected to be relevant to
    - method : Which method was used to generate the summary
    """

    def __init__(self, domains: List[str], methods: List[str], snomed: Snomed):
        """
        Args:
            domains: List of domains to consider
            methods: List of methods to consider
            pruned_dataset_paths: Path to pruned dataset (must be valid PrunedConceptDataset containing the methods and domains)
            verbalized_dataset_paths: Path to verbalized dataset (must be valid VerbalizedExtractionDataset containing the methods and domains)
            snomed: Snomed ontology
        """
        self.domains = domains
        self.methods = methods
        self.snomed = snomed

        self.verify_domains()

    def verify_domains(self):
        valid_domains = list(standard_domain_mapping.keys())
        for domain in self.domains:
            assert domain in valid_domains, f'Domain {domain} not recognized, here is the list of valid domains {valid_domains}'

    def create_columns(self):
        pruned_columns = []
        verbalized_columns = []
        for method in self.methods:
            for domain in self.domains:
                pruned_columns.append(f'{method}_{domain}')
                verbalized_columns.append(f'{method}_{domain}_verbalized')
        return pruned_columns, verbalized_columns

    def generate(self, pruned_dataset_paths: List[str], verbalized_dataset_paths: List[str]):

        final_dataset: HuggingFaceDataset = None

        for pruned_dataset_path, verbalized_dataset_path in zip(pruned_dataset_paths, verbalized_dataset_paths):
            current_dataset = self.generate_single(pruned_dataset_path, verbalized_dataset_path)
            final_dataset = concatenate_datasets(list(filter(lambda x: x is not None, [final_dataset, current_dataset])))

        return final_dataset
    
    def generate_single(self, pruned_dataset_path: str, verbalized_dataset_path: str, remove_same_exractions: bool = True):

        pruned_columns, verbalized_columns = self.create_columns()

        self.pruned_dataset = PrunedConceptDataset(columns=pruned_columns, dataset_path=pruned_dataset_path)
        self.verbalized_dataset = VerbalizedExtractionDataset(columns=verbalized_columns, dataset_path=verbalized_dataset_path)

        dataset = defaultdict(list)
        notes_per_admission_id = self.pruned_dataset.data.groupby('HADM_ID')['TEXT'].aggregate(self.admission_to_prompt)

        for i in tqdm(range(len(self.verbalized_dataset.data)), desc='Generating human evaluation samples'):
            admission_id = self.verbalized_dataset.data['HADM_ID'].iloc[i]
            notes = notes_per_admission_id[admission_id]

            for method in self.methods:
                for domain in self.domains:
                    
                    pruned_col = f'{method}_{domain}'
                    verbalized_col = pruned_col + '_verbalized'
                    summary = self.verbalized_dataset.data[verbalized_col].iloc[i]
                    structured_summary = self.get_pruned_extractions_of_admission(admission_id, pruned_col, remove_same_exractions)

                    if not self.summary_valid(summary) or not self.summary_valid(structured_summary[0]):
                        continue

                    dataset['id'].append(str(uuid.uuid4()))
                    dataset['notes'].append(notes)
                    dataset['admission_id'].append(admission_id)
                    dataset['structured_summary'].append(structured_summary[0])
                    dataset['structured_summary_dict'].append(structured_summary[1])
                    dataset['summary'].append(summary)
                    dataset['method'].append(method)
                    dataset['domain'].append(standard_domain_mapping[domain])

        return HuggingFaceDataset.from_dict(dataset)
    
    def summary_valid(self, summary: str):
        return summary is not None and isinstance(summary, str) and len(summary) > 0

    def admission_to_prompt(self, clinical_notes_series):
        return '\n\n'.join([f'### Clinical note {i+1}\n{note}' for i, note in enumerate(clinical_notes_series.tolist())])

    def get_pruned_extractions_of_admission(self, admission_id: float, pruned_domain_col: str, remove_same_extractions: bool = True, dict_format = False):
        """
        Returns the pruned extractions according to a domain of an admission. This assumes that `pruned_domain_col` contains
        the pruned extractions of a clinical note
        """
        def admission_to_prompt(clinical_notes_series):
            clinical_notes = clinical_notes_series.tolist()
            note_strings = []
            note_dicts = []
            for _, note in enumerate(clinical_notes):
                extraction_set = set([])
                clinical_note_string = ''
                concepts = {}

                for concept_id, extraction in note.items():
                    if extraction == 'N/A':
                        continue

                    if remove_same_extractions and extraction.lower().strip() in extraction_set:
                        continue

                    concept_label = self.snomed.get_label_from_id(concept_id)
                    concepts[f'{concept_label}-{concept_id}'] = extraction
                    clinical_note_string += f'- {concept_label} : {extraction}\n'

                    extraction_set.add(extraction.lower().strip())

                if len(clinical_note_string.strip()) > 0:
                    note_strings.append(clinical_note_string)
                    note_dicts.append(concepts)
                
            if len(note_strings) == 0:
                return 'N/A', str([])

            note_strings = list(filter(lambda x: x != '', note_strings))
            note_strings = [f'### Clinical note {i+1}\n{note}' for i, note in enumerate(note_strings)]
            return '\n\n'.join(note_strings), str(note_dicts)

        return self.pruned_dataset.data.groupby('HADM_ID')[pruned_domain_col].aggregate(admission_to_prompt)[admission_id]


class HumanEvaluationFilesGenerator:

    CLINICAL_NOTES_FILENAME = 'notes'
    SUMMARY_FILENAME = 'summary'
    STRUCTURED_SUMMARY_FILENAME = 'structured_summary'


    def __init__(self, dcfs: Dict[str, DomainClassFrequency], snomed: Snomed, human_evaluation_dataset_path: str = None, human_evaluation_dataset: pd.DataFrame = None):
        self.dataset = HumanEvaluationDataset(dataset_path=human_evaluation_dataset_path, data=human_evaluation_dataset)
        self.dcfs = dcfs
        self.snomed = snomed
    
    def generate_folder_structure(self, path: str):
        """
        Will generate the folder structure to facilitate human evaluation

        Args:
            path: Path where the folders and files will be stored
        """

        if path[-1] != '/':
            path += '/'

        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        domains = self.dataset.data['domain'].unique()

        for domain in domains:
            os.makedirs(path + domain, exist_ok=True)

        def create_domain_list_sheet(writer, formats, domain):

            header_format, bg_text_format, link_text_format, text_format, eval_format = formats 
            domain_sheet = writer.sheets['Domain Concepts']

            # Write headers for the table
            domain_sheet.write(0, 0, 'Concept', header_format)

            current_row = 1

            # Process all notes
            for id in self.dcfs[domain].get_concepts(separate=True)[0]:

                # Create URL for the hyperlink
                label = self.snomed.get_label_from_id(id)
                url = f"https://browser.ihtsdotools.org/?perspective=full&conceptId1={id}&edition=MAIN/2025-04-01&release=&languages=en"
                    
                # Write concept label as a hyperlink
                domain_sheet.write_url(
                    current_row, 
                    0,
                    url,
                    string=label,
                    cell_format=link_text_format
                )
                    
                current_row += 1

            # If there were no items to write
            if current_row == 1:
                domain_sheet.write(1, 0, 'N/A', text_format)
                domain_sheet.write(1, 1, '', text_format)
            
            # Format cell sizes
            domain_sheet.set_column('A:A', 200, text_format)

        def create_evaluation_sheet(sheet, writer, formats, criteria):
            """
            The evaluation for the structured summary contains the following columns :
            Criteria, Score, Comments

            Criteria : 
            - Groundedness : The extracted values are grounded on the clinical note and only contain factual content
            - Completeness : The concepts extracted regroup all concepts in the expected domain that are present in the clinical notes. A structured summary
                is incomplete if a concept that is present in the domain concepts (see "Domain concepts" sheet) is not present in the structured summary, but
                is present in the clinical note
            - Relevance : The extracted values are relevant to the concept
            """
            header_format, bg_text_format, link_text_format, text_format, eval_format = formats 
            eval_sheet = writer.sheets[sheet]
            eval_sheet.set_column('A:A', 200, text_format) # Criteria column
            eval_sheet.set_column('B:B', 10, text_format) # Score column
            eval_sheet.set_column('C:C', 40, text_format) # Comments column
            
            # Apply formatting to evaluation headers
            for col_num, header in enumerate(['Criteria', 'Score', 'Comments']):
                eval_sheet.write(0, col_num, header, header_format)
            
            eval_sheet.data_validation(
                    1,              # First row (0-indexed)
                    1,              # First column (0-indexed, so column B)
                    3,              # Last row
                    1,              # Last column (still column B)
                    {
                        'validate': 'list',
                        'source': list(map(lambda x: str(x), range(6))),
                        'dropdown': True,
                        'input_title': 'Enter the score',
                        'input_message': 'Score must be between 0 and 5'
                    }
                )
                
            # If you need to add criteria rows to the evaluation sheet
            for i, criterion in enumerate(criteria):
                eval_sheet.write(i+1, 0, criterion, text_format)  # Write the criteria in column A

        def create_structured_summary(row, writer, formats):
            """
            Creates the structured summary sheet

            Structured summary sheet contains the following columns : 
            concept, extractions
            """

            header_format, bg_text_format, link_text_format, text_format, eval_format = formats 
            structured_summary_sheet = writer.sheets['Structured Summary']

            # Write headers for the table
            structured_summary_sheet.write(0, 0, 'Concept', header_format)
            structured_summary_sheet.write(0, 1, 'Extraction', header_format)
            structured_summary_sheet.write(0, 2, 'Evaluation', header_format)

            # Access the dictionary directly - assuming it's always present and a dictionary
            structured_summary_dict: Dict[str, str] = ast.literal_eval(row['structured_summary_dict'])

            # Track row index
            current_row = 1

            # Process all notes
            for note_index, note in enumerate(structured_summary_dict):
                structured_summary_sheet.write(current_row, 0, f'CLINICAL NOTE {note_index + 1}', bg_text_format)
                current_row += 2
                
                for concept_id_str, extraction_text in note.items():
                    concept_parts = concept_id_str.split('-')
                    concept_label = concept_parts[0]
                    concept_id = concept_parts[1] if len(concept_parts) > 1 else ""
                    
                    # Create URL for the hyperlink
                    url = f"https://browser.ihtsdotools.org/?perspective=full&conceptId1={concept_id}&edition=MAIN/2025-04-01&release=&languages=en"
                    
                    # Write concept label as a hyperlink
                    structured_summary_sheet.write_url(
                        current_row, 0,         # Row, Column
                        url,                    # URL
                        string=concept_label,   # Display text
                        cell_format=link_text_format
                    )
                    
                    # Write extraction text in the adjacent cell
                    structured_summary_sheet.write(current_row, 1, extraction_text, text_format)
                    current_row += 1
                
                current_row += 1

            # If there were no items to write
            if current_row == 1:
                structured_summary_sheet.write(1, 0, 'N/A', text_format)
                structured_summary_sheet.write(1, 1, '', text_format)
            
            # Format cell sizes
            structured_summary_sheet = writer.sheets['Structured Summary']
            structured_summary_sheet.set_column('A:A', 60, text_format)
            structured_summary_sheet.set_column('B:B', 240, text_format)
            

        def row_to_folder(row):

            excel_path = path + '/' + str(row['domain']) + '/' + str(row['domain']) + '_' + str(row['folder_id']) + '.xlsx'
            structured_summary = pd.DataFrame({'Structured Summary': [row['structured_summary']]})
            clinical_note = pd.DataFrame({'Clinical Notes': [row['notes']]})
            summary = pd.DataFrame({'Summary': [row['summary']]})
            structured_summary_evaluation = pd.DataFrame({
                'Criteria': [
                    'Groundedness : The extracted values are grounded on the clinical note and only contain factual content',
                    'Relevance : The extracted values are relevant to the concept.',
                    f'Completeness : The concepts extracted regroup all concepts in the "" domain that are present in the clinical notes. A structured summary is incomplete if a concept that is present in the domain concepts (see "Domain concepts" sheet) is not present in the structured summary, but is present in the clinical note',
                ],
                'Score': [0, 0, 0],
                'Comments': ['', '', '']
            })
            summary_evaluation = pd.DataFrame({
                'Criteria': [
                    'Groundedness : The extracted values are grounded on the clinical note and only contain factual content.',
                    'Relevance : The summary is relevant to the domain.',
                    'Fluency : The summary is presented smoothly and clearly, making it easy to read and understand.',
                ],
                'Score': [0, 0, 0],
                'Comments': ['', '', ''],
            })
            domain = pd.DataFrame({
                'Domain': []
            })

            with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
                # Write dataframes to Excel sheets
                clinical_note.to_excel(writer, sheet_name='Clinical Notes', index=False)
                structured_summary.to_excel(writer, sheet_name='Structured Summary', index=False)
                summary.to_excel(writer, sheet_name='Summary', index=False)
                structured_summary_evaluation.to_excel(writer, sheet_name='Structured Summary Evaluation', index=False)
                summary_evaluation.to_excel(writer, sheet_name='Summary Evaluation', index=False)
                domain.to_excel(writer, sheet_name='Domain Concepts', index=False)

                # Get workbook and worksheet objects for formatting
                workbook = writer.book
                text_format = workbook.add_format({
                    'text_wrap': True, 
                    'valign': 'top',
                    'font_size': 20
                })

                bg_text_format = workbook.add_format({
                    'text_wrap': True, 
                    'valign': 'top',
                    'font_size': 20,
                    'bold': True,
                })

                link_text_format = workbook.add_format({
                    'text_wrap': True, 
                    'valign': 'top',
                    'font_size': 20,
                    'font_color': 'blue',
                    'underline': 1,
                })
                
                header_format = workbook.add_format({
                    'bold': True, 
                    'border': 1, 
                    'bg_color': '#D9E1F2',
                    'font_size': 20
                })
                
                eval_format = workbook.add_format({
                    'font_size': 20
                })

                formats = [header_format, bg_text_format, link_text_format, text_format, eval_format]
                create_structured_summary(row, writer, formats)

                notes_sheet = writer.sheets['Clinical Notes']
                notes_sheet.write(0, 0, 'Clinical Notes', header_format)

                notes_sheet.set_column('A:A', 200, text_format)
                notes_sheet.set_row(1, 5000)
                
                summary_sheet = writer.sheets['Summary']
                summary_sheet.write(0, 0, 'Summary', header_format)
                summary_sheet.set_column('A:A', 200, text_format)
                summary_sheet.set_row(1, 5000)
                
                create_evaluation_sheet('Structured Summary Evaluation', writer, formats, structured_summary_evaluation['Criteria'])
                create_evaluation_sheet('Summary Evaluation', writer, formats, summary_evaluation['Criteria'])
                create_domain_list_sheet(writer, formats, str(row['domain']))

        self.dataset.data['folder_id'] = list(range(len(self.dataset.data)))
        for _, row in self.dataset.data.iterrows():
            row_to_folder(row)

        self.dataset.data.to_csv(path + 'human_evaluation.csv', index=False)
