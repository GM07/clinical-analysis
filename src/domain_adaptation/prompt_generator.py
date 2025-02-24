import pandas as pd

from src.data.dataset import PrunedConceptDataset
from src.generation.templates import BHC_BASE_TEMPLATE, DOMAIN_ADAPTATION_TEMPLATE, PRUNED_CONCEPT_TEMPLATE
from src.ontology.snomed import Snomed

class PromptGenerator:
    """
    Class used to generate prompts for the BHC pipeline
    """

    def __init__(self, mimic: str | pd.DataFrame, template: str):
        """
        Args:
            mimic: Path to the mimic dataset processed
            template: Template to use for the prompts
        """
        if isinstance(mimic, str):
            self.mimic = mimic
            self.dataset = pd.read_csv(self.mimic)
        else:
            self.mimic = None
            self.dataset = mimic

        self.template = template

        assert '{clinical_notes}' in self.template, 'The tag "{clinical_notes}" must be present in the template'

    def generate_prompts(self,):
        """
        Generate prompts for the BHC pipeline
        """
        self.dataset = self.dataset[self.dataset['CATEGORY'] != 'Discharge summary']

        def admission_to_prompt(clinical_notes_series):
            clinical_notes = clinical_notes_series.tolist()

            clinical_note_string = ''
            for i, note in enumerate(clinical_notes):
                clinical_note_string += f'Clinical note {i+1}:\n{note}\n'

            prompt = self.template.format(clinical_notes=clinical_note_string, domain='%{domain}%')
            return prompt

        prompts = self.dataset.groupby('HADM_ID')['TEXT'].aggregate(admission_to_prompt)
        return pd.DataFrame({'HADM_ID': prompts.index, 'PROMPT': prompts.values})


class DomainAdaptationPromptGenerator(PromptGenerator):
    """
    Class used to generate prompts for the domain adaptation pipeline to establish a baseline for the domain adaptation task
    """

    def __init__(self, mimic: str | pd.DataFrame, domains: list[str] = ['Nursing', 'ECG', 'Radiology']):
        super().__init__(mimic, DOMAIN_ADAPTATION_TEMPLATE)
        self.domains = domains

    def generate_prompts(self):
        """
        Generate prompts for the domain adaptation pipeline
        """
        clinical_notes_prompts = super().generate_prompts()
        hadm_ids, prompts, domains = [], [], []

        for _, row in clinical_notes_prompts.iterrows():
            for domain in self.domains:
                hadm_ids.append(row['HADM_ID'])
                prompts.append(row['PROMPT'].replace('%{domain}%', domain))
                domains.append(domain)

        return pd.DataFrame({'HADM_ID': hadm_ids, 'PROMPT': prompts, 'CATEGORY': domains})

class BHCPromptGenerator(PromptGenerator):
    """
    Class used to generate prompts for the BHC pipeline
    """

    def __init__(self, mimic: str | pd.DataFrame):
        super().__init__(mimic, BHC_BASE_TEMPLATE)

    def generate_prompts(self,):
        """
        Generate prompts for the BHC pipeline
        """
        bhc = self.dataset[self.dataset['BHC'].notna()]
        prompts = super().generate_prompts()
        return pd.merge(bhc, prompts, on='HADM_ID', how='left').drop(columns=['NOTE_ORDER'])


class PrunedConceptPromptGenerator(PromptGenerator):
    """
    Class used to generate prompts for the pruned concept pipeline
    """

    def __init__(self, mimic: str | pd.DataFrame, snomed: Snomed, input_columns: list[str] = None):
        """
        Args:
            mimic: Path to the mimic dataset processed
            snomed: Snomed ontology
            input_column: Column to use for the input (column must be a valid PrunedConceptDataset input column)
        """
        super().__init__(mimic, PRUNED_CONCEPT_TEMPLATE)
        self.snomed = snomed
        self.input_columns = input_columns

        for input_col in self.input_columns:
            assert PrunedConceptDataset.valid_pruned_concept_column(input_col), f'The input column "{input_col}" is not a valid column. It should be of a valid PrunedConceptDataset input column'

    def generate_prompts(self,):
        """
        Generate prompts for the BHC pipeline
        """
        self.dataset = self.dataset[self.dataset['CATEGORY'] != 'Discharge summary']

        def admission_to_prompt(clinical_notes_series):
            clinical_notes = clinical_notes_series.tolist()

            note_strings = []
            for i, note in enumerate(clinical_notes):
                clinical_note_string = ''
                if isinstance(note, str):
                    print(note)

                for concept_id, sentence in note.items():
                    if sentence == 'N/A':
                        continue

                    concept_label = self.snomed.get_label_from_id(concept_id)
                    clinical_note_string += f'{concept_label} : {sentence}\n'

                if len(clinical_note_string.strip()) > 0:
                    note_strings.append(clinical_note_string)
            if len(note_strings) == 0:
                return 'N/A'

            note_strings = list(filter(lambda x: x != '', note_strings))
            note_strings = [f'### Clinical note {i+1}\n{note}' for i, note in enumerate(note_strings)]
            prompt = self.template.format(clinical_notes='\n' + '\n'.join(note_strings))
            return prompt

        prompts = {}
        hadm_ids = []
        for input_column in self.input_columns:
            p = self.dataset.groupby('HADM_ID')[input_column].aggregate(admission_to_prompt)
            prompts[f'{input_column}_verbalizer_prompt'] = p.values

            if len(hadm_ids) == 0:
                hadm_ids = p.index
            else:
                assert (hadm_ids == p.index).all(), 'The HADM_IDs are not the same for all input columns'

        return pd.DataFrame({'HADM_ID': hadm_ids} | prompts)

        