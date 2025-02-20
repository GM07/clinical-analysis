import pandas as pd

from src.generation.templates import BHC_BASE_TEMPLATE, DOMAIN_ADAPTATION_TEMPLATE

class PromptGenerator:
    """
    Class used to generate prompts for the BHC pipeline
    """

    def __init__(self, mimic_bhc_path: str, template: str):
        """
        Args:
            mimic_bhc_path: Path to the mimic dataset containing the BHC section extracted from the discharge summaries
            template: Template to use for the prompts
        """
        self.mimic_bhc_path = mimic_bhc_path
        self.dataset = pd.read_csv(self.mimic_bhc_path)
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
    Class used to generate prompts for the domain adaptation pipeline
    """

    def __init__(self, mimic_bhc_path: str, domains: list[str] = ['Nursing', 'ECG', 'Radiology']):
        super().__init__(mimic_bhc_path, DOMAIN_ADAPTATION_TEMPLATE)
        self.domains = domains

    def generate_prompts(self,):
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

    def __init__(self, mimic_bhc_path: str):
        super().__init__(mimic_bhc_path, BHC_BASE_TEMPLATE)

    def generate_prompts(self,):
        """
        Generate prompts for the BHC pipeline
        """
        bhc = self.dataset[self.dataset['BHC'].notna()]
        prompts = super().generate_prompts()
        return pd.merge(bhc, prompts, on='HADM_ID', how='left').drop(columns=['NOTE_ORDER'])
