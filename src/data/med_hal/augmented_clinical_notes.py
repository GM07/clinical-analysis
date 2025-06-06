from collections import defaultdict
import json
import random
import os
import re
import time
from datasets import load_dataset, load_from_disk, Dataset
from tqdm import tqdm
from src.data.synthetic_dataset import SyntheticDataset
from src.utils import valid_json
import logging

logger = logging.getLogger(__name__)

ALL_CONCEPTS = {
    'Reason',
    'Date',
    'Duration',
    'Care center details',
    'Age',
    'Sex',
    'Ethnicity',
    'Weight',
    'Height',
    'Family medical history',
    'Recent travels',
    'Socio-economic context',
    'Occupation',
    'Physiological context',
    'Psychological context',
    'Vaccination history',
    'Allergies',
    'Exercise frequency',
    'Nutrition',
    'Sexual history',
    'Alcohol consumption',
    'Drug usage',
    'Smoking status',
    'Reason',
    'Type',
    'Time',
    'Outcome',
    'Details',
    'Name',
    'Intensity',
    'Location',
    'Time',
    'Temporalisation',
    'Behaviours affecting symptom',
    'Details',
    'Name',
    'Result',
    'Details',
    'Test',
    'Result',
    'Severity',
    'Condition',
    'Time',
    'Details',
    'Name',
    'Name of Symptom',
    'Related condition',
    'Dosage',
    'Time',
    'Frequency',
    'Duration',
    'Reason for taking',
    'Reaction',
    'Details',
    'Reason',
    'Referral',
    'Follow-up',
    'Summary',
}

ALL_CONCEPTS_CASE_INSENSITIVE = {concept.lower() for concept in ALL_CONCEPTS}

LIST_ELEMENTS = {'surgeries', 'symptoms', 'medical examinations', 'diagnosis tests', 'treatments'}

concept_one_shot_input = {
    'physiological context': 
        {'category': 'patient medical history',
         'value': 'History of left elbow arthrodesis performed for posttraumatic arthritis at the age of 18',
         'concept_reference': 'None'},
    'location': 
        {'category': 'symptoms',
         'value': 'Neck and lower back',
         'concept_reference': 'Neck and lower back'},
    'date': 
        {'category': 'admission',
         'value': 'One year after the initial surgery',
         'concept_reference': 'None'},
    'condition': 
        {'category': 'diagnosis tests',
         'value': 'Idiopathic osteonecrosis of the femoral head',
         'concept_reference': 'Magnetic resonance imaging (MRI) scan'},
    'exercise frequency': 
        {'category': 'patient medical history',
         'value': 'Regular, as the patient is a runner.',
         'concept_reference': 'None'},
    'result': 
        {'category': 'medical examinations',
         'value': 'Severe gait disturbance secondary to hip pain',
         'concept_reference': 'Physical examination'},
    'duration': 
        {'category': 'admission',
         'value': 'Three weeks',
         'concept_reference': 'None'},
    'time': 
        {'category': 'symptoms',
         'value': 'Past four months',
         'concept_reference': 'Neck and lower back'},
    'sex': 
        {'category': 'patient information',
         'value': 'Female',
         'concept_reference': 'None'},
    'temporalisation': 
        {'category': 'symptoms',
         'value': 'Increased over the following three weeks',
         'concept_reference': 'Left hip joint'},
    'ethnicity': 
        {'category': 'patient information',
         'value': 'Yemeni',
         'concept_reference': 'None'},
    'severity': 
        {'category': 'diagnosis tests',
         'value': 'Minimally displaced',
         'concept_reference': 'Radiographs'},
    'height': 
        {'category': 'patient information',
         'value': '144 cm',
         'concept_reference': 'None'},
    'psychological context': 
        {'category': 'patient medical history',
         'value': 'Diagnosed with bipolar affective disorder at the age of eleven, first episode was that of mania.',
         'concept_reference': 'None'},
    'referral': 
        {'category': 'discharge',
         'value': 'Referred to the Department of Cardiology due to a progressive worsening of central',
         'concept_reference': 'None'},
    'drug usage': 
        {'category': 'patient medical history',
         'value': 'Chewing pan of Indian tobacco for the last 15 years',
         'concept_reference': 'None'},
    'weight': 
        {'category': 'patient information',
         'value': '7 kg heavier than at the time of the first procedure',
         'concept_reference': 'None'},
    'vaccination history': 
        {'category': 'patient medical history',
         'value': 'No history of recent vaccination',
         'concept_reference': 'None'},
    'details': 
        {'category': 'symptoms',
         'value': 'Head turned to the right and upwards due to sustained contraction of neck muscles, sideways bending of the back in the lumbar region, limbs positioned to support body weight.',
         'concept_reference': 'Neck and lower back'},
    'outcome': 
        {'category': 'surgeries',
         'value': 'Discharged in good condition without specific complications',
         'concept_reference': 'Idiopathic osteonecrosis of the femoral head'},
    'smoking status': 
        {'category': 'patient medical history',
         'value': 'Quit 10 years ago, five pack-year history',
         'concept_reference': 'None'},
    'sexual history': 
        {'category': 'patient medical history',
         'value': 'Got married at the age of 15 and became pregnant soon after',
         'concept_reference': 'None'},
    'test': 
        {'category': 'diagnosis tests',
         'value': 'Magnetic resonance imaging (MRI) scan',
         'concept_reference': 'Magnetic resonance imaging (MRI) scan'},
    'reason for taking': 
        {'category': 'treatments',
         'value': 'Control of exacerbated mental illness',
         'concept_reference': 'Olanzapine tablets'},
    'recent travels': 
        {'category': 'patient information',
         'value': 'History of prolonged flights; however, none immediately prior to investigation',
         'concept_reference': 'None'},
    'age': 
        {'category': 'patient information',
         'value': 'Sixteen years old',
         'concept_reference': 'None'},
    'family medical history': 
        {'category': 'patient information',
         'value': 'No family history of a similar condition',
         'concept_reference': 'None'},
    'related condition': 
        {'category': 'treatments',
         'value': 'Bipolar affective disorder',
         'concept_reference': 'Olanzapine tablets'},
    'frequency': 
        {'category': 'treatments',
         'value': 'Daily',
         'concept_reference': 'Olanzapine tablets'},
    'allergies': 
        {'category': 'patient medical history',
         'value': 'No known allergies',
         'concept_reference': 'None'},
    'type': 
        {'category': 'surgeries',
         'value': 'Total Hip Arthroplasty (THA)',
         'concept_reference': 'Idiopathic osteonecrosis of the femoral head'},
    'alcohol consumption': 
        {'category': 'patient medical history',
         'value': 'Does not drink alcohol',
         'concept_reference': 'None'},
    'dosage': 
        {'category': 'treatments',
         'value': '5 mg per day',
         'concept_reference': 'Olanzapine tablets'},
    'name': 
        {'category': 'treatments',
         'value': 'Olanzapine tablets',
         'concept_reference': 'Olanzapine tablets'},
    'name of symptom':
        {'category': 'symptoms',
         'value': 'Pain in chest',
         'concept_reference': 'None'},
    'nutrition': 
        {'category': 'patient medical history',
         'value': 'Mostly consumed liquids and soft consistency meals during the time of dysphagia deterioration at the age of nineteen',
         'concept_reference': 'None'},
    'care center details': 
        {'category': 'admission',
         'value': 'Rheumatology clinic',
         'concept_reference': 'None'},
    'occupation': 
        {'category': 'patient information',
         'value': 'Allied health care',
         'concept_reference': 'None'},
    'reason': 
        {'category': 'admission',
         'value': 'Idiopathic osteonecrosis of the femoral head',
         'concept_reference': 'None'}
}

concept_one_shot_examples = {
    'physiological context': 'The patient underwent left elbow arthrodesis as a treatment for posttraumatic arthritis when they were 18 years old.',
    'location': 'The patient presented with symptoms in the neck and lower back region.',
    'date': 'The patient was admitted one year after their initial surgery.',
    'condition': 'An MRI scan revealed idiopathic osteonecrosis of the femoral head.',
    'exercise frequency': 'The patient maintains regular exercise as they are a runner.',
    'result': 'Physical examination showed severe gait disturbance secondary to hip pain.',
    'duration': "The patient's admission lasted for three weeks.",
    'time': "The neck and lower back symptoms have been present for the past four months.",
    'sex': "The patient is female.",
    'temporalisation': "The left hip joint symptoms increased in severity over the following three weeks.",
    'ethnicity': 'The patient is of Yemeni ethnicity.',
    'severity': "Radiographs showed minimal displacement.",
    'height': 'The patient is 144 cm tall.',
    'psychological context': 'The patient was diagnosed with bipolar affective disorder at age eleven, with their first episode being manic in nature.',
    'referral': 'Upon discharge, the patient was referred to the Department of Cardiology due to progressive worsening of central.',
    'drug usage': 'The patient has been chewing pan of Indian tobacco for the past 15 years.',
    'weight': 'The patient is 7 kg heavier than at the time of their first procedure.',
    'vaccination history': 'The patient has no history of recent vaccination.',
    'details': 'The patient presents with head turned right and upwards due to sustained neck muscle contraction, sideways bending of the back in the lumbar region, and limbs positioned to support body weight.',
    'outcome': 'Following surgery for idiopathic osteonecrosis of the femoral head, the patient was discharged in good condition without specific complications.',
    'smoking status': 'The patient quit smoking 10 years ago after a five pack-year history.',
    'sexual history': 'The patient married at age 15 and became pregnant shortly afterward.',
    'test': 'The patient underwent a magnetic resonance imaging (MRI) scan for diagnostic purposes.',
    'reason for taking': 'The patient is taking Olanzapine tablets for the control of exacerbated mental illness.',
    'recent travels': 'The patient has a history of prolonged flights but none immediately before the investigation.',
    'age': 'The patient is sixteen years old.',
    'family medical history': 'The patient has no family history of a similar condition.',
    'related condition': 'The patient is being treated with Olanzapine tablets for bipolar affective disorder.',
    'frequency': 'The patient takes Olanzapine tablets daily.',
    'allergies': 'The patient has no known allergies.',
    'type': 'The patient underwent Total Hip Arthroplasty (THA) for idiopathic osteonecrosis of the femoral head.',
    'alcohol consumption': 'The patient does not drink alcohol.',
    'dosage': 'The patient takes 5 mg of Olanzapine tablets per day.',
    'name': 'The patient is prescribed Olanzapine tablets as treatment.',
    'nutrition': ' The patient primarily consumed liquids and soft consistency meals during their period of dysphagia deterioration at age nineteen.',
    'care center details': 'The patient was admitted to the Rheumatology clinic.',
    'occupation': 'The patient works in allied health care.',
    'reason': 'The patient was admitted due to idiopathic osteonecrosis of the femoral head.',
    'name of symptom': 'The patient reported chest pain.',
}

system_prompt = """You are tasked with transforming structured medical data into natural language statements about a patient. Each input will contain 4 elements:
- concept: The type of information being described (e.g., dosage, age, symptoms)
- value: The specific information or measurement
- category: The broad medical category this information belongs to (e.g., treatment, patient information, symptoms)
- concept_reference: The specific element that the value refers to (e.g., a specific medication, a specific symptom)

Your task is to generate a clear, grammatically correct sentence that conveys this information in a medical context. Follow these rules:
1. Use appropriate verbs based on the concept:
- For treatments: 'takes', 'receives', 'is prescribed'
- For symptoms: 'experiences', 'reports', 'presents with'
- For measurements/states: 'is', 'has', 'shows'
- For time-related concepts: 'has been', 'started', 'continues'
2. Incorporate the concept_reference when it adds clarity
3. Use present tense
4. Maintain medical terminology as provided
5. When the concept_reference is 'None' or does not add clarity, don't include it in the statement
6. The statement should be a single sentence.

Do not include any other information in the statement aside from the concept and the extraction. Only output the statement and nothing else."""

concept_descriptions = """
Here are possible concepts that can be mentioned as well as a description of them:

Visit motivation : Reason for the patient's visit

**Category : Admissions**
Reason : Reason for admission to a care center
Date : Date of first admission
Duration : Length of patient's stay
Care center details : Any details of care center the patient was admitted to

**Category : Patient information**
Age : Patient's age
Sex : Patient's sex
Ethnicity : Patient's ethnicity or nationality
Weight : Patient's weight
Height : Patient's height
Family medical history : Information about family medical history
Recent travels : Details about patient's recent travels
Socio-economic context : Patient's socio-economic background
Occupation : Patient's occupation

**Category : Patient medical history**
Physiological context : Relevant physiological history of the patient
Psychological context : Relevant psychological history of the patient
Vaccination history : History of vaccinations received by the patient
Allergies : Any known allergies of the patient
Exercise frequency : Frequency of patient's exercise activity history
Nutrition : Information about patient's nutrition
Sexual history : Relevant details about patient's sexual history
Alcohol consumption : Patient's alcohol consumption habits
Drug usage : Information about any recreative drugs used by patient
Smoking status : Patient's smoking status

**Category : Surgeries**
Reason : Reason for surgery
Type : Type of surgery performed
Time : Time of surgery
Outcome : Outcome of surgery
Details : Details about the surgery

**Category : Symptoms**
Name : Specific symptom experienced by the patient
Intensity : Severity or intensity of the symptom 
Location : Where the symptom is localized
Time : Any temporal details about when the symptom appears
Temporalisation : Any specific timing patterns associated with the symptom
Behaviours affecting symptom : Activities or actions that influence the severity or occurrence of symptom
Details : All additional details about the symptom

**Category : Medical examinations**
Name : Name of medical examination performed
Result : Result or measurement of the physical examination
Details : All additional details about the physical examination

**Category : Diagnosis tests**
Test : Name of medical test performed to diagnose the condition
Result : Result or measurement of the medical test performed
Severity : Severity level of diagnosed condition
Condition : Name of medical conditions diagnosed
Time : Any temporal details about when the diagnosis test was performed
Details : All additional details about the medical test

**Category : Treatments**
Name : Name of treatment or medication prescribed to the patient
Related condition : Medical condition that the treatment is prescribed for
Dosage : Amount or strength of the treatment
Time : Any temporal details about when the treatment was performed
Frequency : How often the treatment is taken
Duration : The length of time the patient should take the treatment
Reason for taking : The medical reason for taking the treatment
Reaction : Patient's reaction or response to the prescribed treatment
Details : All additional details about the treatment

**Category : Discharge**
Reason : Reason motivating patient's discharge
Referral : Details about any referrals
Follow-up : Details about any follow up appointments
Summary : Summary of patient's discharge

"""

class AugmentedClinicalNotes(SyntheticDataset):
    """
    Class to load and prepare the AugmentedClinicalNotes dataset.

    Important definitions:
    - Concept : A medical concept that can be extracted from a note (e.g. "symptoms", "treatments", "diagnosis tests", etc.)
    - Value : The value associated with a concept (e.g. "Headache", "Olanzapine tablets", "Magnetic resonance imaging (MRI) scan", etc.)
    - Concept reference : The specific element that the value refers to (e.g. a specific medication, a specific symptom)

    For example, if the note is "The patient has a fever and a cough" and the extraction is :
    {
        "symptoms": [
            {"name": "Fever", "severity": "Mild"},
            {"name": "Cough", "severity": "Mild"}
        ]
    }
    For a given extraction, the concept could be "severity", the value could be "Mild" and the concept reference could be "Fever"
    """

    def __init__(self, dataset_path: str):

        self.dataset_path = dataset_path
        self.values_per_concepts = {}
        self.one_shot_chats_per_concept = {}

        self.load()

    def load(self):
        """
        Load the dataset from the given path
        """
        self.dataset_name = os.path.basename(self.dataset_path)
        self.dataset_folder_path = self.dataset_path.replace(self.dataset_name, '')

        if self.dataset_path.endswith('.jsonl'):
            self.data = Dataset.from_json(self.dataset_path)
        else:
            self.data = load_from_disk(
                self.dataset_path,
                # trust_remote_code=True,
            )['train']

        logger.info(f"Loaded {len(self.data)} samples from {self.dataset_path} with columns : {self.data.column_names}")

    def build_caches(self):
        """
        Build the caches for the dataset. This consists of building a mapping of concepts to their values and a mapping of concepts to their one-shot chats.
        """

        if self.values_per_concepts and self.one_shot_chats_per_concept:
            return

        self.values_per_concepts = {}
        self.one_shot_chats_per_concept = {}

        for row in self.data:
            concept, value = row['concept'], row['value']
            if concept not in self.values_per_concepts:
                self.values_per_concepts[concept] = []
                self.one_shot_chats_per_concept[concept] = self._create_one_shot_chat(concept)
            self.values_per_concepts[concept].append(value)


    def filter_extraction_summaries(self):
        """
        Filter the dataset to remove invalid extraction summaries. Invalid summaries are samples where the summary is not valid JSON.
        """
        self.data = self.data.filter(lambda x: valid_json(x['summary']), desc="Filtering invalid extraction summaries")
        return self.data

    def extract_information_from_summaries(self):
        """
        Extract the information from the summaries of the dataset. For every row, this function will extract the information from 
        the summary and return the row with the extractions. Each extraction is a string detailing the path to the value in the 
        json tree separated by <sep>.

        For example, if the summary is:
        {
            "symptoms": [
                {"name": "Headache", "severity": "Mild"}
            ]
        }
        The extraction will be:
        "symptoms/0/name<sep>Headache"

        
        """
        self.data = self.data.map(self._extract_information_from_row, desc="Extracting information from summaries")
        return self.data

    def scatter_extractions(self):
        """
        Scatter the extractions of the dataset into a list of processed extractions.
        """
        self.data = self.data.map(
            self._scatter_extractions_from_row,
            batched=True,
            batch_size=1,
            remove_columns=self.data.column_names, # Remove original columns
            desc="Scattering extractions"
        ).flatten()
        return self.data

    def generate_positive_samples(self):
        """
        Generate positive samples from the dataset.
        """
        self.data = self.data.map(self._generate_positive_sample_from_row, desc="Generating positive samples")
        return self.data

    def generate_negative_samples(self):
        """
        Generate negative samples from the dataset.
        """

        self.data = self.data.map(
            self._generate_negative_samples_from_row, 
            batched=True, 
            batch_size=100, 
            desc="Generating negative samples"
            # with_indices=True
        )

        return self.data

    def generate_prompts(self, output_path: str, max_rows: int = None):
        """
        Prepares the dataset by filtering invalid extraction summaries, extracting information from summaries, and scattering extractions.

        After this step, a row containing a summary in the form of a json will be scatted into multiple rows, each containing an extraction from the json.

        Args:
            output_path (str): The path to save the prompts to
            max_rows (int): The maximum number of rows to save in the output file
        """
        initial_len = len(self.data)
        self.filter_extraction_summaries()
        logger.info(f"Removed invalid extraction summaries : {initial_len} -> {len(self.data)}")

        initial_len = len(self.data)
        self.extract_information_from_summaries()
        logger.info(f"Extracted information from summaries : {initial_len} -> {len(self.data)}")

        initial_len = len(self.data)
        self.scatter_extractions()
        logger.info(f"Scattered extractions : {initial_len} -> {len(self.data)}")

        self.build_caches()

        initial_len = len(self.data)
        self.generate_positive_samples()
        logger.info(f"Generated positive samples : {initial_len} -> {len(self.data)}")
    
        initial_len = len(self.data)
        self.generate_negative_samples()
        logger.info(f"Generated negative samples : {initial_len} -> {len(self.data)}")

        # self.data = self.data.shuffle(seed=42) # Make sure all extractions are not linked to the same notes
        
        if max_rows is not None:
            self.data = self.data.select(range(max_rows))

        self.save(output_path)

    def save(self, path: str):
        """
        Save the dataset to a csv file
        """
        self.data.to_csv(path)


    @staticmethod
    def fix_sex_generations(factual_statement: str, hallucinated_statement: str):
        """
        Fix the sex generations

        Args:
            factual_statement (str): The factual statement
            hallucinated_statement (str): The hallucinated statement

        Returns:
            tuple: A tuple containing the fixed factual statement and the hallucinated statement
        """

        patient_male = len(re.findall(r'\s(?:male|man|boy)', factual_statement, re.IGNORECASE)) > 0
        hallucinated_male = len(re.findall(r'\s(?:male|man|boy)', hallucinated_statement, re.IGNORECASE)) > 0

        if patient_male and not hallucinated_male:
            return factual_statement, hallucinated_statement
        elif patient_male and hallucinated_male:
            return factual_statement, re.sub(r'\s(?:male|man|boy)', ' female', hallucinated_statement)
        elif not patient_male and not hallucinated_male:
            return factual_statement, re.sub(r'\s(?:female|woman|girl)', ' male', hallucinated_statement)
        else:
            return factual_statement, hallucinated_statement


    def _extract_information_from_row(self, row):
        """
        Given a row, extract the information from the summary and return the row with the extractions

        Each extraction is a string detailing the path to the value in the json tree separated by <sep>

        For example, if the summary is:
        {
            "symptoms": [
                {"name": "Headache", "severity": "Mild"}
            ]
        }
        The extraction will be:
        "symptoms/0/name<sep>Headache"

        Returns the row with the extractions
        """
        json_summary = json.loads(row['summary'])
        
        def flatten_dict(d, parent_key='', sep='/'):
            """
            Flatten a dictionary into a list of strings, each representing a path to a value in the dictionary
            """
            items = []
            
            def _flatten(obj, prefix=''):
                """
                Recursively flatten a dictionary into a list of strings, each representing a path to a value in the dictionary
                """
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        new_key = f"{prefix}{sep}{k}" if prefix else k
                        _flatten(v, new_key)
                elif isinstance(obj, list):
                    # Handle multiple items in list
                    for i, item in enumerate(obj):
                        if isinstance(item, str):
                            items.append(f'{prefix}<sep>{str(item)}')
                        else:
                            new_key = f"{prefix}{sep}{i}" if prefix else str(i)
                            _flatten(item, new_key)
                else:
                    # Only add if not None
                    items.append(f'{prefix}<sep>{str(obj)}')
            
            _flatten(d, parent_key)
            return items
        
        flattened = flatten_dict(json_summary)
        row['extractions'] = flattened
        return row

    def _scatter_extractions_from_row(self, row):
        """
        Scatter the extractions of a row into a list of processed extractions.

        Each processed extraction is a dictionary with the following keys:
        - category: The category of the extraction
        - concept: The concept of the extraction
        - value: The value of the extraction
        - concept_reference: The concept reference of the extraction
        - full_note: The full note of the extraction
        - key_path: The path to the value in the json tree
        - idx: The index of the row

        """

        # IMPORTANT : We consider that an extraction is valid even if it does not capture everything that is said in the note
        # For example, if the note is "The patient has a fever and a cough", the extraction "fever" is valid even if it does not capture the cough

        idx = row['idx'][0]
        full_note = row['full_note'][0]
        extractions = row['extractions'][0]

        processed_extractions = []
        last_category, feature_reference_name, feature_reference_index = None, None, None
        processed_extractions = {
            'category': [],
            'concept': [],
            'value': [],
            'concept_reference': [],
            'full_note': [],
            'key_path': [],
            'idx': []
        }
        
        for extraction in extractions:
            key, value = extraction.split('<sep>')

            key_path = key.split('/')
            category, feature = key_path[0].lower(), key_path[-1].lower()

            if feature.lower() not in ALL_CONCEPTS_CASE_INSENSITIVE or value == 'None':
                continue

            if category != last_category:
                # If we change category, we reset the feature reference
                feature_reference_index = None
                feature_reference_name = None

            if category in LIST_ELEMENTS and value != 'None':
                feature_index = key_path[1]
                
                if feature_reference_index != feature_index:
                    # If category has changed, the feature reference is None and it will
                    # not equal the feature index. If the category has not changed, we
                    # update the feature reference if the feature index has changed.
                    feature_reference_index = feature_index
                    feature_reference_name = value

                processed_extractions['concept_reference'].append(feature_reference_name)
            else:
                feature_reference_index = None
                feature_reference_name = None
                processed_extractions['concept_reference'].append('None')

            processed_extractions['category'].append(category)
            processed_extractions['concept'].append(feature)
            processed_extractions['value'].append(value)
            processed_extractions['full_note'].append(full_note)
            processed_extractions['key_path'].append(key_path)
            processed_extractions['idx'].append(idx)

            if category != last_category:
                last_category = category

        return processed_extractions

    def _generate_positive_sample_from_row(self, row):
        category, concept, value, concept_reference = row['category'], row['concept'], row['value'], row['concept_reference']

        prompt = f"concept : {concept}\ncategory : {category}\nvalue : {value}\nconcept_reference : {concept_reference}"
        chat = self.one_shot_chats_per_concept[concept] + [{'role': 'user', 'content': prompt}]

        return row | {
            'chat': chat, 
            'factual': True,
            'system_prompt': chat[0]['content'],
            'one_shot_user_input': chat[1]['content'],
            'one_shot_assistant_output': chat[2]['content'],
            'user_input': prompt,
        }

    def _generate_negative_samples_from_row(self, row):
        """
        For every row, generates a false positive that contains the same concept as the row but with a different value

        This function should always be called with batched=True and batch_size=1
        """

        final_dicts = defaultdict(list)

        # Positive sample
        for k, v in row.items():
            for val in v:
                final_dicts[k].append(val)
        for i in range(len(row['category'])):
            final_dicts['corrupted_value'].append(None)

        other_keys = set(row.keys())
        for key in ['chat', 'factual', 'corrupted_value', 'system_prompt', 'one_shot_user_input', 'one_shot_assistant_output', 'user_input']:
            other_keys.discard(key)

        # Negative sample
        for i in range(len(row['category'])):

            category, concept, value, concept_reference = row['category'][i], row['concept'][i], row['value'][i], row['concept_reference'][i]
            sampled_value = self._sample_random_value_from_concept(concept, value)

            prompt = f"concept : {concept}\ncategory : {category}\nvalue : {sampled_value}\nconcept_reference : {concept_reference}"
            chat = self.one_shot_chats_per_concept[concept] + [{'role': 'user', 'content': prompt}]
            final_dicts['chat'].append(chat)
            final_dicts['factual'].append(False)
            final_dicts['corrupted_value'].append(sampled_value)
            final_dicts['system_prompt'].append(chat[0]['content'])
            final_dicts['one_shot_user_input'].append(chat[1]['content'])
            final_dicts['one_shot_assistant_output'].append(chat[2]['content'])
            final_dicts['user_input'].append(prompt)

            for k in other_keys:
                final_dicts[k].append(row[k][i])

            # positive_sample = row | {'corrupted_value': [None]}
            # negative_sample = row | {
            #     'chat': [chat], 
            #     'factual': [False], 
            #     'corrupted_value': [sampled_value],
            #     'system_prompt': [chat[0]['content']],
            #     'one_shot_user_input': [chat[1]['content']],
            #     'one_shot_assistant_output': [chat[2]['content']],
            #     'user_input': [prompt],
            # }
            
            # final_dicts.append(positive_sample)
            # final_dicts.append(negative_sample)
        
        # result = self._list_of_dicts_to_dict(final_dicts)
        # for k, v in final_dicts.items():
            # print(f'{k} : {len(v)}')

        return final_dicts

    def _sample_random_value_from_concept(self, concept, value):
        """
        Sample a random value associated with a concept

        Args:
            concept (str): The concept to sample a value from
            value (str): The value to exclude from the sampling

        Returns:
            str: A random value associated with the concept or None if the 
            concept has only one value and it is the same as the value to exclude

        """

        if concept not in self.values_per_concepts:
            return None
        
        values = self.values_per_concepts[concept]

        # If the concept has only one value and it is the same as the value to exclude, we return None
        if len(values) == 1 and values[0] == value:
            return None

        sampled_value = random.choice(values)
        # while sampled_value == value:
            # sampled_value = random.choice(values)
        return sampled_value

    def _list_of_dicts_to_dict(self, list_of_dicts):
        """
        Transforms a list of dictionaries into a dictionary of lists by concatenating the values of each key.

        Args:
            list_of_dicts (list): List of dictionaries to transform

        """
        result = {}
        for d in list_of_dicts:
            for key, value in d.items():
                if key not in result:
                    result[key] = []
                result[key].append(value[0])
        return result

    def _create_one_shot_chat(self, concept):
        """
        Creates a one-shot chat for a given concept including the system prompt and the one-shot input and output.

        Args:
            concept (str): The concept to create a one-shot chat for

        Returns:
            list: List of dictionaries representing the one-shot chat
        """
        one_shot = concept_one_shot_input[concept.lower()] # Input dictionary
        one_shot_output = concept_one_shot_examples[concept.lower()] # Expected output

        system_prompt_chat = [
            {'role': 'system', 'content': system_prompt},
        ]

        one_shot_chat = [
            {'role': 'user', 'content': f"concept : {concept}\ncategory : {one_shot['category']}\nvalue : {one_shot['value']}\nconcept_reference : {one_shot['concept_reference']}"},
            {'role': 'assistant', 'content': one_shot_output}
        ]

        return system_prompt_chat + one_shot_chat

    def __getitem__(self, idx):
        return self.data[idx]
    
    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)


class AugmentedClinicalNotesValidator:

    PROMPT_TEMPLATE = """You must evaluate whether an extraction a clinical note is factual or not. If it is factual, answer with YES. If it is not factual, answer with NO. \nHere is the clinical note : {clinical_note}\nHere is the extraction : {extraction}\n\nOnly answer with YES or NO. Do not generate any other explanation."""

    def __init__(self, processed_path: str):
        self.processed_path = processed_path

        self.load()

    def load(self):

        if self.processed_path.endswith('.csv'):
            self.dataset = Dataset.from_csv(self.processed_path)
        else:
            self.dataset = load_from_disk(self.processed_path)

    def generate_validation_prompts(self):


        def generate(x):
            return {'prompt': [self.PROMPT_TEMPLATE.format(clinical_note=note, extraction=out) for note, out in zip(x['full_note'], x['output'])]}

        columns = ['chat', 'system_prompt', 'one_shot_user_input', 'one_shot_assistant_output', 'user_input']
        return self.dataset.map(generate, remove_columns=columns, batched=True)
