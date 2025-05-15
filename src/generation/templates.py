BASE_PROMPT_TEMPLATE="""Here is a clinical note about a patient: 
-------------------
{clinical_note}
-------------------
Extract the information that is related to the "{label}" medical concept from the clinical note. 
{properties}
If the concept is not mentioned in the note, respond with 'N/A'. Only output the extracted information.
"""

BHC_BASE_TEMPLATE="""
Here are multiple clinical notes associated to the hospital course of a patient ordered by the time they were recorded: 
-------------------
{clinical_notes}
-------------------
Summarize the hospital course of the patient based on the clinical notes. Only output the summary.
"""

DOMAIN_ADAPTATION_TEMPLATE="""
Here are multiple clinical notes associated to the hospital course of a patient ordered by the time they were recorded: 

{clinical_notes}

Summarize the hospital course of the patient only using the information related to the "{domain}" medical domain in a text. Only output the summary without any additional text.
"""

PRUNED_CONCEPT_TEMPLATE="""
Sentences were extracted from multiple clinical notes based on a medical concepts. For each clinical note, we have a dictionary where the keys are the concepts and the values are the sentences that were extracted linked to those concepts. The clinical notes are ordered by the time they were recorded. Here are the clinical notes' extractions:

{clinical_notes}

Summarize the clinical notes of the patient based on the extractions of each clinical note in a text. Only output the summary without any additional text.
"""


LLAMA_BIO_SYSTEM_PROMPT="""You are an expert and experienced from the healthcare and biomedical domain with extensive medical knowledge and practical experience. Your name is OpenBioLLM, and you were developed by Saama AI Labs. who's willing to help answer the user's query with explanation. In your explanation, leverage your deep medical expertise such as relevant anatomical structures, physiological processes, diagnostic criteria, treatment guidelines, or other pertinent medical concepts. Use precise medical terminology while still aiming to make the explanation clear and accessible to a general audience."""
DEFAULT_SYSTEM_ENTRY = "You are an expert and experienced from the healthcare and biomedical domain with extensive medical knowledge and practical experience who's willing to help answer the user's query with explanation. In your explanation, leverage your deep medical expertise such as relevant anatomical structures, physiological processes, diagnostic criteria, treatment guidelines, or other pertinent medical concepts. Use precise medical terminology while still aiming to make the explanation clear and accessible to a general audience."
