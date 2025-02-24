BASE_PROMPT_TEMPLATE="""Here is a clinical note about a patient: 
-------------------
{clinical_note}
-------------------
In a short sentence, extract the information that is related to the "{label}" medical concept from the clinical note. If the concept is not mentioned in the note, respond with 'N/A'. Only output the extracted information.
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
-------------------
{clinical_notes}
-------------------
Summarize the hospital course of the patient only using the information related to the "{domain}" medical domain. Only output the summary.
"""

PRUNED_CONCEPT_TEMPLATE="""
Sentences were extracted from multiple clinical notes based on a medical concepts. For each clinical note, we have a dictionary where the keys are the concepts and the values are the sentences that were extracted linked to those concepts. The clinical notes are ordered by the time they were recorded. Here are the clinical notes' extractions:
-------------------
{clinical_notes}
-------------------
Summarize the hospital course of the patient based on the extractions of each clinical note. Only output the summary.
"""
