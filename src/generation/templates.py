BASE_PROMPT_TEMPLATE="""Here is a clinical note about a patient : 
-------------------
{clinical_note}
-------------------
In a short sentence, extract the information that is related to the "{label}" medical concept from the clinical note. If the concept is not mentioned in the note, respond with 'N/A'. Only output the extracted information.
"""
