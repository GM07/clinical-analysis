import re
from typing import Optional, Tuple

class MedHalParser:


    LABEL_PATTERN = r'Factual(?::)?(?:\n| |\*)*(YES|NO)'
    EXPLANATION_PATTERN = r'### Explanation([\s\S]+)'

    def __call__(self, x: str) -> Optional[Tuple[bool, str]]:

        matches = re.findall(self.LABEL_PATTERN, x, re.IGNORECASE)
        if len(matches) == 0:
            return None
        label = True if matches[0].lower() == 'yes' else False

        matches = re.findall(self.EXPLANATION_PATTERN, x, re.IGNORECASE)
        if len(matches) == 0:
            return None
        explanation = matches[0]

        return (label, explanation)        
