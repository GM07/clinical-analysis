from typing import Any, Dict, List
from datasets import load_dataset

MEDHAL_FORMAT_WITH_CONTEXT = """### Context
{context}

### Statement
{statement}

### Factual
{label}

### Explanation
{explanation}
"""

MEDHAL_FORMAT_WITHOUT_CONTEXT = """### Statement
{statement}

### Factual
{label}

### Explanation
{explanation}
"""

class Formatter:

    def __call__(self, samples: List[Dict[str, Any]]) -> List[str]:

        output_texts = []
        for i in range(len(samples['statement'])):
            context = samples['context'][i]
            statement = samples['statement'][i]
            label = 'YES' if samples['label'][i] else 'NO'
            explanation = samples['explanation'][i]

            if context is not None and context != 'None' and context != '':
                output_texts.append(MEDHAL_FORMAT_WITH_CONTEXT.format(context=context, statement=statement, label=label, explanation=explanation))
            else:
                output_texts.append(MEDHAL_FORMAT_WITHOUT_CONTEXT.format(statement=statement, label=label, explanation=explanation))

        return output_texts
