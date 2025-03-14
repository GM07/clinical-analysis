from typing import Any, Dict, List
from datasets import load_dataset

TASK_DESCRIPTION = """### Task Description:
You will be given a statement and you're role is to determine whether the statement is factual or not. 
The statement can be based on a given context or not.
In the factual section, you must respond with "YES" if the statement is factual and "NO" if it is not.
Also generate an explanation for your answer stating why you think the statement is factual or not.

"""

MEDHAL_FORMAT_WITH_CONTEXT = TASK_DESCRIPTION + """### Context
{context}

### Statement
{statement}

### Factual
{label}

### Explanation
{explanation}
"""

MEDHAL_FORMAT_WITHOUT_CONTEXT = TASK_DESCRIPTION + """### Statement
{statement}

### Factual
{label}

### Explanation
{explanation}
"""

class Formatter:

    def __call__(self, x: List[Dict[str, Any]]) -> List[str]:
        if isinstance(x, Dict):
            return self.format_dict(x)

        return self.format_batched_dict(x)

    def format_dict(self, x: Dict[str, Any]) -> str:
        if x['context'] is not None and x['context'] != 'None' and x['context'] != '':
            return MEDHAL_FORMAT_WITH_CONTEXT.format(context=x['context'], statement=x['statement'], label=x['label'], explanation=x['explanation'])
        else:
            return MEDHAL_FORMAT_WITHOUT_CONTEXT.format(statement=x['statement'], label=x['label'], explanation=x['explanation'])

    def format_batched_dict(self, samples: List[Dict[str, Any]]) -> List[str]:

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
