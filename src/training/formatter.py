from typing import Any, Dict, List
from datasets import load_dataset

TASK_DESCRIPTION = """### Task Description
You will evaluate whether a medical statement is factually accurate.
The statement may reference a provided context.
Respond with "YES" if the statement is factually correct or "NO" if it contains inaccuracies.
Provide a brief explanation justifying your determination, citing specific evidence or reasoning.

"""

CONTEXT = """### Context
{context}

"""

MEDHAL_FORMAT_TRAINING_CONTEXT = TASK_DESCRIPTION + CONTEXT + """### Statement
{statement}

### Factual
{label}

### Explanation
{explanation}
"""

MEDHAL_FORMAT_TRAINING_NO_CONTEXT = TASK_DESCRIPTION + """### Statement
{statement}

### Factual
{label}

### Explanation
{explanation}
"""

MEDHAL_FORMAT_INFERENCE_CONTEXT = TASK_DESCRIPTION + CONTEXT + """### Statement
{statement}

### Factual
"""

MEDHAL_FORMAT_INFERENCE_NO_CONTEXT = TASK_DESCRIPTION + """### Statement
{statement}

### Factual
"""

class Formatter:

    def __init__(self, tokenizer, training=True):
        self.tokenizer = tokenizer
        self.training = training

    def __call__(self, x) -> List[str]:
        if isinstance(x['context'], str):
            return self.format_sample(x['context'], x['statement'], x['label'], x['explanation'])

        return self.format_batched_dict(x)

    def format_batched_dict(self, samples: List[Dict[str, Any]]) -> List[str]:

        output_texts = []
        for i in range(len(samples['statement'])):
            context = samples['context'][i]
            statement = samples['statement'][i]
            label = samples['label'][i]
            explanation = samples['explanation'][i]
            output_texts.append(self.format_sample(context, statement, label, explanation))

        return output_texts


    def format_sample(self, context, statement, label, explanation) -> str:

        yes_no_label = 'YES' if label else 'NO'
        
        if self.training:
            med_hal_format_context = MEDHAL_FORMAT_TRAINING_CONTEXT
            med_hal_format_no_context = MEDHAL_FORMAT_TRAINING_NO_CONTEXT
        else:
            med_hal_format_context = MEDHAL_FORMAT_INFERENCE_CONTEXT
            med_hal_format_no_context = MEDHAL_FORMAT_INFERENCE_NO_CONTEXT

        if context is not None and context != 'None' and context != '':
            return med_hal_format_context.format(context=context, statement=statement, label=yes_no_label, explanation=explanation) + self.tokenizer.eos_token
        else:
            return med_hal_format_no_context.format(statement=statement, label=yes_no_label, explanation=explanation) + self.tokenizer.eos_token
