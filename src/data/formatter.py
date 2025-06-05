from typing import Any, Dict, List

TASK_DESCRIPTION = """### Task Description
- You will evaluate whether a medical statement is factually accurate.
- The statement may reference a provided context.
- Respond with "YES" if the statement is factually correct or "NO" if it contains inaccuracies.
- In order to answer YES, everything in the statement must be supported by the context.
- In order to answer NO, there must be at least one piece of information in the statement that is not supported by the context.

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

    def __call__(self, x) -> Dict[str, str] | Dict[str, List[str]]:
        if isinstance(x['statement'], str):
            return {'text': self.format_sample(x['context'], x['statement'], x['label'] if self.training else None, x['explanation'] if self.training else None)}

        return {'text': self.format_batched(x)}

    def format_batched(self, samples: Dict[str, List[str]]) -> List[str]:
        if samples['statement'] is None:
            return []

        output_texts = []
        for i in range(len(samples['statement'])):
            context = samples['context'][i]
            statement = samples['statement'][i]
            label = None
            explanation = None

            if self.training:
                label = samples['label'][i]
                explanation = samples['explanation'][i] if samples['explanation'][i] else ''

            output_texts.append(self.format_sample(context, statement, label, explanation))

        return output_texts


    def format_sample(self, context, statement, label, explanation = None) -> str:
        yes_no_label = self.get_formatted_label(label)

        if self.training and not label:
            assert explanation is not None, f'When generating training samples, an explanation must be provided.\nStatement : {statement}\nLabel : {yes_no_label}'
        
        if self.training:
            med_hal_format_context = MEDHAL_FORMAT_TRAINING_CONTEXT
            med_hal_format_no_context = MEDHAL_FORMAT_TRAINING_NO_CONTEXT
        else:
            med_hal_format_context = MEDHAL_FORMAT_INFERENCE_CONTEXT
            med_hal_format_no_context = MEDHAL_FORMAT_INFERENCE_NO_CONTEXT

        if context is not None and context != 'None' and context != '':
            output = med_hal_format_context.format(context=context, statement=statement, label=yes_no_label, explanation=explanation)
        else:
            output = med_hal_format_no_context.format(statement=statement, label=yes_no_label, explanation=explanation)

        if self.training:
            output += self.tokenizer.eos_token

        return output

    def get_formatted_label(self, label):
        return 'YES' if label else 'NO'

    def format_one_shot_with_sample(
        self, 
        context_one_shot, 
        statement_one_shot, 
        label_one_shot, 
        explanation_one_shot, 
        context,
        statement) -> str:
        formatted_sample = MEDHAL_FORMAT_INFERENCE_CONTEXT.format(context=context, statement=statement)
        one_shot = self.format_one_shot(context_one_shot, statement_one_shot, label_one_shot, explanation_one_shot) + '\n'
        return one_shot + formatted_sample


    def format_one_shot(
        self, 
        context_one_shot, 
        statement_one_shot, 
        label_one_shot, 
        explanation_one_shot) -> str:
        return MEDHAL_FORMAT_TRAINING_CONTEXT.format(context=context_one_shot, statement=statement_one_shot, label=self.get_formatted_label(label_one_shot), explanation=explanation_one_shot)

