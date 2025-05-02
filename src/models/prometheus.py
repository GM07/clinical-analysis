

class Prometheus:

    SYSTEM_PROMPT = "You are a fair judge assistant tasked with providing clear, objective feedback based on specific criteria, ensuring each assessment reflects the absolute standards set for performance."

    PROMPT_TEMPLATE = """###Task Description:
An instruction (might include an Input inside it), a response to evaluate, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of two responses strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, choose a better response between Response A and Response B. You should refer to the score rubric.
3. The output format should look as follows: "(write a feedback for criteria) [RESULT] (A or B)"
4. Please do not generate any other opening, closing, and explanations.

###Instruction:
{instruction}

###Response A:
{response_A}

###Response B:
{response_B}

###Score Rubric:
{rubric}

###Feedback: """

    RUBRICS = {
        'factuality': """[Are the model's responses factually correct and well-supported by evidence?]
Score 1: The model's responses are mostly incorrect or based on unfounded information.
Score 2: The model sometimes provides factually correct responses, but inaccuracies are common.
Score 3: The model generally provides factually correct information, though some errors occur.
Score 4: The model often provides factually accurate information with only occasional minor errors.
Score 5: The model consistently provides responses that are factually correct and well-supported by evidence.""",
        'relevance': """[Are the model's responses relevant to the medical concept mentioned?]
Score 1: The model's answer is irrelevant to the medical concept and completely misses information that is related to the medical concept
Score 2: The model's short summary is mainly irrelevant, but mentions one or two things related to the medical concept mentioned
Score 3: The model's short summary is somewhat irrelevant, but contains key elements related to the concept mentioned
Score 4: The model's short summary is mainly relevant, but contains some elements that are not linked to the medical concept
Score 5: The model's short summary mentions everything related the the medical concept perfectly without missing any detail"""
    }

    @staticmethod
    def create_prompt(instruction: str, response_a: str, response_b: str, rubric_type: str = 'factuality') -> str:

        assert rubric_type in list(Prometheus.RUBRICS.keys()), f"Only these rubrics are supported {list(Prometheus.RUBRICS.keys())}"

        rubric = Prometheus.RUBRICS[rubric_type]

        return Prometheus.SYSTEM_PROMPT + '\n\n' + Prometheus.PROMPT_TEMPLATE.format(
            instruction=instruction,
            response_A=response_a,
            response_B=response_b,
            rubric=rubric
        )
