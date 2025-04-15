from nltk.tokenize import sent_tokenize


class HallOumi:
    PROMPT_TEMPLATE = "<context>\n{context}\n</context>\n\n<claims>\n{claim}\n</claims>"

    @staticmethod
    def create_prompt_classifier(context: str, claim: str) -> str:
        return HallOumi.PROMPT_TEMPLATE.format(
            context=context,
            claim=claim
        )

    @staticmethod
    def create_prompt_generator(context: str, claim: str, answer: str) -> str:
        """Generates a prompt for the HallOumi model."""

        # Taken from https://github.com/oumi-ai/oumi/tree/main/configs/projects/halloumi
        def _split_into_sentences(text: str) -> list[str]:
            sentences = sent_tokenize(text.strip())
            return [s.strip() for s in sentences if s.strip()]

        def _annotate_sentences(sentences: list[str], annotation_char: str) -> str:
            annotated_sentences = []
            for idx, sentence in enumerate(sentences, start=1):
                annotated_sentences.append(
                    f"<|{annotation_char}{idx}|><{sentence}><end||{annotation_char}>"
                )
            return "".join(annotated_sentences)

        # Context: Split it into sentences and annotate them.
        context_sentences = _split_into_sentences(context)
        annotated_context_sentences = _annotate_sentences(context_sentences, "s")
        annotated_context = f"<|context|>{annotated_context_sentences}<end||context>"

        # Request: Annotate the request.
        annotated_request = f"<|request|><{claim.strip()}><end||request>"

        # Response: Split it into sentences and annotate them.
        response_sentences = _split_into_sentences(answer)
        annotated_response_sentences = _annotate_sentences(response_sentences, "r")
        annotated_response = f"<|response|>{annotated_response_sentences}<end||response>"

        # Combine all parts into the final prompt.
        return f"{annotated_context}{annotated_request}{annotated_response}"
