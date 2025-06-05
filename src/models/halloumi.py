from nltk.tokenize import sent_tokenize

import contextlib
from dataclasses import dataclass, field



@dataclass
class Claim:
    claim_id: int = -1
    claim_string: str = ""
    subclaims: list[str] = field(default_factory=list)
    citations: list[int] = field(default_factory=list)
    rationale: str = ""
    supported: bool = True

class HallOumi:
    PROMPT_TEMPLATE = "<context>\n{context}\n</context>\n\n<claims>\n{claim}\n</claims>"

    @staticmethod
    def create_prompt_classifier(context: str, claim: str) -> str:
        return HallOumi.PROMPT_TEMPLATE.format(
            context=context,
            claim=claim
        )

    @staticmethod
    def create_prompt_generator(context: str, request: str, answer: str) -> str:
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
        annotated_request = f"<|request|><{request.strip()}><end||request>"

        # Response: Split it into sentences and annotate them.
        response_sentences = _split_into_sentences(answer)
        annotated_response_sentences = _annotate_sentences(response_sentences, "r")
        annotated_response = f"<|response|>{annotated_response_sentences}<end||response>"

        # Combine all parts into the final prompt.
        return f"{annotated_context}{annotated_request}{annotated_response}"

    def get_claims_from_response(response: str) -> list[Claim]:
        """Extracts claims from the response string."""

        def _get_claim_id_from_subsegment(subsegment: str) -> int:
            claim_id_part = subsegment.split("|")[1]
            claim_id_no_r = claim_id_part.lstrip("r")
            return int(claim_id_no_r)

        def _get_claim_citations_from_subsegment(subsegment: str) -> list[int]:
            citation_segments = subsegment.split(",")
            citations = []
            for citation_segment in citation_segments:
                citation = citation_segment.replace("|", "").replace("s", "").strip()
                if "-" in citation:
                    start, end = map(int, citation.split("-"))
                    citations.extend(range(start, end + 1))
                elif "to" in citation:
                    start, end = map(int, citation.split("to"))
                    citations.extend(range(start, end + 1))
                else:
                    with contextlib.suppress(ValueError):
                        citation_int = int(citation)
                        citations.append(citation_int)
            return citations

        def _get_claim_from_segment(segment: str) -> Claim:
            claim_segments = segment.split("><")
            claim = Claim()
            claim.claim_id = _get_claim_id_from_subsegment(claim_segments[0])
            claim.claim_string = claim_segments[1]

            subclaims = []
            claim_progress_index = 3  # start parsing subclaims from index 3
            for i in range(claim_progress_index, len(claim_segments)):
                subsegment = claim_segments[i]
                if subsegment.startswith("end||subclaims"):
                    claim_progress_index = i + 1
                    break
                subclaims.append(subsegment)

            citation_index = -1
            rationale_index = -1
            label_index = -1

            for i in range(claim_progress_index, len(claim_segments)):
                subsegment = claim_segments[i]
                if subsegment.startswith("|cite|"):
                    citation_index = i + 1
                elif subsegment.startswith("|explain|"):
                    rationale_index = i + 1
                elif subsegment.startswith("|supported|") or subsegment.startswith(
                    "|unsupported|"
                ):
                    label_index = i

            claim.subclaims = subclaims
            claim.citations = (
                _get_claim_citations_from_subsegment(claim_segments[citation_index])
                if citation_index != -1
                else []
            )
            claim.rationale = (
                claim_segments[rationale_index] if rationale_index != -1 else ""
            )
            claim.supported = (
                claim_segments[label_index].startswith("|supported|")
                if label_index != -1
                else True
            )
            return claim

        segments = response.split("<end||r>")
        claims = []
        for segment in segments:
            if segment.strip():
                claim = _get_claim_from_segment(segment)
                claims.append(claim)
        return claims
