import re


_CITATION_PATTERN = re.compile(r"\[(\d+)\]")


def validate_citations(
    answer: str,
    citation_count: int,
) -> dict:
    """
    Validate that citations in the answer reference
    actual retrieved evidence.
    """

    matches = _CITATION_PATTERN.findall(answer)

    cited_numbers = sorted(
        {int(number) for number in matches}
    )

    valid_citations = [
        number
        for number in cited_numbers
        if 1 <= number <= citation_count
    ]

    invalid_citations = [
        number
        for number in cited_numbers
        if number < 1 or number > citation_count
    ]

    return {
        "is_valid": (
            bool(valid_citations)
            and not invalid_citations
        ),
        "cited_numbers": valid_citations,
        "invalid_citations": invalid_citations,
    }
