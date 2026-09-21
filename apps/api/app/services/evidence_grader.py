from langchain_openai import ChatOpenAI

from app.core.config import settings


grader_llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=settings.openai_api_key,
    temperature=0,
)


def build_evidence(results: list[dict]) -> str:
    parts = []

    for index, result in enumerate(results, start=1):
        page = result.get("page_number")

        if page is None:
            source = f"Evidence [{index}]"
        else:
            source = f"Evidence [{index}], page {page}"

        parts.append(
            f"{source}\n{result['content']}"
        )

    return "\n\n".join(parts)


async def grade_evidence(
    query: str,
    results: list[dict],
) -> dict:
    if not results:
        return {
            "is_sufficient": False,
            "score": 0.0,
            "reason": "No evidence was retrieved.",
        }

    evidence = build_evidence(results)

    prompt = f"""
You are an evidence-quality grader for an enterprise RAG system.

Determine whether the provided evidence contains enough information
to answer the user's question accurately.

Rules:
- Judge ONLY the provided evidence.
- Do not use outside knowledge.
- The evidence does not need to contain the exact wording of the question.
- It must contain enough factual information to construct a reliable answer.
- If the evidence is unrelated, incomplete, or ambiguous, mark it insufficient.
- Return ONLY valid JSON.

Return exactly:
{{
  "is_sufficient": true or false,
  "score": number between 0 and 1,
  "reason": "brief explanation"
}}

User question:
{query}

Evidence:
{evidence}
"""

    response = await grader_llm.ainvoke(prompt)

    content = response.content

    if isinstance(content, list):
        content = "".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict)
        )

    content = str(content).strip()

    # Remove markdown JSON fences if the model adds them.
    if content.startswith("```"):
        content = content.replace("```json", "")
        content = content.replace("```", "")
        content = content.strip()

    import json

    try:
        result = json.loads(content)

        return {
            "is_sufficient": bool(
                result.get("is_sufficient", False)
            ),
            "score": max(
                0.0,
                min(
                    1.0,
                    float(result.get("score", 0.0)),
                ),
            ),
            "reason": str(
                result.get(
                    "reason",
                    "No reason provided.",
                )
            ),
        }

    except (json.JSONDecodeError, ValueError, TypeError):
        return {
            "is_sufficient": False,
            "score": 0.0,
            "reason": "Evidence grader returned an invalid result.",
        }
