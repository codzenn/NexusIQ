from langchain_openai import ChatOpenAI

from app.core.config import settings
from app.retrieval.hybrid import hybrid_search
from app.services.citation_validator import validate_citations
from app.services.evidence_grader import grade_evidence


llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=settings.openai_api_key,
    temperature=0,
)


def build_context(results: list[dict]) -> str:
    context_parts = []

    for index, result in enumerate(results, start=1):
        page = result.get("page_number")

        if page is None:
            source = f"Evidence {index}"
        else:
            source = f"Evidence {index}, page {page}"

        context_parts.append(
            f"[{index}] {source}\n"
            f"{result['content']}"
        )

    return "\n\n".join(context_parts)


def _extract_text(content) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []

        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(item.get("text", ""))

        return "".join(text_parts)

    return str(content)


async def repair_answer(
    query: str,
    answer: str,
    context: str,
    citation_count: int,
) -> str:
    valid_range = ", ".join(
        f"[{index}]"
        for index in range(1, citation_count + 1)
    )

    prompt = f"""
Rewrite the answer so every factual statement is supported
by the provided evidence.

Strict rules:
- Use ONLY the provided evidence.
- Do not add outside knowledge.
- Use only these citation identifiers: {valid_range}.
- Every important factual statement must contain a citation.
- Never invent or modify citation numbers.
- Keep the meaning of the original answer.
- If the evidence does not support the answer, say that clearly.

User question:
{query}

Original answer:
{answer}

Evidence:
{context}
"""

    response = await llm.ainvoke(prompt)

    return _extract_text(response.content)


async def rewrite_query(
    query: str,
    history: list[dict],
) -> str:
    """
    Convert a conversational follow-up into a standalone
    retrieval query.
    """

    if not history:
        return query

    history_text = "\n".join(
        f"{message['role']}: {message['content']}"
        for message in history
    )

    prompt = f"""
Rewrite the user's latest question into a standalone search query.

Use the conversation history only to resolve references
such as "it", "that", "they", "the previous one", etc.

Rules:
- Preserve the user's actual intent.
- Do not answer the question.
- Do not add unsupported information.
- Return ONLY the rewritten query.
- Keep it concise.

Conversation history:
{history_text}

Latest user question:
{query}
"""

    response = await llm.ainvoke(prompt)

    return _extract_text(response.content).strip()


async def generate_rag_answer(
    db,
    query: str,
    top_k: int = 5,
    history: list[dict] | None = None,
) -> dict:
    history = history or []

    # Rewrite follow-up questions into standalone retrieval queries.
    retrieval_query = await rewrite_query(
        query=query,
        history=history,
    )

    # Retrieve evidence using the rewritten query.
    results = await hybrid_search(
        db=db,
        query=retrieval_query,
        top_k=top_k,
    )

    if not results:
        return {
            "answer": (
                "I could not find relevant information "
                "in the knowledge base."
            ),
            "citations": [],
            "sources": [],
            "retrieval_query": retrieval_query,
            "evidence_grade": {
                "is_sufficient": False,
                "score": 0.0,
                "reason": "No evidence was retrieved.",
            },
            "citation_validation": {
                "is_valid": True,
                "cited_numbers": [],
                "invalid_citations": [],
            },
        }

    # Check whether the retrieved evidence is sufficient
    # to answer the user's actual question.
    evidence_grade = await grade_evidence(
        query=query,
        results=results,
    )

    # Stop before generation if evidence is insufficient.
    if not evidence_grade["is_sufficient"]:
        return {
            "answer": (
                "I could not find enough reliable information "
                "in the knowledge base to answer this question."
            ),
            "citations": [],
            "sources": [
                {
                    "chunk_id": str(result["chunk_id"]),
                    "document_id": str(result["document_id"]),
                    "page_number": result.get("page_number"),
                    "content": result["content"],
                    "score": result.get("rerank_score"),
                }
                for result in results
            ],
            "retrieval_query": retrieval_query,
            "evidence_grade": evidence_grade,
            "citation_validation": {
                "is_valid": True,
                "cited_numbers": [],
                "invalid_citations": [],
            },
        }

    context = build_context(results)

    history_text = "\n".join(
        f"{message['role']}: {message['content']}"
        for message in history
    )

    prompt = f"""
You are NexusIQ, an enterprise knowledge assistant.

Answer the user's latest question using ONLY the provided evidence.

Conversation history:
{history_text if history_text else "No previous conversation."}

Rules:
- Do not invent facts.
- Do not use outside knowledge.
- Use the conversation history only to understand references
  and conversational context.
- Factual answers must come from the provided evidence.
- Every important factual statement must have a citation.
- Use citation numbers exactly as provided: [1], [2], [3], etc.
- Never create a citation number that does not exist.
- If the evidence is insufficient, say so clearly.
- Keep the answer concise but useful.

User's latest question:
{query}

Evidence:
{context}
"""

    response = await llm.ainvoke(prompt)

    answer = _extract_text(response.content)

    # Validate generated citations.
    validation = validate_citations(
        answer=answer,
        citation_count=len(results),
    )

    # Repair once if citation validation fails.
    if not validation["is_valid"]:
        answer = await repair_answer(
            query=query,
            answer=answer,
            context=context,
            citation_count=len(results),
        )

        validation = validate_citations(
            answer=answer,
            citation_count=len(results),
        )

    # Safe fallback if repair also fails.
    if not validation["is_valid"]:
        answer = (
            "I found relevant evidence, but I could not "
            "produce a response with valid citations."
        )

    citations = []

    for index, result in enumerate(results, start=1):
        citations.append(
            {
                "citation": f"[{index}]",
                "chunk_id": str(result["chunk_id"]),
                "document_id": str(result["document_id"]),
                "page_number": result.get("page_number"),
                "chunk_index": result["chunk_index"],
                "rerank_score": result.get("rerank_score"),
            }
        )

    return {
        "answer": answer,
        "citations": citations,
        "sources": [
            {
                "chunk_id": str(result["chunk_id"]),
                "document_id": str(result["document_id"]),
                "page_number": result.get("page_number"),
                "content": result["content"],
                "score": result.get("rerank_score"),
            }
            for result in results
        ],
        "retrieval_query": retrieval_query,
        "evidence_grade": evidence_grade,
        "citation_validation": validation,
    }
