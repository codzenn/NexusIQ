from langchain_openai import ChatOpenAI

from app.core.config import settings
from app.retrieval.hybrid import hybrid_search


llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=settings.openai_api_key,
    temperature=0,
)


def build_context(results: list[dict]) -> str:
    context_parts = []

    for index, result in enumerate(results, start=1):
        page = result.get("page_number")

        source = (
            f"Document chunk {index}"
            if page is None
            else f"Document chunk {index}, page {page}"
        )

        context_parts.append(
            f"[{source}]\n{result['content']}"
        )

    return "\n\n".join(context_parts)


async def generate_rag_answer(
    db,
    query: str,
    top_k: int = 5,
) -> dict:
    results = await hybrid_search(
        db=db,
        query=query,
        top_k=top_k,
    )

    if not results:
        return {
            "answer": "I could not find relevant information in the knowledge base.",
            "citations": [],
            "sources": [],
        }

    context = build_context(results)

    prompt = f"""
You are NexusIQ, an enterprise knowledge assistant.

Answer the user's question using ONLY the provided evidence.

Rules:
- Do not invent facts.
- Do not use outside knowledge.
- If the evidence is insufficient, say so clearly.
- Keep the answer concise but useful.
- Reference evidence using citation numbers such as [1], [2].

User question:
{query}

Evidence:
{context}
"""

    response = await llm.ainvoke(prompt)

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
        "answer": response.content,
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
    }
