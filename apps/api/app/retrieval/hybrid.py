from sqlalchemy.ext.asyncio import AsyncSession

from app.retrieval.keyword_search import keyword_search
from app.retrieval.vector_search import vector_search
from app.retrieval.reranker import rerank


def reciprocal_rank_fusion(
    result_lists: list[list[dict]],
    k: int = 60,
) -> list[dict]:
    """
    Combine ranked result lists using Reciprocal Rank Fusion.

    RRF score = sum(1 / (k + rank))
    """

    fused: dict[str, dict] = {}

    for results in result_lists:
        for rank, result in enumerate(results, start=1):
            chunk_id = str(result["chunk_id"])

            if chunk_id not in fused:
                fused[chunk_id] = {
                    **result,
                    "rrf_score": 0.0,
                }

            fused[chunk_id]["rrf_score"] += 1 / (k + rank)

    return sorted(
        fused.values(),
        key=lambda item: item["rrf_score"],
        reverse=True,
    )


async def hybrid_search(
    db: AsyncSession,
    query: str,
    top_k: int = 5,
) -> list[dict]:

    candidate_k = max(top_k * 4, 20)

    vector_results = await vector_search(
        db=db,
        query=query,
        top_k=candidate_k,
    )

    keyword_results = await keyword_search(
        db=db,
        query=query,
        top_k=candidate_k,
    )

    fused_results = reciprocal_rank_fusion(
        [
            vector_results,
            keyword_results,
        ]
    )

    candidates = fused_results[:candidate_k]

    return rerank(
        query=query,
        results=candidates,
        top_k=top_k,
    )
