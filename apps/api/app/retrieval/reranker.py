from sentence_transformers import CrossEncoder


MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_reranker: CrossEncoder | None = None


def get_reranker() -> CrossEncoder:
    global _reranker

    if _reranker is None:
        _reranker = CrossEncoder(MODEL_NAME)

    return _reranker


def rerank(
    query: str,
    results: list[dict],
    top_k: int = 5,
) -> list[dict]:
    if not results:
        return []

    model = get_reranker()

    pairs = [
        [query, result["content"]]
        for result in results
    ]

    scores = model.predict(pairs)

    reranked = []

    for result, score in zip(results, scores):
        reranked.append(
            {
                **result,
                "rerank_score": float(score),
            }
        )

    reranked.sort(
        key=lambda item: item["rerank_score"],
        reverse=True,
    )

    return reranked[:top_k]
