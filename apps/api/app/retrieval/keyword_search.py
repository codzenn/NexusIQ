from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document_chunk import DocumentChunk


async def keyword_search(
    db: AsyncSession,
    query: str,
    top_k: int = 10,
) -> list[dict]:
    """
    PostgreSQL full-text keyword search.
    """

    search_vector = func.to_tsvector(
        "english",
        DocumentChunk.content,
    )

    search_query = func.plainto_tsquery(
        "english",
        query,
    )

    rank = func.ts_rank_cd(
        search_vector,
        search_query,
    )

    statement = (
        select(
            DocumentChunk.id,
            DocumentChunk.document_id,
            DocumentChunk.content,
            DocumentChunk.chunk_index,
            DocumentChunk.page_number,
            rank.label("score"),
        )
        .where(search_vector.op("@@")(search_query))
        .order_by(rank.desc())
        .limit(top_k)
    )

    result = await db.execute(statement)

    rows = result.all()

    return [
        {
            "chunk_id": row.id,
            "document_id": row.document_id,
            "content": row.content,
            "chunk_index": row.chunk_index,
            "page_number": row.page_number,
            "score": float(row.score),
        }
        for row in rows
    ]
