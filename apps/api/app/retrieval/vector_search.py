from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document_chunk import DocumentChunk
from app.retrieval.embeddings import generate_embedding


async def vector_search(
    db: AsyncSession,
    query: str,
    top_k: int = 5,
    document_id: UUID | None = None,
) -> list[dict]:
    """
    Perform semantic similarity search using pgvector.
    """

    query_embedding = await generate_embedding(query)

    similarity = DocumentChunk.embedding.cosine_distance(
        query_embedding
    )

    statement = (
        select(
            DocumentChunk.id,
            DocumentChunk.document_id,
            DocumentChunk.content,
            DocumentChunk.chunk_index,
            DocumentChunk.page_number,
            similarity.label("distance"),
        )
        .order_by(similarity)
        .limit(top_k)
    )

    if document_id is not None:
        statement = statement.where(
            DocumentChunk.document_id == document_id
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
            "distance": float(row.distance),
            "score": 1 - float(row.distance),
        }
        for row in rows
    ]
