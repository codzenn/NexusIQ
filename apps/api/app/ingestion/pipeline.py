from pathlib import Path
from uuid import UUID

from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.ingestion.chunker import create_chunks
from app.ingestion.cleaner import clean_text
from app.ingestion.parser import parse_document
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.retrieval.embeddings import generate_embedding


async def process_document(
    db: AsyncSession,
    document: Document,
    file_path: str,
) -> None:
    """
    Process a document through the complete ingestion pipeline:

    Parse → Clean → Chunk → Embed → Store
    """

    try:
        document.status = "processing"
        document.error_message = None

        await db.flush()

        # 1. Parse
        pages = await parse_document(
            file_path=file_path,
            content_type=document.content_type,
        )

        # 2. Clean
        cleaned_pages = []

        for page in pages:
            cleaned_text = clean_text(page["text"])

            if not cleaned_text:
                continue

            cleaned_pages.append(
                {
                    "text": cleaned_text,
                    "page_number": page["page_number"],
                }
            )

        # 3. Chunk
        chunks = create_chunks(cleaned_pages)

        if not chunks:
            raise ValueError("No usable text found in document")

        # Remove existing chunks if document is reprocessed
        await db.execute(
            delete(DocumentChunk).where(
                DocumentChunk.document_id == document.id
            )
        )

        # 4. Embed + store
        for chunk in chunks:
            embedding = await generate_embedding(
                chunk["content"]
            )

            document_chunk = DocumentChunk(
                document_id=document.id,
                content=chunk["content"],
                chunk_index=chunk["chunk_index"],
                page_number=chunk["page_number"],
                embedding=embedding,
            )

            db.add(document_chunk)

        # 5. Mark complete
        document.status = "completed"

        await db.commit()

    except Exception as exc:
        await db.rollback()

        document.status = "failed"
        document.error_message = str(exc)

        await db.commit()

        raise

    finally:
        # Remove temporary uploaded file
        path = Path(file_path)

        if path.exists():
            path.unlink(missing_ok=True)
