import tempfile
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.ingestion.pipeline import process_document
from app.models.document import Document

router = APIRouter(prefix="/documents", tags=["Documents"])


ALLOWED_CONTENT_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "text/plain",
    "text/markdown",
}


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is required",
        )

    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}",
        )

    suffix = Path(file.filename).suffix

    temporary_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
    )

    temporary_path = temporary_file.name

    try:
        contents = await file.read()

        temporary_file.write(contents)
        temporary_file.close()

        document = Document(
            filename=file.filename,
            content_type=file.content_type,
            source_type="upload",
            status="pending",
        )

        db.add(document)

        await db.flush()

        await process_document(
            db=db,
            document=document,
            file_path=temporary_path,
        )

        return {
            "id": str(document.id),
            "filename": document.filename,
            "status": document.status,
        }

    except Exception as exc:
        temporary_file.close()

        raise HTTPException(
            status_code=500,
            detail=f"Document processing failed: {exc}",
        )
