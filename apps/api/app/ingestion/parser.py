from pathlib import Path

from docx import Document as DocxDocument
from pypdf import PdfReader


async def parse_document(
    file_path: str,
    content_type: str,
) -> list[dict]:
    path = Path(file_path)

    if content_type == "application/pdf":
        return _parse_pdf(path)

    if content_type == (
        "application/vnd.openxmlformats-officedocument."
        "wordprocessingml.document"
    ):
        return _parse_docx(path)

    if content_type in {
        "text/plain",
        "text/markdown",
    }:
        return _parse_text(path)

    raise ValueError(f"Unsupported file type: {content_type}")


def _parse_pdf(path: Path) -> list[dict]:
    reader = PdfReader(path)

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        pages.append(
            {
                "text": text,
                "page_number": page_number,
            }
        )

    return pages


def _parse_docx(path: Path) -> list[dict]:
    document = DocxDocument(path)

    text = "\n".join(
        paragraph.text
        for paragraph in document.paragraphs
        if paragraph.text.strip()
    )

    return [
        {
            "text": text,
            "page_number": None,
        }
    ]


def _parse_text(path: Path) -> list[dict]:
    return [
        {
            "text": path.read_text(
                encoding="utf-8"
            ),
            "page_number": None,
        }
    ]
