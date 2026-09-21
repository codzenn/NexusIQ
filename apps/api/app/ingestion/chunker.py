from langchain_text_splitters import RecursiveCharacterTextSplitter


splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=120,
    separators=[
        "\n\n",
        "\n",
        ". ",
        " ",
        "",
    ],
)


def create_chunks(
    pages: list[dict],
) -> list[dict]:

    chunks = []

    chunk_index = 0

    for page in pages:
        text = page["text"]

        page_chunks = splitter.split_text(text)

        for chunk in page_chunks:
            chunks.append(
                {
                    "content": chunk,
                    "chunk_index": chunk_index,
                    "page_number": page["page_number"],
                }
            )

            chunk_index += 1

    return chunks
