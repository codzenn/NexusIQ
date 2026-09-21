from langchain_openai import OpenAIEmbeddings

from app.core.config import settings


embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=settings.openai_api_key,
)


async def generate_embedding(text: str) -> list[float]:
    return await embeddings.aembed_query(text)
