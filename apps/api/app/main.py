from fastapi import FastAPI

from app.api.router import router


app = FastAPI(
    title="NexusIQ API",
    version="0.1.0",
    description=(
        "Enterprise Agentic Intelligence Platform"
    ),
)

app.include_router(router)


@app.get("/")
async def root():
    return {
        "name": "NexusIQ",
        "description": "Enterprise Agentic Intelligence Platform",
        "version": "0.1.0",
        "status": "running",
    }
