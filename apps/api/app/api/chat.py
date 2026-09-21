from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.services.conversation import (
    add_message,
    create_conversation,
    get_conversation,
    get_recent_messages,
)
from app.services.rag import generate_rag_answer


router = APIRouter(
    prefix="/chat",
    tags=["Chat"],
)


class ChatRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=10)
    conversation_id: UUID | None = None


@router.post("")
async def chat(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
):
    # Create a new conversation when one isn't supplied.
    if request.conversation_id is None:
        conversation = await create_conversation(db)
    else:
        conversation = await get_conversation(
            db=db,
            conversation_id=request.conversation_id,
        )

        if conversation is None:
            raise HTTPException(
                status_code=404,
                detail="Conversation not found",
            )

    # Load previous conversation context.
    previous_messages = await get_recent_messages(
        db=db,
        conversation_id=conversation.id,
        limit=10,
    )

    history = [
        {
            "role": message.role,
            "content": message.content,
        }
        for message in previous_messages
    ]

    # Persist the user's message.
    await add_message(
        db=db,
        conversation_id=conversation.id,
        role="user",
        content=request.query,
    )

    # Generate grounded RAG response.
    result = await generate_rag_answer(
        db=db,
        query=request.query,
        top_k=request.top_k,
        history=history,
    )

    # Persist the assistant's response.
    await add_message(
        db=db,
        conversation_id=conversation.id,
        role="assistant",
        content=result["answer"],
    )

    return {
        "conversation_id": str(conversation.id),
        **result,
    }
