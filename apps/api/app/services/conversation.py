from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.conversation import Conversation
from app.models.message import Message


async def create_conversation(
    db: AsyncSession,
) -> Conversation:
    conversation = Conversation()

    db.add(conversation)

    await db.commit()
    await db.refresh(conversation)

    return conversation


async def get_conversation(
    db: AsyncSession,
    conversation_id: UUID,
) -> Conversation | None:
    result = await db.execute(
        select(Conversation).where(
            Conversation.id == conversation_id
        )
    )

    return result.scalar_one_or_none()


async def add_message(
    db: AsyncSession,
    conversation_id: UUID,
    role: str,
    content: str,
) -> Message:
    message = Message(
        conversation_id=conversation_id,
        role=role,
        content=content,
    )

    db.add(message)

    await db.commit()
    await db.refresh(message)

    return message


async def get_recent_messages(
    db: AsyncSession,
    conversation_id: UUID,
    limit: int = 10,
) -> list[Message]:
    result = await db.execute(
        select(Message)
        .where(
            Message.conversation_id == conversation_id
        )
        .order_by(Message.created_at.desc())
        .limit(limit)
    )

    messages = list(result.scalars().all())

    messages.reverse()

    return messages
