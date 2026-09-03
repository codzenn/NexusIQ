from fastapi import APIRouter

from app.api.documents import router as documents_router
from app.api.health import router as health_router
from app.api.search import router as search_router


router = APIRouter(prefix="/api/v1")

router.include_router(health_router)
router.include_router(documents_router)
router.include_router(search_router)
