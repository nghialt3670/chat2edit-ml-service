from typing import Any, List, Tuple

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from core import get_model, get_model_manager
from schemas.image_model import ImageModel

router = APIRouter(prefix="/api/v1")


class GroundingDINORequest(BaseModel):
    image: ImageModel
    prompt: str


class GroundingDINOResponse(BaseModel):
    boxes: List[Tuple[int, int, int, int]]
    scores: List[float]


@router.post("/predict/grounding-dino", response_model=GroundingDINOResponse)
async def predict(
    request: GroundingDINORequest, model_manager: Any = Depends(get_model_manager)
):
    try:
        image = await request.image.to_image()
        prompt = request.prompt

        async with get_model(model_manager, "grounding_dino") as model:
            boxes, scores = model(image, prompt)

        return GroundingDINOResponse(boxes=boxes, scores=scores)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
