import traceback
from typing import Any, List

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from core import get_model, get_model_manager
from schemas.image_model import ImageModel
from schemas.mask_model import MaskModel

router = APIRouter(prefix="/api/v1")


class GroundedSamRequest(BaseModel):
    image: ImageModel
    prompt: str


class GroundedSamResponse(BaseModel):
    masks: List[MaskModel]
    scores: List[float]


@router.post("/predict/grounded-sam", response_model=GroundedSamResponse)
async def predict(
    request: GroundedSamRequest, model_manager: Any = Depends(get_model_manager)
):
    try:
        image = await request.image.to_image()
        prompt = request.prompt

        async with get_model(model_manager, "grounding_dino") as model:
            boxes, scores = model(image, prompt)

        async with get_model(model_manager, "sam") as model:
            masks = [model(image, box) for box in boxes]

        masks = [MaskModel.from_array(mask) for mask in masks]
        return GroundedSamResponse(masks=masks, scores=scores)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
