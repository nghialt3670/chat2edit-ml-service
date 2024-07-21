from typing import Any, List, Literal, Optional, Tuple

from core import get_model, get_model_manager
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from schemas.image_model import ImageModel
from schemas.mask_model import MaskModel

router = APIRouter(prefix="/api/v1")


class SamRequest(BaseModel):
    image: ImageModel
    box: Optional[Tuple[int, int, int, int]] = Field(default=None)
    points: Optional[List[Tuple[int, int]]] = Field(default=None)
    point_labels: Optional[List[Literal[0, 1]]] = Field(default=None)


class SamResponse(BaseModel):
    mask: MaskModel


@router.post("/predict/sam", response_model=SamResponse)
async def predict(request: SamRequest, model_manager: Any = Depends(get_model_manager)):
    try:
        image = await request.image.to_image()
        box = request.box
        points = request.points
        point_labels = request.point_labels

        async with get_model(model_manager, "sam") as model:
            mask_array = model(image, box, points, point_labels)

        return SamResponse(mask=MaskModel.from_array(mask_array))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
