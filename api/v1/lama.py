import traceback
from typing import Any, List

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from core import get_model, get_model_manager
from schemas.image_model import ImageModel
from schemas.mask_model import MaskModel
from utils.image import expand_mask

router = APIRouter(prefix="/api/v1")


class LaMaRequest(BaseModel):
    image: ImageModel
    masks: List[MaskModel]
    expand: bool = True


class LaMaResponse(BaseModel):
    image: ImageModel


@router.post("/predict/lama", response_model=LaMaResponse)
async def predict(
    request: LaMaRequest, model_manager: Any = Depends(get_model_manager)
):
    try:
        image = await request.image.to_image()
        array_masks = [mask.to_array() for mask in request.masks]

        if request.expand:
            array_masks = [expand_mask(mask) for mask in array_masks]

        combined_mask = np.maximum.reduce(array_masks)

        async with get_model(model_manager, "lama") as model:
            inpainted_image = model(image, combined_mask)

        return LaMaResponse(image=ImageModel.from_pil_image(inpainted_image))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
