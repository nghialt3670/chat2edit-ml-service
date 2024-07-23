import traceback
from typing import Any, List

import numpy as np
from core import get_model, get_model_manager
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from schemas.image_model import ImageModel
from schemas.mask_model import MaskModel

router = APIRouter(prefix="/api/v1")


class StableDiffusionInpaintRequest(BaseModel):
    image: ImageModel
    masks: List[MaskModel]
    prompt: str


class StableDiffusionInpaintResponse(BaseModel):
    image: ImageModel


@router.post("/predict/lama", response_model=StableDiffusionInpaintResponse)
async def predict(
    request: StableDiffusionInpaintRequest,
    model_manager: Any = Depends(get_model_manager),
):
    try:
        image = await request.image.to_image()
        array_masks = [mask.to_array() for mask in request.masks]

        combined_mask = np.maximum.reduce(array_masks)

        async with get_model(model_manager, "stable_diffusion_inpaint") as model:
            inpainted_image = model(image, combined_mask, request.prompt)

        return StableDiffusionInpaintResponse(
            image=ImageModel.from_pil_image(inpainted_image)
        )

    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(500, detail=str(e))
