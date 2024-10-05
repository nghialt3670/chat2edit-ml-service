import traceback
from typing import List, Literal

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import TypeAdapter

from core.types.box import Box
from core.types.point import Point
from deps.manager import get_predictor
from utils.convert import image_to_buffer, upload_file_to_image

router = APIRouter(prefix="/api/v1")


@router.post("/sam2")
async def predict(
    image: UploadFile = File(...),
    box: str = Form(None),
    points: str = Form(None),
    point_labels: str = Form(None),
):
    try:
        image = await upload_file_to_image(image)

        try:
            if box:
                box = TypeAdapter(Box).validate_json(box)
            if points:
                assert len(points) == len(point_labels)
                points = TypeAdapter(List[Point]).validate_json(points)
                point_labels = TypeAdapter(List[Literal[0, 1]]).validate_json(
                    point_labels
                )

        except Exception as e:
            return HTTPException(422, str(e))

        async with get_predictor("SAM2Predictor") as predictor:
            mask = predictor(image, box, points, point_labels)

        buffer = image_to_buffer(mask)
        return StreamingResponse(buffer, media_type="image/png")

    except Exception as e:
        print(traceback.format_exc())
        return HTTPException(500)
