import traceback

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from deps.manager import get_predictor
from utils.convert import image_to_buffer, upload_file_to_image

router = APIRouter(prefix="/api/v1")


@router.post("/lama")
async def predict(
    image: UploadFile = File(...), mask: UploadFile = File(...), prompt: str = Form(...)
):
    try:
        image = await upload_file_to_image(image)
        mask = await upload_file_to_image(mask)

        async with get_predictor("SDInpaintPredictor") as predictor:
            inpainted_image = predictor(image, mask, prompt)

        buffer = image_to_buffer(inpainted_image)
        return StreamingResponse(buffer, media_type="image/png")

    except Exception as e:
        print(traceback.format_exc())
        return HTTPException(500)
