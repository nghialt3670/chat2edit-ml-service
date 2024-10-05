import io
import traceback

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from deps.manager import get_predictor
from utils.convert import upload_file_to_image

router = APIRouter()


@router.post("/lama")
async def predict(
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
):
    try:
        image = await upload_file_to_image(image)
        mask = await upload_file_to_image(mask)

        async with get_predictor("LaMaPredictor") as predictor:
            inpainted_image = predictor(image, mask)

        buffer = io.BytesIO()
        inpainted_image.save(buffer, format="PNG")
        buffer.seek(0)

        return StreamingResponse(buffer, media_type="image/png")

    except Exception:
        print(traceback.format_exc())
        return HTTPException(500)
