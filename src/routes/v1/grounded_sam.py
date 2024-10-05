import io
import traceback
import zipfile

import PIL.Image
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from deps.manager import get_predictor

router = APIRouter()


@router.post("/grounded-sam")
async def predict(
    image: UploadFile = File(...),
    prompt: str = Form(...),
):
    try:
        buffer = await image.read()
        image = PIL.Image.open(io.BytesIO(buffer))

        async with get_predictor("GDinoPredictor") as predictor:
            scores, boxes = predictor(image, prompt)

        zip_buffer = io.BytesIO()

        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            async with get_predictor("SAM2Predictor") as predictor:
                for score, box in zip(scores, boxes):
                    mask = predictor(image, box=box)

                    mask_buffer = io.BytesIO()
                    mask.save(mask_buffer, format="PNG")
                    mask_buffer.seek(0)

                    zip_file.writestr(f"{score:.3f}.png", mask_buffer.getvalue())

        zip_buffer.seek(0)

        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={"Content-Disposition": "attachment; filename=masks.zip"},
        )

    except Exception as e:
        print(traceback.format_exc())
        return HTTPException(500)
