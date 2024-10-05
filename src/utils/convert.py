import io

import PIL.Image
from fastapi import UploadFile
from PIL.Image import Image


async def upload_file_to_image(file: UploadFile) -> Image:
    buffer = await file.read()
    image = PIL.Image.open(io.BytesIO(buffer))
    return image
