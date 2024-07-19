from io import BytesIO
from typing import Literal, Optional, Union

import aiohttp
from PIL import Image
from pydantic import AnyUrl, Base64UrlStr, BaseModel, Field, HttpUrl

from utils.image import base64_to_image, image_to_base64


class ImageModel(BaseModel):
    src: str
    src_type: Optional[Literal["http_url", "data_url", "base64"]] = None

    async def to_image(self) -> Image.Image:
        if self.src_type is None:
            if self.src.startswith("http:") or self.src.startswith("https:"):
                self.src_type = "http_url"
            elif self.src.startswith("data:"):
                self.src_type = "data_url"
            else:
                self.src_type = "base64"

        if self.src_type == "data_url":
            _, base64 = self.src.split(",")
            return base64_to_image(base64)
        elif self.src_type == "http_url":
            async with aiohttp.ClientSession() as session:
                async with session.get(str(self.src)) as response:
                    image_bytes = await response.read()
            return Image.open(BytesIO(image_bytes))
        elif self.src_type == "base64":
            return base64_to_image(self.src)
        else:
            raise ValueError("Unsupported src_type or URL format")

    @classmethod
    def from_pil_image(cls, image: Image.Image) -> "ImageModel":
        base64 = image_to_base64(image)
        return cls(src=base64, src_type="base64")
