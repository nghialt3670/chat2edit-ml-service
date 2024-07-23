import logging
from contextlib import asynccontextmanager

from api.v1 import (grounded_sam, grounding_dino, lama, sam,
                    stable_diffusion_inpaint)
from core import model_manager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from GPUtil import getGPUs

logger = logging.getLogger("uvicorn")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not getGPUs():
        logger.warning("No GPUs found")
    logger.info(f"Creating models: {model_manager.get_name_to_quantity()}")
    model_manager.create_models()
    for gpu in getGPUs():
        logger.info(
            f"GPU {gpu.id}: {gpu.memoryUsed}/{gpu.memoryTotal} MB, Load: {gpu.load*100:.1f}%"
        )
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(grounded_sam.router)
app.include_router(grounding_dino.router)
app.include_router(lama.router)
app.include_router(sam.router)
app.include_router(stable_diffusion_inpaint.router)
