import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from deps.manager import manager
from routes.v1 import grounded_sam, lama, sam2, sd_inpaint
from dotenv import load_dotenv

load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    manager.init()
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
app.include_router(lama.router)
app.include_router(sam2.router)
app.include_router(sd_inpaint.router)
