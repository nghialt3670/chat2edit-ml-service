from core.inference.manage.predictor_init import PredictorInit
from core.inference.predictors.gdino_predictor import GDinoPredictor
from core.inference.predictors.lama_predictor import LaMaPredictor
from core.inference.predictors.sam2_predictor import SAM2Predictor
from core.inference.predictors.sd_inpaint_predictor import SDInpaintPredictor

PREDICTOR_INITS = [
    PredictorInit(
        type=GDinoPredictor,
        params={
            "device": "cuda",
            "repo_id": "IDEA-Research/grounding-dino-base",
            "box_threshold": 0.35,
            "text_threshold": 0.25,
        },
    ),
    PredictorInit(type=LaMaPredictor, params={"device": "cuda"}),
    PredictorInit(
        type=SAM2Predictor,
        params={
            "device": "cuda",
            "config": "//home/nghialt/projects/chat2edit/ml-service/static/configs/sam2_hiera_t.yaml",
            "checkpoint": "//home/nghialt/projects/chat2edit/ml-service/static/checkpoints/sam2_hiera_tiny.pt",
        },
    ),
    PredictorInit(
        type=SDInpaintPredictor,
        params={
            "device": "cuda",
            "repo_id": "stabilityai/stable-diffusion-2-inpainting",
        },
    ),
]
