from core.inference.manage.predictor_init import PredictorInit
from core.inference.predictors.gdino_predictor import GDinoPredictor
from core.inference.predictors.lama_predictor import LaMaPredictor
from core.inference.predictors.sam2_predictor import SAM2Predictor
from core.inference.predictors.sd_inpaint_predictor import SDInpaintPredictor
import hydra

hydra.core.global_hydra.GlobalHydra.instance().clear()
hydra.initialize_config_module('static/configs')

PREDICTOR_INITS = [
    PredictorInit(
        type=SAM2Predictor,
        params={
            "device": "cuda:3",
            "config": "sam2_hiera_t.yaml",
            "checkpoint": "./static/checkpoints/sam2_hiera_tiny.pt",
        },
    ),
    PredictorInit(
        type=GDinoPredictor,
        params={
            "device": "cuda:1",
            "repo_id": "IDEA-Research/grounding-dino-base",
            "box_threshold": 0.35,
            "text_threshold": 0.25,
        },
    ),
    PredictorInit(type=LaMaPredictor, params={"device": "cuda:1"}),
    PredictorInit(
        type=SDInpaintPredictor,
        params={
            "device": "cuda:3",
            "repo_id": "stabilityai/stable-diffusion-2-inpainting",
        },
    ),
]
