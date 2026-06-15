"""
FastAPI application — routes, lifespan, static files.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request

from app.model_manager import ModelManager
from app.schemas import PredictRequest, BatchRequest, PredictResponse, BatchResponse, ModelInfo

manager = ModelManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    manager.load_all()
    yield
    manager.unload_all()


app = FastAPI(
    title="Sentiment Analysis API",
    description="Pipeline NLP · TALN · UMMTO 4ᵉ Ing-IA",
    version="1.0.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", include_in_schema=False)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
async def health():
    return {"status": "ok", "models_loaded": manager.loaded_models()}


@app.get("/models", response_model=list[ModelInfo])
async def get_models():
    return manager.model_info()


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    if req.model not in ModelManager.AVAILABLE:
        raise HTTPException(status_code=400, detail=f"Unknown model: {req.model}")
    return manager.predict(req.text, req.model)


@app.post("/predict/batch", response_model=BatchResponse)
async def predict_batch(req: BatchRequest):
    if req.model not in ModelManager.AVAILABLE:
        raise HTTPException(status_code=400, detail=f"Unknown model: {req.model}")
    return manager.predict_batch(req.texts, req.model)


@app.post("/compare")
async def compare(req: PredictRequest):
    results = {}
    for key in ModelManager.AVAILABLE:
        try:
            res = manager.predict(req.text, key)
            results[key] = res.model_dump()
        except Exception as e:
            results[key] = {"error": str(e)}
    return {"text": req.text, "comparisons": results}
