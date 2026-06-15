from pydantic import BaseModel, Field
from typing import Literal, Optional


ModelKey = Literal["lr_tfidf", "lr_word2vec", "bilstm", "distilbert"]


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=512, example="I love this movie so much!")
    model: ModelKey = Field("lr_tfidf", description="Model to use for inference")


class BatchRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=50)
    model: ModelKey = "lr_tfidf"


class PredictResponse(BaseModel):
    text: str
    model: str
    label: Literal["positive", "negative"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    latency_ms: float
    preprocessing_steps: list[str]


class BatchResponse(BaseModel):
    model: str
    total: int
    results: list[PredictResponse]
    total_latency_ms: float


class ModelInfo(BaseModel):
    key: str
    name: str
    description: str
    representation: str
    f1_score: Optional[float]
    loaded: bool
    contextual: bool
