from src.inference.inference import BertInference
from fastapi import HTTPException,APIRouter
from pathlib import Path
from .models.prediction_request import PredictionRequest,PredictionResponse
from src.config import USE_GPU
# Initialize FastAPI app
router = APIRouter()
project_root = Path(__file__).parent.parent.parent 
model_path = project_root / "data" / "models" / "bert-base-uncased_v1"

# Initialize the BertInference model
bert_inference = BertInference(model_path=str(model_path), use_gpu=USE_GPU)

@router.post("/api/predict", response_model=PredictionResponse)
async def predict_sentiment(request: PredictionRequest):
    """
    Predict the sentiment of the input text.
    """
    try:
        # Perform inference using BertInference
        predicted_label, confidence = bert_inference.predict_single(request.text)

        return PredictionResponse(
            predicted_label=predicted_label,
            confidence=confidence
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
