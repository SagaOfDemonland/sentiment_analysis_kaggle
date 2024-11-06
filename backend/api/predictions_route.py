from src.inference.inference import BertInference
from fastapi import HTTPException,APIRouter
from src.utils.validate_review_message import TextValidator
from .models.prediction_request import PredictionRequest,PredictionResponse
from src.config import USE_GPU
# Initialize FastAPI app
router = APIRouter()
MODEL_ID = "sagaofdemonland/review_sentiment_analysis_binary"

# Initialize the BertInference model
bert_inference = BertInference(model_path=MODEL_ID, use_gpu=USE_GPU)
text_validator = TextValidator()

@router.post("/api/predict", response_model=PredictionResponse)
async def predict_sentiment(request: PredictionRequest):
    """
    Predict the sentiment of the input text with comprehensive input validation.
    """
    try:
        # Validate the review text
        is_valid, error_message = text_validator.validate_review_text(request.text)
        if not is_valid:
            raise HTTPException(
                status_code=422,  # Unprocessable Entity
                detail=error_message
            )

        # Clean the text before prediction (remove extra spaces, normalize)
        cleaned_text = ' '.join(request.text.split())
        
        # Rate limiting check could be added here
        
        # Perform inference using BertInference
        predicted_label, confidence = bert_inference.predict_single(cleaned_text)
        
        # Validate confidence threshold
        if confidence < 0.6:  # Adjust threshold as needed
            return PredictionResponse(
                predicted_label="uncertain",
                confidence=confidence,
                warning="Low confidence prediction, consider rephrasing your review."
            )

        return PredictionResponse(
            predicted_label=predicted_label,
            confidence=confidence
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )