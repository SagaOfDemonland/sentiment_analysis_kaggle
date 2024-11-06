from pydantic import BaseModel, Field
from typing import Optional
class PredictionRequest(BaseModel):
    text: Optional[str] = Field(
        None, 
        example="The food was amazing and the service was fantastic!", 
        description="The text input to analyze sentiment for. Required for making a prediction."
    )

class PredictionResponse(BaseModel):
    predicted_label: Optional[int] = Field(
        None, 
        example=1, 
        description="The predicted sentiment label. 1 for positive and 0 for negative sentiment."
    )
    confidence: Optional[float] = Field(
        None, 
        example=0.95, 
        description="Confidence score for the predicted label, ranging from 0 to 1."
    )