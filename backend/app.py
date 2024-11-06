import sys
from pathlib import Path

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)
from fastapi import FastAPI,HTTPException
from api.predictions_route import router as predictions_router  
from starlette.middleware.cors import CORSMiddleware
import logging
import traceback
import socket
import os
from dotenv import load_dotenv
load_dotenv()
logger = logging.getLogger("customer_review_sentiment_analysis_backend")
logger.setLevel(logging.INFO)

# Initialize the FastAPI app
app = FastAPI()

# Include the inference router
app.include_router(predictions_router)
@app.on_event("startup")
async def startup_event():
    """
    Initializes necessary components upon application startup.
    """

    

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/healthcheck")
async def healthcheck():
    """
    API endpoint for health check.

    Returns:
    - str: Returns 'OK' if the server is healthy

    Raises:
    - HTTPException: If an error occurs during health check
    """
    try:
        logger.info("Healthcheck endpoint was hit")

    except Exception as e:
        logger.error(f"[healthcheck] Exception occurred: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    try:
        socket.setdefaulttimeout(30)
        import uvicorn
        uvicorn.run("app:app", host="0.0.0.0", port=int(os.environ.get('PORT', 8001)), log_level="info")
    except Exception as e:
        logger.error(f"[main] Exception occurred: {str(e)}")
        logger.error(traceback.format_exc())
