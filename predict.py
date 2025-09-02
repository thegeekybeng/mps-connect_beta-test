"""Prediction endpoint for MPS Connect testers."""

import os
from fastapi import FastAPI, Header, HTTPException, Depends
from pydantic import BaseModel

app = FastAPI()


class PredictReq(BaseModel):
    """Request model for predictions."""

    texts: list[str]


API_KEY = os.environ.get("API_KEY", "")


async def require_key(x_api_key: str = Header(None)):
    """Validate API key from request headers."""
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(401, "invalid api key")


@app.post("/predict")
def predict(_req: PredictReq, _=Depends(require_key)):
    """Handle prediction requests with API key validation."""
    return {"message": "Not implemented"}
