# app.py
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

API_KEY = "mps-85-whampoa"

app = FastAPI(title="WhiteVision Hierarchical ZS API")

# CORS: keep wide-open while testing. Tighten allow_origins to your Pages origin later.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],                         # e.g. ["https://thegeekybeng.github.io"]
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key"], # <- include your custom header
)

def require_api_key(x_api_key: Optional[str]):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

@app.get("/healthz")
def healthz():
    # No API key here — avoids preflight when you call it from the browser
    return {"ok": True, "labels": 129, "tops": 18}

class PredictIn(BaseModel):
    texts: List[str]
    threshold_top: float = 0.30
    threshold_child: float = 0.36
    top_k_top: int = 6
    top_k_child: int = 6
    top_k_total: int = 18
    seed_prior_threshold: float = 0.45
    use_priors: bool = True
    return_providers: bool = True

@app.post("/predict")
def predict(body: PredictIn, x_api_key: Optional[str] = Header(None)):
    require_api_key(x_api_key)
    # ... your real inference here ...
    return {"predictions": [{
        "text": body.texts[0],
        "labels": ["social_support/comcare_short_mid_term"],
        "scores": {"social_support/comcare_short_mid_term": 0.61},
        "top_categories": [{"top": "social_support", "score": 0.72}],
        "providers": {"social_support/comcare_short_mid_term":
                      {"provider": "MSF/SSO", "type": "email", "channel": ""}}
    }]}

@app.get("/")
def root():
    return {"ok": True, "endpoints": ["/healthz", "/predict"]}
