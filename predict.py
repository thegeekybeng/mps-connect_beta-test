from fastapi import Header, HTTPException, Depends

API_KEY = os.environ.get("API_KEY", "")


async def require_key(x_api_key: str = Header(None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(401, "invalid api key")


@app.post("/predict")
def predict(req: PredictReq, _=Depends(require_key)): ...
