# api/app.py
import os, json
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np

API_KEY = "mps-85-whampoa"

MODEL_DIR = os.getenv("MODEL_DIR", "/app/artifacts_zs_hier_plus")
PROVIDERS_JSON = os.getenv("PROVIDERS_JSON", "/app/providers_map.json")

# -------- tiny loader for your saved label embeddings --------
def load_artifacts(model_dir: str):
    art_path = os.path.join(model_dir, "artifacts.json")
    if not os.path.isfile(art_path):
        raise FileNotFoundError(f"Missing {art_path}. Build with label_embed_zero_shot_hier.py.")
    with open(art_path, "r") as f:
        arts = json.load(f)
    # expect: {"labels":[...], "tops":[...], "label_vecs":[[...],...], "top_vecs":[[...],...]}
    labels = arts.get("labels") or []
    tops = arts.get("tops") or []
    L = np.array(arts.get("label_vecs") or [], dtype="float32")
    T = np.array(arts.get("top_vecs") or [], dtype="float32")
    if L.size == 0 or T.size == 0:
        raise ValueError("Embeddings not found in artifacts.json")
    # normalize
    def norm(v):
        n = np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
        return v / n
    return {
        "labels": labels,
        "tops": tops,
        "L": norm(L),
        "T": norm(T),
    }

def load_providers(pth: str) -> Dict[str, Any]:
    if not os.path.isfile(pth):
        return {}
    with open(pth, "r") as f:
        return json.load(f)

ARTS = load_artifacts(MODEL_DIR)
PROVIDERS = load_providers(PROVIDERS_JSON)

# -------- FastAPI app --------
app = FastAPI(title="WhiteVision Hierarchical Zero-Shot API")

# CORS: allow everything for testers (lock down later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*", "X-API-Key", "Content-Type"],
)

# -------- Schemas --------
class PredictRequest(BaseModel):
    texts: List[str]
    threshold_top: float = 0.30
    threshold_child: float = 0.36
    top_k_top: int = 6
    top_k_child: int = 6
    top_k_total: int = 18
    seed_prior_threshold: float = 0.45
    use_priors: bool = True
    return_providers: bool = True

# -------- Utilities --------
def encode(texts: List[str]) -> np.ndarray:
    # Zero-shot “encoder” stub to keep this file self-contained:
    # In your real code you used sentence-transformers and cached vectors;
    # here we just map to a stable random-like vector using hash (deterministic).
    rngs = [np.random.default_rng(abs(hash(t)) % (2**32)) for t in texts]
    vecs = np.vstack([rng.normal(size=(ARTS["L"].shape[1],)).astype("float32") for rng in rngs])
    vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
    return vecs

def cosine_sim(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return A @ B.T  # both rows normalized

def _predict_one(text: str, req: PredictRequest) -> Dict[str, Any]:
    q = encode([text])  # (1, d)
    sim_top = cosine_sim(q, ARTS["T"])[0]  # (n_top,)
    # pick top categories by score
    order_top = np.argsort(sim_top)[::-1][:req.top_k_top]
    top_cats = []
    valid_tops_idx = []
    for j in order_top:
        sc = float(sim_top[j])
        if sc < req.threshold_top: 
            continue
        top_cats.append({"top": ARTS["tops"][j], "score": round(sc, 4)})
        valid_tops_idx.append(j)

    # child scores
    sim_lab = cosine_sim(q, ARTS["L"])[0]  # (n_label,)
    order_lab = np.argsort(sim_lab)[::-1][: max(req.top_k_total, 64)]
    labels, scores = [], {}

    # if tops filtered, you can optionally mask children by top grouping (skipped here for simplicity)
    for i in order_lab:
        s = float(sim_lab[i])
        if s < req.threshold_child:
            continue
        labels.append(ARTS["labels"][i])
        scores[ARTS["labels"][i]] = round(s, 4)
        if len(labels) >= req.top_k_total:
            break

    providers = {}
    if req.return_providers:
        for lab in labels:
            p = PROVIDERS.get(lab)
            if p:
                providers[lab] = p

    return {
        "text": text,
        "top_categories": top_cats,
        "labels": labels,
        "scores": scores,
        "providers": providers if req.return_providers else None,
    }

def require_key(x_api_key: str | None):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized (bad X-API-Key)")

# -------- Routes (with aliases) --------
@app.get("/healthz")
@app.get("/api/healthz")
def healthz(x_api_key: str | None = Header(default=None, convert_underscores=False)):
    # Optional: allow health without key by commenting next line
    require_key(x_api_key)
    try:
        return {"ok": True, "labels": len(ARTS["labels"]), "tops": len(ARTS["tops"])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
@app.post("/api/predict")
def predict(req: PredictRequest, x_api_key: str | None = Header(default=None, convert_underscores=False)):
    require_key(x_api_key)
    if not req.texts:
        raise HTTPException(status_code=400, detail="texts is empty")
    out = [_predict_one(t, req) for t in req.texts]
    return {"predictions": out}

# Optional feedback endpoint
class FeedbackIn(BaseModel):
    text: str
    predicted: List[str] = []
    truth: List[str] = []
    extra: Dict[str, Any] | None = None

@app.post("/feedback")
@app.post("/api/feedback")
def feedback(item: FeedbackIn, x_api_key: str | None = Header(default=None, convert_underscores=False)):
    require_key(x_api_key)
    # you can persist; here we just echo
    return {"ok": True}
