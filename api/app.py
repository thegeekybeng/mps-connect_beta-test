import os
import json
import csv
from pathlib import Path
from typing import List, Dict, Optional, Any

import numpy as np
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# -----------------------
# Config
# -----------------------
API_KEY = "mps-85-whampoa"
MODEL_DIR = Path(os.getenv("MODEL_DIR", "/app/artifacts_zs_hier_plus"))
PROVIDERS_JSON = Path(os.getenv("PROVIDERS_JSON", "/app/providers_map.json"))
FEEDBACK_PATH = Path(os.getenv("FEEDBACK_PATH", "/app/feedback/feedback.csv"))

SENTENCE_MODEL_NAME = os.getenv(
    "SENTENCE_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
)

# -----------------------
# App + CORS (wide open for beta)
# -----------------------
app = FastAPI(title="WhiteVision Hierarchical ZS API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],            # tighten later (e.g., your Pages domain)
    allow_methods=["*"],
    allow_headers=["*"],            # important for X-API-Key preflight
    expose_headers=["*"],
)

# -----------------------
# Model / Artifacts loader
# -----------------------
_sentence_model = None
_label_slugs: List[str] = []
_label_tops: List[str] = []                 # top category for each label (same idx)
_label_vecs: Optional[np.ndarray] = None
_top_unique: List[str] = []
_providers: Dict[str, Any] = {}

def _lazy_sentence_model():
    global _sentence_model
    if _sentence_model is None:
        from sentence_transformers import SentenceTransformer
        _sentence_model = SentenceTransformer(SENTENCE_MODEL_NAME)
    return _sentence_model

def _norm(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9
    return x / denom

def _infer_label_tops_from_slugs(slugs: List[str]) -> List[str]:
    # "housing/hdb_loan_arrears" -> "housing"
    return [s.split("/", 1)[0] if "/" in s else s for s in slugs]

def _load_artifacts() -> None:
    """Be robust to different artifact layouts produced by your build script."""
    global _label_slugs, _label_tops, _label_vecs, _top_unique, _providers

    # providers map (optional)
    if PROVIDERS_JSON.exists():
        try:
            _providers = json.loads(PROVIDERS_JSON.read_text())
        except Exception:
            _providers = {}
    else:
        _providers = {}

    labels: List[str] = []
    vecs: Optional[np.ndarray] = None

    art_json = MODEL_DIR / "artifacts.json"
    labels_csv = None
    for p in [MODEL_DIR / "labels_sg_gov_hier_plus.csv",
              MODEL_DIR / "labels_sg_gov_hier.csv",
              MODEL_DIR / "labels.csv"]:
        if p.exists():
            labels_csv = p
            break

    # 1) labels and vectors from artifacts.json + label_embeds.npy (preferred)
    if art_json.exists() and (MODEL_DIR / "label_embeds.npy").exists():
        meta = json.loads(art_json.read_text())
        # meta may use different keys; try a few
        labels = meta.get("labels") or meta.get("label_strings") or meta.get("label_slugs") or []
        vecs = np.load(MODEL_DIR / "label_embeds.npy")
        if not labels or vecs is None:
            raise RuntimeError("artifacts.json / label_embeds.npy present but invalid")

    # 2) else, fall back to CSV (label, description) and embed descriptions
    elif labels_csv and labels_csv.exists():
        import pandas as pd
        df = pd.read_csv(labels_csv)
        # expect columns: label, description
        if "label" not in df.columns:
            raise RuntimeError(f"{labels_csv} missing 'label' column")
        labels = df["label"].astype(str).tolist()
        texts = df["description"].astype(str).tolist() if "description" in df.columns else labels
        m = _lazy_sentence_model()
        vecs = _norm(m.encode(texts, convert_to_numpy=True, batch_size=256, show_progress_bar=False))
    else:
        # Last resort: tiny static set to keep API alive
        labels = [
            "social_support/comcare_short_mid_term",
            "employment/career_services_e2i",
            "utilities_comms/electricity_spgroup",
            "housing/town_council_scc_arrears",
            "housing/hdb_loan_arrears",
        ]
        m = _lazy_sentence_model()
        vecs = _norm(m.encode(labels, convert_to_numpy=True, batch_size=64, show_progress_bar=False))

    _label_slugs = labels
    _label_tops = _infer_label_tops_from_slugs(labels)
    _top_unique = sorted(list(dict.fromkeys(_label_tops)))  # stable unique
    _label_vecs = _norm(vecs.astype(np.float32))


def _score_texts(texts: List[str]) -> np.ndarray:
    """Cosine similarity against label vectors."""
    m = _lazy_sentence_model()
    X = _norm(m.encode(texts, convert_to_numpy=True, batch_size=64, show_progress_bar=False)).astype(np.float32)
    return X @ _label_vecs.T  # (B, L)


# Load at startup
_load_artifacts()

# -----------------------
# Schemas
# -----------------------
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

# -----------------------
# Helpers
# -----------------------
def _select_labels_for_row(scores: np.ndarray, p: PredictIn) -> Dict[str, Any]:
    """Hierarchical top->child selection from a score vector over labels."""
    # per-top max score
    top_to_max = {}
    for lbl, top, s in zip(_label_slugs, _label_tops, scores.tolist()):
        cur = top_to_max.get(top, -1.0)
        if s > cur:
            top_to_max[top] = s

    # choose tops
    tops_ranked = sorted(top_to_max.items(), key=lambda x: x[1], reverse=True)
    chosen_tops = [(t, sc) for t, sc in tops_ranked if sc >= p.threshold_top][: p.top_k_top]

    # within each top, choose labels
    chosen_labels = []
    for t, _ in chosen_tops:
        # collect labels of that top
        idxs = [i for i, tp in enumerate(_label_tops) if tp == t]
        # sort those labels by score
        labels_sc = sorted([( _label_slugs[i], scores[i]) for i in idxs],
                           key=lambda x: x[1], reverse=True)
        # threshold + cap
        keep = [(lab, sc) for lab, sc in labels_sc if sc >= p.threshold_child][: p.top_k_child]
        chosen_labels.extend(keep)

    # global cap
    chosen_labels = chosen_labels[: p.top_k_total]

    # package
    out_labels = [lab for lab, _ in chosen_labels]
    out_scores = {lab: float(sc) for lab, sc in chosen_labels}
    out_tops = [{"top": t, "score": float(sc)} for t, sc in chosen_tops]
    # optional providers
    prov = {}
    if out_labels and p.return_providers and _providers:
        for lab in out_labels:
            if lab in _providers:
                prov[lab] = _providers[lab]
    return {"labels": out_labels, "scores": out_scores, "top_categories": out_tops, "providers": prov}


def _require_api_key(key: Optional[str]):
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


# -----------------------
# Routes
# -----------------------
@app.get("/healthz")
def healthz():
    return {
        "ok": True,
        "labels": len(_label_slugs),
        "tops": len(_top_unique),
        "model_dir": str(MODEL_DIR),
        "providers_loaded": bool(_providers),
    }

@app.post("/predict")
def predict(body: PredictIn, x_api_key: Optional[str] = Header(None)):
    _require_api_key(x_api_key)
    if not body.texts:
        return {"predictions": []}
    S = _score_texts(body.texts)  # (B, L)
    preds = []
    for i, txt in enumerate(body.texts):
        sel = _select_labels_for_row(S[i], body)
        sel["text"] = txt
        preds.append(sel)
    return {"predictions": preds}


class FeedbackIn(BaseModel):
    text: str
    predicted: List[str]
    truth: List[str]
    meta: Dict[str, Any] = {}

@app.post("/feedback")
def feedback(body: FeedbackIn, x_api_key: Optional[str] = Header(None)):
    _require_api_key(x_api_key)
    FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    is_new = not FEEDBACK_PATH.exists()
    with FEEDBACK_PATH.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(["text", "predicted", "truth", "meta_json"])
        w.writerow([
            body.text,
            "|".join(body.predicted),
            "|".join(body.truth),
            json.dumps(body.meta, ensure_ascii=False),
        ])
    return {"ok": True, "saved_to": str(FEEDBACK_PATH)}
