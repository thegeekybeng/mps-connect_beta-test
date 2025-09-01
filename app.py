# app.py
import os, csv, json, time, uuid
from typing import List, Dict, Any, Optional
from collections import defaultdict

import numpy as np
import pandas as pd  # type: ignore
from fastapi import FastAPI, Header, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

from sentence_transformers import SentenceTransformer

# -----------------------
# Config & Security
# -----------------------
API_KEY = "mps-85-whampoa"  # << requested fixed key
MODEL_DIR = os.environ.get("MODEL_DIR", "./artifacts_zs_hier_plus")  # must exist
PROVIDERS_JSON = os.environ.get("PROVIDERS_JSON", "./providers_map.json")
FEEDBACK_PATH = os.environ.get("FEEDBACK_PATH", "./feedback/feedback.csv")


def require_key(x_api_key: Optional[str] = Header(None)):
    if not API_KEY:
        return
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="invalid api key")


# -----------------------
# App
# -----------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global ART
    ART = Artifacts(MODEL_DIR, PROVIDERS_JSON)
    os.makedirs(os.path.dirname(FEEDBACK_PATH) or ".", exist_ok=True)
    yield


app = FastAPI(
    title="WhiteVision Hierarchical ZS API", version="0.2.0", lifespan=lifespan
)

# CORS: open for testing; restrict to your deployed tester origins later.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------
# Artifacts Loader
# -----------------------
class Artifacts:
    def __init__(self, model_dir: str, providers_path: Optional[str]):
        self.model_dir = model_dir
        self.providers_path = providers_path

        # Load artifacts.json
        meta_path = os.path.join(model_dir, "artifacts.json")
        if not os.path.exists(meta_path):
            raise RuntimeError(f"artifacts.json not found in {model_dir}")
        with open(meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)

        # Model
        model_name = self.meta.get(
            "model_name", "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.model = SentenceTransformer(model_name)

        # Embeddings
        lab_emb_path = os.path.join(model_dir, "label_emb.npy")
        top_emb_path = os.path.join(model_dir, "top_emb.npy")
        if not os.path.exists(lab_emb_path) or not os.path.exists(top_emb_path):
            raise RuntimeError("label_emb.npy or top_emb.npy missing in model dir")

        self.label_emb = self._normalize(np.load(lab_emb_path))
        self.top_emb = self._normalize(np.load(top_emb_path))

        # Labels & Tops
        # Expect artifacts.json to have labels (list of slugs) and tops (list of slugs)
        self.labels: List[str] = self.meta.get("labels", [])
        self.tops: List[str] = self.meta.get("tops", [])

        if not self.labels:
            # fallback: attempt labels.csv
            labels_csv = os.path.join(model_dir, "labels.csv")
            if os.path.exists(labels_csv):
                df = pd.read_csv(labels_csv)
                if "label" in df.columns:
                    self.labels = df["label"].tolist()
                    self.tops = sorted(
                        set([l.split("/")[0] if "/" in l else l for l in self.labels])
                    )
            if not self.labels:
                raise RuntimeError(
                    "No labels found in artifacts.json and labels.csv missing"
                )

        if not self.tops:
            self.tops = sorted(
                set([l.split("/")[0] if "/" in l else l for l in self.labels])
            )

        # Map label -> top
        self.label_to_top: List[int] = self.meta.get("label_to_top", [])
        if not self.label_to_top or len(self.label_to_top) != len(self.labels):
            # derive by prefix
            top_index = {t: i for i, t in enumerate(self.tops)}
            derived = []
            for l in self.labels:
                t = l.split("/")[0] if "/" in l else l
                derived.append(top_index.get(t, 0))
            self.label_to_top = derived

        # Providers map (optional)
        self.providers: Dict[str, Any] = {}
        if self.providers_path and os.path.exists(self.providers_path):
            try:
                with open(self.providers_path, "r", encoding="utf-8") as f:
                    self.providers = json.load(f)
            except Exception:
                self.providers = {}

        # cache
        self.dim = self.label_emb.shape[1]
        if self.top_emb.shape[1] != self.dim:
            raise RuntimeError("top_emb and label_emb dimensions mismatch")

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        # Safely L2-normalize embeddings row-wise
        denom = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
        return x / denom

    def encode_text(self, text: str) -> np.ndarray:
        v = self.model.encode(text, normalize_embeddings=True)
        return v.cpu().numpy().astype(np.float32)


ART: Optional[Artifacts] = None


# -----------------------
# Schemas
# -----------------------
class PredictReq(BaseModel):
    texts: List[str]

    # thresholds & knobs
    threshold_top: float = 0.36
    threshold_child: float = 0.36
