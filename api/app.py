"""FastAPI app for hierarchical zero-shot text classification (testers API)."""

import os
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

import numpy as np
from fastapi import FastAPI, Depends, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore[import]
from sklearn.preprocessing import normalize  # type: ignore[import]

# ------------------ CONFIG ------------------
API_KEY = "mps-85-whampoa"  # header: X-API-Key
MODEL_DIR = os.environ.get("MODEL_DIR", "/app/artifacts_zs_hier_plus")
PROVIDERS_JSON = os.environ.get("PROVIDERS_JSON", "/app/providers_map.json")

# ------------------ APP ---------------------
app = FastAPI()

ALLOWED_ORIGINS = [
    "https://thegeekybeng.github.io",
    "https://thegeekybeng.github.io/mps-connect_testers",  # if you serve under a subpath
    "https://api.mpsconnect.thegeekybeng.com",  # optional: allows testing healthz in browser
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,  # keep False; you don't send cookies/auth
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


def require_key(x_api_key: Optional[str] = Header(None)):
    """Validate the request's X-API-Key header against the configured API key."""
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")


def _must_exist(p: str) -> str:
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    return p


def load_artifacts(
    model_dir: str,
) -> Tuple[List[str], List[str], np.ndarray, Dict[str, Any]]:
    """Load labels, tops, label embeddings, and metadata from MODEL_DIR."""
    model_dir = _must_exist(model_dir)
    art_json = _must_exist(os.path.join(model_dir, "artifacts.json"))
    with open(art_json, "r", encoding="utf-8") as f:
        artifacts = json.load(f)

    # embeddings file (first match wins)
    emb_path = None
    for cand in (
        "label_embeds.npy",
        "label_embeddings.npy",
        "labels_emb.npy",
        "embeddings.npy",
    ):
        p = os.path.join(model_dir, cand)
        if os.path.exists(p):
            emb_path = p
            break
    if emb_path is None:
        # fallback: any .npy file
        for fname in os.listdir(model_dir):
            if fname.endswith(".npy"):
                emb_path = os.path.join(model_dir, fname)
                break
    if emb_path is None:
        raise FileNotFoundError("No .npy embeddings file found in MODEL_DIR")

    label_embeds = np.load(emb_path)
    if label_embeds.ndim != 2:
        raise RuntimeError(f"Embeddings shape invalid: {label_embeds.shape}")

    labels = artifacts.get("labels") or artifacts.get("label_list")
    if not labels or len(labels) != label_embeds.shape[0]:
        raise RuntimeError("Labels list missing or length mismatch with embeddings")

    tops = artifacts.get("tops") or artifacts.get("top_categories") or []
    meta = {
        "embedding_model": artifacts.get("embedding_model"),
        "embeddings_file": os.path.basename(emb_path),
        "artifact_file": "artifacts.json",
    }
    return labels, tops, label_embeds, meta


def load_providers_map(path: str) -> Dict[str, Any]:
    """Load providers mapping JSON from the given path; return {} if missing."""
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# 1) Define a container for state (near the top)
@dataclass
class VectorSpace:
    """Holds vectorizer, TF-IDF matrices, and projection weights."""

    vect: TfidfVectorizer | None
    label_tfidf: Any
    label_tfidf_unit: Any
    w_cache: np.ndarray | None


@dataclass
class ModelState:
    """Application-wide model/cache state stored on app.state."""

    labels: List[str]
    tops: List[str]
    label_emb: np.ndarray
    label_emb_unit: np.ndarray
    providers: Dict[str, Any]
    meta: Dict[str, Any]
    vs: VectorSpace


# ------------------ GLOBALS ------------------
# _VECT: Optional[TfidfVectorizer] = None  # TF-IDF vectorizer over labels
# _LABEL_TFIDF = None  # TF-IDF matrix for labels
# _LABEL_TFIDF_UNIT = None  # normalized label TF-IDF for cosine fallback
# _W_CACHE: Optional[np.ndarray] = None  # ridge projection weights (V -> D)


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return x / n


def _normalize_tfidf(a_matrix):
    rn = np.sqrt(a_matrix.multiply(a_matrix).sum(axis=1)).A1 + 1e-12
    return a_matrix.multiply(1.0 / rn[:, None])


def _fit_projection(label_tfidf, label_emb_unit):
    try:
        at = label_tfidf.T.astype(np.float32)
        a_mat = label_tfidf.astype(np.float32)
        e = label_emb_unit.astype(np.float32)
        lam = 1e-2
        w_cache = np.linalg.solve(
            (at @ a_mat) + lam * np.eye(at.shape[0], dtype=np.float32), at @ e
        )
    except (ValueError, np.linalg.LinAlgError, MemoryError, TypeError):
        w_cache = None
    return w_cache


# 2) In startup: populate app.state instead of globals
@app.on_event("startup")
def _startup():
    labels, tops, label_emb, meta = load_artifacts(MODEL_DIR)
    label_emb_unit = _normalize_rows(label_emb.astype(np.float32))
    vect = TfidfVectorizer(min_df=1, max_df=1.0, ngram_range=(1, 2))
    label_tfidf = vect.fit_transform(labels)
    label_tfidf_unit = _normalize_tfidf(label_tfidf)
    w_cache = _fit_projection(label_tfidf, label_emb_unit)
    vs = VectorSpace(
        vect=vect,
        label_tfidf=label_tfidf,
        label_tfidf_unit=label_tfidf_unit,
        w_cache=w_cache,
    )
    app.state.ms = ModelState(
        labels=labels,
        tops=tops,
        label_emb=label_emb,
        label_emb_unit=label_emb_unit,
        providers=load_providers_map(PROVIDERS_JSON),
        meta=meta,
        vs=vs,
    )


# 3) Replace global uses with app.state.ms (example in helpers)
def _embed_texts(texts: List[str]) -> np.ndarray:
    ms = app.state.ms
    if ms.vs.vect is None:
        raise RuntimeError("Vectorizer not initialized")
    x_matrix = ms.vs.vect.transform(texts)
    if ms.vs.w_cache is not None:
        z_matrix = (x_matrix @ ms.vs.w_cache).astype(np.float32)
        z_matrix = z_matrix / np.clip(
            np.linalg.norm(z_matrix, axis=1, keepdims=True), 1e-12, None
        )
        return z_matrix
    z_matrix = normalize(x_matrix, norm="l2", axis=1)
    if hasattr(z_matrix, "toarray"):
        return z_matrix.toarray()  # type: ignore[attr-defined]
    return np.asarray(z_matrix)


def score_texts(texts: List[str]) -> np.ndarray:
    """Compute label scores for input texts using embedding or TF-IDF cosine."""
    z_matrix = _embed_texts(texts)  # (n, D) or (n, V)
    if z_matrix.shape[1] == app.state.ms.label_emb_unit.shape[1]:
        scores = z_matrix @ app.state.ms.label_emb_unit.T  # (n, L)
        return scores.astype(np.float32)

    # de-indented (no else)
    x_tfidf = app.state.ms.vs.vect.transform(texts)
    rn = np.sqrt(x_tfidf.multiply(x_tfidf).sum(axis=1)).A1 + 1e-12
    x_tfidf = x_tfidf.multiply(1.0 / rn[:, None])
    scores = (x_tfidf @ app.state.ms.vs.label_tfidf_unit.T).toarray()
    return scores.astype(np.float32)


class PredictIn(BaseModel):
    """Request model for prediction parameters and input texts."""

    texts: List[str]
    threshold_top: float = 0.30
    threshold_child: float = 0.36
    top_k_top: int = 6
    top_k_child: int = 6
    top_k_total: int = 18
    seed_prior_threshold: float = 0.45  # reserved for future priors
    use_priors: bool = True
    return_providers: bool = True


def _top_of(label: str) -> str:
    return label.split("/", 1)[0] if "/" in label else label


def _select_labels(scores: np.ndarray, params: PredictIn):
    """
    Selection strategy:
      1) Gate tops by threshold_top (best child per top).
      2) Within each kept top, keep children that pass:
           - absolute threshold_child OR
           - relative: score >= rel_child * (top's max child score)
      3) Ensure at least min_labels by relaxing relative cutoff gradually.
      4) Fallback to global top_k_total if still empty.
    """
    rel_child = 0.85
    min_labels = 4
    relax_steps = [0.9, 0.8, 0.7, 0.6]

    items = [
        (app.state.ms.labels[i], float(scores[i]))
        for i in range(len(app.state.ms.labels))
    ]

    def select_once(th_top: float, th_child: float, rel: float):
        # best score per top
        top_best: Dict[str, float] = {}
        for lab, sc in items:
            t = _top_of(lab)
            top_best[t] = max(top_best.get(t, 0.0), sc)

        tops_ranked = sorted(
            [(t, s) for t, s in top_best.items() if s >= th_top],
            key=lambda x: x[1],
            reverse=True,
        )[: params.top_k_top]

        chosen = []
        for t, _ in tops_ranked:
            max_child = max((sc for lab, sc in items if _top_of(lab) == t), default=0.0)
            rel_cut = rel * max_child
            children = [
                (lab, sc)
                for (lab, sc) in items
                if _top_of(lab) == t and (sc >= th_child or sc >= rel_cut)
            ]
            children = sorted(children, key=lambda x: x[1], reverse=True)[
                : params.top_k_child
            ]
            for lab, sc in children:
                chosen.append({"label": lab, "score": sc})

        chosen = sorted(chosen, key=lambda x: x["score"], reverse=True)[
            : params.top_k_total
        ]
        return chosen, tops_ranked

    chosen, tops_ranked = select_once(
        params.threshold_top, params.threshold_child, rel_child
    )
    for r in relax_steps:
        if len(chosen) >= min_labels:
            break
        chosen, tops_ranked = select_once(
            params.threshold_top, params.threshold_child, r
        )

    if not chosen:
        topk = sorted(items, key=lambda x: x[1], reverse=True)[
            : max(1, params.top_k_total)
        ]
        chosen = [{"label": lab, "score": sc} for lab, sc in topk]
        tsc: Dict[str, float] = {}
        for lab, sc in topk:
            tt = _top_of(lab)
            tsc[tt] = max(tsc.get(tt, 0.0), sc)
        tops_ranked = sorted(tsc.items(), key=lambda x: x[1], reverse=True)[
            : params.top_k_top
        ]

    return chosen, tops_ranked


def mount_routes(prefix: str = ""):
    """Mount API endpoints with an optional URL prefix."""
    pfx = prefix

    @app.get(pfx + "/healthz")
    def healthz():
        ms = app.state.ms
        return {
            "status": "ok",
            "labels": len(ms.labels),
            "tops": len(set(map(_top_of, ms.labels))),
            "model_dir": MODEL_DIR,
            "providers_json": os.path.exists(PROVIDERS_JSON),
            "meta": ms.meta,
        }

    @app.get(pfx + "/labels")
    def labels():
        ms = app.state.ms
        return {"labels": ms.labels, "tops": list(sorted(set(map(_top_of, ms.labels))))}

    @app.post(pfx + "/predict", dependencies=[Depends(require_key)])
    def predict(pi: PredictIn, _request: Request):
        if not pi.texts:
            raise HTTPException(status_code=400, detail="texts is empty")
        scores = score_texts(pi.texts)
        out = []
        for i, text in enumerate(pi.texts):
            chosen, tops_ranked = _select_labels(scores[i], pi)
            labels = [c["label"] for c in chosen]
            score_map = {c["label"]: c["score"] for c in chosen}
            providers = {}
            if pi.return_providers:
                for lab in labels:
                    providers[lab] = app.state.ms.vs.providers.get(lab) or {}
            out.append(
                {
                    "text": text,
                    "top_categories": [{"top": t, "score": s} for t, s in tops_ranked],
                    "labels": labels,
                    "scores": score_map,
                    "providers": providers,
                }
            )
        return {"predictions": out}


# mount both root and /api for convenience
mount_routes("")
mount_routes("/api")

# Optional: run directly
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
