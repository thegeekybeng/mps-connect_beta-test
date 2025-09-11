"""FastAPI app for hierarchical zero-shot text classification (testers API)."""

import os
import sys
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
import time
from dataclasses import dataclass
import threading

# pylint: disable=import-error
import numpy as np
from fastapi import FastAPI, Depends, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel  # type: ignore[attr-defined]

# Security imports (graceful fallbacks if not present)
# pylint: disable=import-error
try:
    from security.middleware import (  # type: ignore
        SecurityMiddleware,
        CORSecurityMiddleware,
        ContentSecurityMiddleware,
    )
except Exception:  # pragma: no cover

    class SecurityMiddleware:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass

    class CORSecurityMiddleware:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass

    class ContentSecurityMiddleware:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass


# pylint: disable=import-error
from database.connection import get_db  # type: ignore
from sqlalchemy.orm import Session  # type: ignore
from security.audit import log_api_access  # type: ignore

# Simple in-memory rate limit (per IP, per endpoint)
_RATE_BUCKETS: Dict[str, Tuple[float, int]] = {}
_RATE_WINDOW_SECONDS = 60.0
_RATE_MAX_REQUESTS = 60


def _rate_check(ip: Optional[str], endpoint: str) -> bool:
    import time

    if not ip:
        return True
    key = f"{endpoint}:{ip}"
    now = time.time()
    window_start, count = _RATE_BUCKETS.get(key, (now, 0))
    if now - window_start > _RATE_WINDOW_SECONDS:
        _RATE_BUCKETS[key] = (now, 1)
        return True
    count += 1
    _RATE_BUCKETS[key] = (window_start, count)
    return count <= _RATE_MAX_REQUESTS


def _is_staff(req: Request) -> bool:
    """Heuristic staff check without enforcing auth.
    Accepts either X-User-Role: mp_staff/admin or X-Staff-Mode: true.
    If absent, treat as citizen (no confidences).
    """
    role = (req.headers.get("X-User-Role") or "").lower()
    if role in ("mp_staff", "admin"):
        return True
    staff_mode = (req.headers.get("X-Staff-Mode") or "").lower() in ("1", "true", "yes")
    return staff_mode


# Gemini AI Integration
try:
    from api.gemini_integration import (
        get_gemini_integration,
        CaseAnalysis,
        LetterDraft,
        ApprovalRecommendation,
    )

    GEMINI_AVAILABLE = True
except Exception:  # noqa: BLE001
    GEMINI_AVAILABLE = False
    CaseAnalysis = None
    LetterDraft = None
    ApprovalRecommendation = None

# AI Governance imports (tolerate missing optional packages at runtime)
# Prefer the api.* packages where available; fall back to stubs if not.
try:  # Explainability
    from api.explainability import MPSExplainabilityEngine  # type: ignore
except Exception:  # noqa: BLE001

    class MPSExplainabilityEngine:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass


try:  # Transparency
    from api.transparency import TransparencyEngine  # type: ignore
except Exception:  # noqa: BLE001

    class TransparencyEngine:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass


try:  # Governance (optional)
    from api.governance import GovernanceEngine  # type: ignore
except Exception:  # noqa: BLE001
    try:
        from governance import GovernanceEngine  # type: ignore
    except Exception:  # noqa: BLE001

        class GovernanceEngine:  # type: ignore
            def __init__(self, *args, **kwargs):
                pass


try:  # Immutable storage
    from api.immutable import ImmutableStorage  # type: ignore
except Exception:  # noqa: BLE001

    class ImmutableStorage:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass


try:  # Audit
    from api.audit import AuditLogger  # type: ignore
except Exception:  # noqa: BLE001

    class AuditLogger:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass


try:  # Security manager
    from security import SecurityManager  # type: ignore
except Exception:  # noqa: BLE001

    class SecurityManager:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass


# pylint: disable=import-error

# sklearn imports (grouped)
# pylint: disable=import-error
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore[import]
from sklearn.metrics import pairwise as sk_pairwise  # type: ignore[import]
from sklearn.preprocessing import normalize  # type: ignore[import]

# Avoid importing heavy ML libraries at module import time.
# They are lazily imported inside functions when needed.

# (sklearn imported above — keep a single alias: sk_pairwise)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger(__name__)

# Fix uvicorn logging
uvicorn_logger = logging.getLogger("uvicorn")
uvicorn_logger.setLevel(logging.INFO)
uvicorn_access_logger = logging.getLogger("uvicorn.access")
uvicorn_access_logger.setLevel(logging.INFO)

# ------------------ CONFIG ------------------
API_KEY = os.environ.get("API_KEY", "mps-85-whampoa")  # header: X-API-Key
MODEL_DIR = os.environ.get("MODEL_DIR", "./artifacts_zs_hier_plus")
PROVIDERS_JSON = os.environ.get("PROVIDERS_JSON", "./providers_map.json")

# ------------------ APP ---------------------
app = FastAPI()

# Initialize AI Governance modules
explainability_engine = MPSExplainabilityEngine()
transparency_engine = TransparencyEngine()
governance_engine = GovernanceEngine()
immutable_storage = ImmutableStorage()
audit_logger = AuditLogger()
security_manager = SecurityManager()


def _parse_cors(origins_env: Optional[str]) -> list[str]:
    if not origins_env:
        return []
    return [o.strip() for o in origins_env.split(",") if o.strip()]


DEFAULT_ALLOWED_ORIGINS = [
    "http://localhost:8080",
    "http://localhost:3000",
    "http://localhost:5173",
    "https://mps-connect-web-987575541268.asia-southeast1.run.app",
]

# Allow overriding via CORS_ORIGINS env (comma-separated)
ALLOWED_ORIGINS_ENV = os.environ.get("CORS_ORIGINS")
ALLOWED_ORIGINS = _parse_cors(ALLOWED_ORIGINS_ENV) or DEFAULT_ALLOWED_ORIGINS

# Add security middleware
# app.add_middleware(SecurityMiddleware, db_session_factory=get_db)
# app.add_middleware(CORSecurityMiddleware, allowed_origins=ALLOWED_ORIGINS)
# app.add_middleware(ContentSecurityMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
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

    # embeddings file (prioritize label_emb.npy)
    emb_path = None
    for cand in (
        "label_emb.npy",
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
        # fallback: any .npy file except top_emb.npy
        for fname in os.listdir(model_dir):
            if fname.endswith(".npy") and fname != "top_emb.npy":
                emb_path = os.path.join(model_dir, fname)
                break
    if emb_path is None:
        raise FileNotFoundError("No .npy embeddings file found in MODEL_DIR")

    label_embeds = np.load(emb_path)
    if label_embeds.ndim != 2:
        raise RuntimeError("Embeddings shape invalid: %s" % (label_embeds.shape,))

    labels = artifacts.get("labels") or artifacts.get("label_list")
    if not labels or len(labels) != label_embeds.shape[0]:
        msg = (
            "Labels list missing or length mismatch with embeddings: labels=%s, "
            "embeddings=%s, file=%s"
        )
        raise RuntimeError(
            msg % (len(labels) if labels else 0, label_embeds.shape[0], emb_path)
        )
    # Remove duplicate top categories
    top_categories = artifacts.get("tops") or artifacts.get("top_categories") or []
    tops = list(set(top_categories))  # Remove duplicate top categories  (just in case)
    print("Tops: %s" % (tops,))  # for debugging
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
def _initialize_model_state():
    """Initialize model state - can be called during startup or for testing."""
    try:
        logger.info("Loading artifacts from: %s", MODEL_DIR)

        labels, tops, label_emb, meta = load_artifacts(MODEL_DIR)
        logger.info("Loaded %d labels and %d top categories", len(labels), len(tops))

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

        providers = load_providers_map(PROVIDERS_JSON)
        logger.info("Loaded %d provider mappings", len(providers))

        return ModelState(
            labels=labels,
            tops=tops,
            label_emb=label_emb,
            label_emb_unit=label_emb_unit,
            providers=providers,
            meta=meta,
            vs=vs,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        logger.error("Model state initialization failed: %s", e)
        raise


@app.on_event("startup")
def _startup():
    """Start quickly and warm the model in the background.

    Cloud Run expects the container to bind to $PORT promptly. Heavy
    initialization in the startup hook can delay binding and cause deployment
    to fail. We therefore spawn a background thread to load artifacts while the
    server is already listening. First requests will still ensure the model is
    initialized via _ensure_model_state().
    """
    app.state.ms = None  # type: ignore[assignment]

    def _bg_warm():
        try:
            logger.info("[warmup] Initializing model state in background…")
            app.state.ms = _initialize_model_state()
            logger.info("[warmup] Model state ready")
        except Exception as e:  # noqa: BLE001
            logger.error("[warmup] Initialization failed: %s", e)

    threading.Thread(target=_bg_warm, daemon=True).start()


# 3) Replace global uses with app.state.ms (example in helpers)
def _ensure_model_state():
    """Ensure model state is initialized (for testing)."""
    if not hasattr(app.state, "ms") or app.state.ms is None:
        app.state.ms = _initialize_model_state()


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
    """Compute label scores for input texts using natural language similarity."""
    _ensure_model_state()
    try:
        # Lazy import to avoid heavy startup cost
        from sentence_transformers import SentenceTransformer  # type: ignore

        # Load model with proper device handling
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        # Create natural language descriptions for labels
        label_descriptions = []
        for label in app.state.ms.labels:
            parts = label.split("/")
            if len(parts) == 2:
                category, subcategory = parts
                # Convert to natural language
                category_nl = category.replace("_", " ").title()
                subcategory_nl = subcategory.replace("_", " ").title()
                description = f"{category_nl} related to {subcategory_nl}"
            else:
                description = label.replace("_", " ").title()
            label_descriptions.append(description)

        # Encode texts and labels
        text_embeddings = model.encode(texts, convert_to_tensor=False)
        label_embeddings = model.encode(label_descriptions, convert_to_tensor=False)

        # Compute cosine similarity (prefer sklearn, fall back to numpy)
        if sk_pairwise is not None:
            scores = sk_pairwise.cosine_similarity(text_embeddings, label_embeddings)  # type: ignore[call-arg]
        else:
            te = np.asarray(text_embeddings, dtype=np.float32)
            le = np.asarray(label_embeddings, dtype=np.float32)
            te /= np.linalg.norm(te, axis=1, keepdims=True) + 1e-12
            le /= np.linalg.norm(le, axis=1, keepdims=True) + 1e-12
            scores = te @ le.T

        logger.info(
            "Natural language scores shape: %s, max score: %s",
            scores.shape,
            scores.max(),
        )
        return scores.astype(np.float32)

    except (ImportError, OSError, RuntimeError, ValueError) as e:
        logger.warning(
            "Natural language processing failed, fallback to keywords: %s", e
        )
        # Simple keyword-based fallback
        return _simple_keyword_scoring(texts)


def _simple_keyword_scoring(texts: List[str]) -> np.ndarray:
    """Simple keyword-based scoring as fallback."""
    _ensure_model_state()

    scores = np.zeros((len(texts), len(app.state.ms.labels)), dtype=np.float32)

    for i, text in enumerate(texts):
        text_lower = text.lower()

        for j, label in enumerate(app.state.ms.labels):
            parts = label.split("/")
            if len(parts) == 2:
                category, subcategory = parts

                # Simple keyword matching
                category_keywords = category.replace("_", " ").split()
                subcategory_keywords = subcategory.replace("_", " ").split()

                score = 0.0
                for keyword in category_keywords + subcategory_keywords:
                    if keyword in text_lower:
                        score += 0.1

                # Boost for exact matches
                if category.replace("_", " ") in text_lower:
                    score += 0.3
                if subcategory.replace("_", " ") in text_lower:
                    score += 0.5

                scores[i, j] = min(score, 1.0)

    logger.info(
        "Keyword fallback scores shape: %s, max score: %s",
        scores.shape,
        scores.max(),
    )
    return scores


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
    return_confidence_breakdown: bool = True


class ConfidenceBreakdown(BaseModel):
    """Confidence breakdown for auditing and approval workflows."""

    label: str
    confidence_score: float
    category: str
    subcategory: str
    semantic_similarity: float
    provider_info: Dict[str, Any]
    approval_recommendation: str  # "auto_approve", "manual_review", "reject"
    risk_level: str  # "low", "medium", "high"


# LLM-Guided Chat Models
class ChatMessage(BaseModel):
    """Individual chat message."""

    role: str  # "user" or "assistant"
    content: str
    timestamp: str
    language: Optional[str] = None


class ChatSession(BaseModel):
    """Chat session data."""

    session_id: str
    messages: List[ChatMessage]
    context: Dict[str, Any]
    language: str = "en"
    confidence: float = 0.0
    facts_extracted: List[Dict[str, Any]] = []
    suggested_agencies: List[Dict[str, Any]] = []


class ChatRequest(BaseModel):
    """Request for chat interaction."""

    session_id: str
    message: str
    language: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


class ChatResponse(BaseModel):
    """Response from chat system."""

    success: bool
    message: str
    language: str
    confidence: float
    facts_extracted: List[Dict[str, Any]]
    suggested_agencies: List[Dict[str, Any]]
    next_question: Optional[str] = None
    is_complete: bool = False
    archival_english: Optional[str] = None
    missing_facts: List[str] = []
    error: Optional[str] = None


class FactChecklistRequest(BaseModel):
    """Request for fact checklist review."""

    session_id: str
    facts: List[Dict[str, Any]]
    corrections: Optional[Dict[str, str]] = None


class FactChecklistResponse(BaseModel):
    """Response for fact checklist."""

    success: bool
    reviewed_facts: List[Dict[str, Any]]
    confidence: float
    ready_for_letter: bool
    missing_facts: List[str] = []
    error: Optional[str] = None


# Gemini AI Integration Models
class CaseAnalysisRequest(BaseModel):
    """Request model for case analysis."""

    case_text: str
    feedback: Optional[str] = None


class LetterGenerationRequest(BaseModel):
    """Request model for letter generation."""

    case_data: Dict[str, Any]
    feedback: Optional[str] = None


class ApprovalRecommendationRequest(BaseModel):
    """Request model for approval recommendations."""

    case_analysis: Dict[str, Any]
    feedback: Optional[str] = None


class GeminiResponse(BaseModel):
    """Generic response model for Gemini operations."""

    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    model_used: str  # "flash" or "pro"


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

        # Ensure tops_ranked is populated from chosen items
        tsc: Dict[str, float] = {}
        for lab, sc in topk:
            tt = _top_of(lab)
            tsc[tt] = max(tsc.get(tt, 0.0), sc)
        tops_ranked = sorted(tsc.items(), key=lambda x: x[1], reverse=True)[
            : params.top_k_top
        ]

        # If still empty, create from all available tops
        if not tops_ranked:
            all_tops: Dict[str, float] = {}
            for lab, sc in items:
                tt = _top_of(lab)
                all_tops[tt] = max(all_tops.get(tt, 0.0), sc)
            tops_ranked = sorted(all_tops.items(), key=lambda x: x[1], reverse=True)[
                : params.top_k_top
            ]

    return chosen, tops_ranked


def mount_routes(prefix: str = ""):
    """Mount API endpoints with an optional URL prefix."""
    pfx = prefix

    @app.get(pfx + "/healthz")
    def healthz():
        ms = getattr(app.state, "ms", None)
        if ms is None:
            return {
                "status": "warming",
                "labels": 0,
                "tops": 0,
                "model_dir": MODEL_DIR,
                "providers_json": os.path.exists(PROVIDERS_JSON),
                "meta": {},
            }
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
        confidence_breakdowns = []

        for i, text in enumerate(pi.texts):
            chosen, tops_ranked = _select_labels(scores[i], pi)
            labels = [c["label"] for c in chosen]
            score_map = {c["label"]: c["score"] for c in chosen}
            providers = {}
            if pi.return_providers:
                for lab in labels:
                    providers[lab] = app.state.ms.providers.get(lab) or {}

            # Generate confidence breakdown for auditing
            text_confidence_breakdown = []
            if pi.return_confidence_breakdown:
                for c in chosen:
                    label = c["label"]
                    score = c["score"]
                    category = _top_of(label)
                    subcategory = label.split("/", 1)[1] if "/" in label else ""

                    # Determine approval recommendation based on confidence
                    if score >= 0.7:
                        approval_rec = "auto_approve"
                        risk_level = "low"
                    elif score >= 0.4:
                        approval_rec = "manual_review"
                        risk_level = "medium"
                    else:
                        approval_rec = "reject"
                        risk_level = "high"

                    text_confidence_breakdown.append(
                        ConfidenceBreakdown(
                            label=label,
                            confidence_score=score,
                            category=category,
                            subcategory=subcategory,
                            semantic_similarity=score,
                            provider_info=providers.get(label, {}),
                            approval_recommendation=approval_rec,
                            risk_level=risk_level,
                        )
                    )

            out.append(
                {
                    "text": text,
                    "top_categories": [{"top": t, "score": s} for t, s in tops_ranked],
                    "labels": labels,
                    "scores": score_map,
                    "providers": providers,
                }
            )
            confidence_breakdowns.append(text_confidence_breakdown)

        result = {"predictions": out}
        if pi.return_confidence_breakdown:
            result["confidence_breakdown"] = confidence_breakdowns  # type: ignore[assignment]
        return result


# LLM-Guided Chat Endpoints
@app.post("/api/chat/start", dependencies=[Depends(require_key)])
async def start_chat_session(
    request: ChatRequest, http_req: Request, db: Session = Depends(get_db)
):
    """Start a new LLM-guided chat session."""
    if not GEMINI_AVAILABLE:
        raise HTTPException(status_code=503, detail="Gemini AI not available")

    gemini = get_gemini_integration()
    if not gemini:
        raise HTTPException(
            status_code=503, detail="Gemini integration not initialized"
        )

    start_time = time.monotonic()
    if not _rate_check(
        http_req.client.host if http_req.client else None, str(http_req.url.path)
    ):
        raise HTTPException(status_code=429, detail="Too Many Requests")
    if not _rate_check(
        http_req.client.host if http_req.client else None, str(http_req.url.path)
    ):
        raise HTTPException(status_code=429, detail="Too Many Requests")
    if not _rate_check(
        http_req.client.host if http_req.client else None, str(http_req.url.path)
    ):
        raise HTTPException(status_code=429, detail="Too Many Requests")
    # Basic per-IP rate limiting
    if not _rate_check(
        http_req.client.host if http_req.client else None, str(http_req.url.path)
    ):
        raise HTTPException(status_code=429, detail="Too Many Requests")
    try:
        # Initialize chat session with LLM
        response = await gemini.start_guided_chat(
            session_id=request.session_id,
            initial_message=request.message,
            language=request.language or "en",
            context=request.context or {},
        )
        # Role-based redaction: hide confidences for citizens
        if not _is_staff(http_req):
            response = dict(response)
            response["confidence"] = 0.0
        resp_model = ChatResponse(**response)
        try:
            log_api_access(
                db=db,
                user_id=None,
                endpoint=str(http_req.url.path),
                method=http_req.method,
                status_code=200 if resp_model.success else 500,
                response_time_ms=int((time.monotonic() - start_time) * 1000),
                ip_address=http_req.client.host if http_req.client else None,
                user_agent=http_req.headers.get("user-agent"),
            )
        except Exception:
            pass
        return resp_model
    except Exception as e:
        logger.error(f"Error starting chat session: {e}")
        resp_model = ChatResponse(
            success=False,
            message="Sorry, I couldn't start the chat session. Please try again.",
            language=request.language or "en",
            confidence=0.0,
            facts_extracted=[],
            suggested_agencies=[],
            error=str(e),
        )
        try:
            log_api_access(
                db=db,
                user_id=None,
                endpoint=str(http_req.url.path),
                method=http_req.method,
                status_code=500,
                response_time_ms=int((time.monotonic() - start_time) * 1000),
                ip_address=http_req.client.host if http_req.client else None,
                user_agent=http_req.headers.get("user-agent"),
            )
        except Exception:
            pass
        return resp_model


@app.post("/api/chat/continue", dependencies=[Depends(require_key)])
async def continue_chat(
    request: ChatRequest, http_req: Request, db: Session = Depends(get_db)
):
    """Continue an existing chat session with adaptive questioning."""
    if not GEMINI_AVAILABLE:
        raise HTTPException(status_code=503, detail="Gemini AI not available")

    gemini = get_gemini_integration()
    if not gemini:
        raise HTTPException(
            status_code=503, detail="Gemini integration not initialized"
        )

    start_time = time.monotonic()
    try:
        # Continue chat with LLM-guided adaptive questioning
        response = await gemini.continue_guided_chat(
            session_id=request.session_id,
            message=request.message,
            language=request.language or "en",
            context=request.context or {},
        )
        if not _is_staff(http_req):
            response = dict(response)
            response["confidence"] = 0.0
        resp_model = ChatResponse(**response)
        try:
            log_api_access(
                db=db,
                user_id=None,
                endpoint=str(http_req.url.path),
                method=http_req.method,
                status_code=200 if resp_model.success else 500,
                response_time_ms=int((time.monotonic() - start_time) * 1000),
                ip_address=http_req.client.host if http_req.client else None,
                user_agent=http_req.headers.get("user-agent"),
            )
        except Exception:
            pass
        return resp_model
    except Exception as e:
        logger.error(f"Error continuing chat: {e}")
        resp_model = ChatResponse(
            success=False,
            message="Sorry, I couldn't process your message. Please try again.",
            language=request.language or "en",
            confidence=0.0,
            facts_extracted=[],
            suggested_agencies=[],
            error=str(e),
        )
        try:
            log_api_access(
                db=db,
                user_id=None,
                endpoint=str(http_req.url.path),
                method=http_req.method,
                status_code=500,
                response_time_ms=int((time.monotonic() - start_time) * 1000),
                ip_address=http_req.client.host if http_req.client else None,
                user_agent=http_req.headers.get("user-agent"),
            )
        except Exception:
            pass
        return resp_model


@app.post("/api/chat/facts-checklist", dependencies=[Depends(require_key)])
async def review_facts_checklist(
    request: FactChecklistRequest, http_req: Request, db: Session = Depends(get_db)
):
    """Review and validate extracted facts before letter generation."""
    if not GEMINI_AVAILABLE:
        raise HTTPException(status_code=503, detail="Gemini AI not available")

    gemini = get_gemini_integration()
    if not gemini:
        raise HTTPException(
            status_code=503, detail="Gemini integration not initialized"
        )

    start_time = time.monotonic()
    try:
        # Review facts with LLM
        response = await gemini.review_facts_checklist(
            session_id=request.session_id,
            facts=request.facts,
            corrections=request.corrections or {},
        )
        resp_model = FactChecklistResponse(**response)
        try:
            log_api_access(
                db=db,
                user_id=None,
                endpoint=str(http_req.url.path),
                method=http_req.method,
                status_code=200 if resp_model.success else 500,
                response_time_ms=int((time.monotonic() - start_time) * 1000),
                ip_address=http_req.client.host if http_req.client else None,
                user_agent=http_req.headers.get("user-agent"),
            )
        except Exception:
            pass
        return resp_model
    except Exception as e:
        logger.error(f"Error reviewing facts checklist: {e}")
        resp_model = FactChecklistResponse(
            success=False,
            reviewed_facts=[],
            confidence=0.0,
            ready_for_letter=False,
            error=str(e),
        )
        try:
            log_api_access(
                db=db,
                user_id=None,
                endpoint=str(http_req.url.path),
                method=http_req.method,
                status_code=500,
                response_time_ms=int((time.monotonic() - start_time) * 1000),
                ip_address=http_req.client.host if http_req.client else None,
                user_agent=http_req.headers.get("user-agent"),
            )
        except Exception:
            pass
        return resp_model


@app.post("/api/chat/generate-letter", dependencies=[Depends(require_key)])
async def generate_letter_from_chat(
    request: ChatRequest, http_req: Request, db: Session = Depends(get_db)
):
    """Generate letter from completed chat session."""
    if not GEMINI_AVAILABLE:
        raise HTTPException(status_code=503, detail="Gemini AI not available")

    gemini = get_gemini_integration()
    if not gemini:
        raise HTTPException(
            status_code=503, detail="Gemini integration not initialized"
        )

    start_time = time.monotonic()
    try:
        # Generate letter from chat session
        response = await gemini.generate_letter_from_chat(
            session_id=request.session_id,
            language=request.language or "en",
            context=request.context or {},
        )
        # No redaction in final letter payload; confidences are part of staff UI only
        resp = GeminiResponse(success=True, data=response, model_used="pro")
        try:
            log_api_access(
                db=db,
                user_id=None,
                endpoint=str(http_req.url.path),
                method=http_req.method,
                status_code=200,
                response_time_ms=int((time.monotonic() - start_time) * 1000),
                ip_address=http_req.client.host if http_req.client else None,
                user_agent=http_req.headers.get("user-agent"),
            )
        except Exception:
            pass
        return resp
    except Exception as e:
        logger.error(f"Error generating letter from chat: {e}")
        resp = GeminiResponse(success=False, error=str(e), model_used="pro")
        try:
            log_api_access(
                db=db,
                user_id=None,
                endpoint=str(http_req.url.path),
                method=http_req.method,
                status_code=500,
                response_time_ms=int((time.monotonic() - start_time) * 1000),
                ip_address=http_req.client.host if http_req.client else None,
                user_agent=http_req.headers.get("user-agent"),
            )
        except Exception:
            pass
        return resp


# Gemini AI Integration Endpoints
@app.post("/api/gemini/analyze-case-preview", dependencies=[Depends(require_key)])
async def analyze_case_preview(request: CaseAnalysisRequest):
    """Quick case analysis using Gemini Flash."""
    if not GEMINI_AVAILABLE:
        raise HTTPException(status_code=503, detail="Gemini AI not available")

    gemini = get_gemini_integration()
    if not gemini:
        raise HTTPException(
            status_code=503, detail="Gemini integration not initialized"
        )

    try:
        analysis = await gemini.analyze_case_preview(request.case_text)
        return GeminiResponse(success=True, data=analysis.__dict__, model_used="flash")
    except Exception as e:
        logger.error(f"Error in case analysis preview: {e}")
        return GeminiResponse(success=False, error=str(e), model_used="flash")


@app.post("/api/gemini/analyze-case-final", dependencies=[Depends(require_key)])
async def analyze_case_final(request: CaseAnalysisRequest):
    """Detailed case analysis using Gemini Pro."""
    if not GEMINI_AVAILABLE:
        raise HTTPException(status_code=503, detail="Gemini AI not available")

    gemini = get_gemini_integration()
    if not gemini:
        raise HTTPException(
            status_code=503, detail="Gemini integration not initialized"
        )

    try:
        analysis = await gemini.analyze_case_final(request.case_text, request.feedback)
        return GeminiResponse(success=True, data=analysis.__dict__, model_used="pro")
    except Exception as e:
        logger.error(f"Error in case analysis final: {e}")
        return GeminiResponse(success=False, error=str(e), model_used="pro")


@app.post("/api/gemini/generate-letter-preview", dependencies=[Depends(require_key)])
async def generate_letter_preview(request: LetterGenerationRequest):
    """Quick letter draft using Gemini Flash."""
    if not GEMINI_AVAILABLE:
        raise HTTPException(status_code=503, detail="Gemini AI not available")

    gemini = get_gemini_integration()
    if not gemini:
        raise HTTPException(
            status_code=503, detail="Gemini integration not initialized"
        )

    try:
        letter = await gemini.generate_letter_preview(request.case_data)
        return GeminiResponse(success=True, data=letter.__dict__, model_used="flash")
    except Exception as e:
        logger.error(f"Error in letter generation preview: {e}")
        return GeminiResponse(success=False, error=str(e), model_used="flash")


@app.post("/api/gemini/generate-letter-final", dependencies=[Depends(require_key)])
async def generate_letter_final(request: LetterGenerationRequest):
    """Polished letter using Gemini Pro."""
    if not GEMINI_AVAILABLE:
        raise HTTPException(status_code=503, detail="Gemini AI not available")

    gemini = get_gemini_integration()
    if not gemini:
        raise HTTPException(
            status_code=503, detail="Gemini integration not initialized"
        )

    try:
        letter = await gemini.generate_letter_final(request.case_data, request.feedback)
        return GeminiResponse(success=True, data=letter.__dict__, model_used="pro")
    except Exception as e:
        logger.error(f"Error in letter generation final: {e}")
        return GeminiResponse(success=False, error=str(e), model_used="pro")


@app.post("/api/gemini/recommend-approval-preview", dependencies=[Depends(require_key)])
async def recommend_approval_preview(request: ApprovalRecommendationRequest):
    """Quick approval recommendation using Gemini Flash."""
    if not GEMINI_AVAILABLE:
        raise HTTPException(status_code=503, detail="Gemini AI not available")

    gemini = get_gemini_integration()
    if not gemini:
        raise HTTPException(
            status_code=503, detail="Gemini integration not initialized"
        )

    try:
        # Convert dict back to CaseAnalysis object
        case_analysis = CaseAnalysis(**request.case_analysis)
        recommendation = await gemini.recommend_approval_preview(case_analysis)
        return GeminiResponse(
            success=True, data=recommendation.__dict__, model_used="flash"
        )
    except Exception as e:
        logger.error(f"Error in approval recommendation preview: {e}")
        return GeminiResponse(success=False, error=str(e), model_used="flash")


@app.post("/api/gemini/recommend-approval-final", dependencies=[Depends(require_key)])
async def recommend_approval_final(request: ApprovalRecommendationRequest):
    """Detailed approval recommendation using Gemini Pro."""
    if not GEMINI_AVAILABLE:
        raise HTTPException(status_code=503, detail="Gemini AI not available")

    gemini = get_gemini_integration()
    if not gemini:
        raise HTTPException(
            status_code=503, detail="Gemini integration not initialized"
        )

    try:
        # Convert dict back to CaseAnalysis object
        case_analysis = CaseAnalysis(**request.case_analysis)
        recommendation = await gemini.recommend_approval_final(
            case_analysis, request.feedback
        )
        return GeminiResponse(
            success=True, data=recommendation.__dict__, model_used="pro"
        )
    except Exception as e:
        logger.error(f"Error in approval recommendation final: {e}")
        return GeminiResponse(success=False, error=str(e), model_used="pro")


# Import and include authentication routes
from api.auth_endpoints import router as auth_router  # type: ignore

# Import and include governance routes
from api.governance_endpoints import router as governance_router  # type: ignore

app.include_router(auth_router)
app.include_router(governance_router)

# mount both root and /api for convenience
mount_routes("")
mount_routes("/api")

# Optional: run directly
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
        access_log=True,
    )
