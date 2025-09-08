import os, json, numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from sentence_transformers import SentenceTransformer

def l2norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-9
    return x / n

def load_artifacts(model_dir):
    import numpy as np
    with open(os.path.join(model_dir,"artifacts.json")) as f:
        A=json.load(f)
    labels=A["labels"]; tops=A["tops"]; model_name=A["model"]
    L=np.load(os.path.join(model_dir,"label_emb.npy"))
    T=np.load(os.path.join(model_dir,"top_emb.npy"))
    idx_by_top={t:[] for t in tops}
    for j,lab in enumerate(labels):
        t=lab.split("/")[0]
        if t in idx_by_top: idx_by_top[t].append(j)
    return {
        "labels": labels,
        "tops": tops,
        "idx_by_top": idx_by_top,
        "L": l2norm(L.astype("float32")),
        "T": l2norm(T.astype("float32")),
        # Lazy-load model to reduce cold start memory/time
        "model": None,
        "model_name": model_name,
    }

def get_model(arts):
    if arts.get("model") is None:
        arts["model"] = SentenceTransformer(arts["model_name"])
    return arts["model"]

CO_PRIORS = {
  "social_support/comcare_short_mid_term": {
    "utilities_comms/electricity_spgroup": 0.15,
    "utilities_comms/telecom_bills_singtel": 0.12,
    "utilities_comms/telecom_bills_starhub": 0.10,
    "utilities_comms/telecom_bills_m1": 0.10,
    "utilities_comms/telecom_bills_tpg": 0.10,
    "housing/town_council_scc_arrears": 0.15,
    "housing/hdb_loan_arrears": 0.12,
    "tax_finance/mortgage_arrears_bank": 0.10
  },
  "employment/career_services_wsg": {"employment/career_services_e2i": 0.06},
  "employment/career_services_e2i": {"employment/career_services_wsg": 0.06},
  "housing/hdb_rental_flat": {"social_support/comcare_short_mid_term": 0.12}
}
def apply_priors(label_scores, seed_threshold=0.45):
    seeds=[lab for lab,sc in label_scores.items() if sc>=seed_threshold]
    for s in seeds:
        for tgt,w in CO_PRIORS.get(s,{}).items():
            if tgt in label_scores:
                label_scores[tgt] = min(1.0, label_scores[tgt] + w*label_scores[s])
    return label_scores

def predict_batch(arts, texts, threshold_top=0.30, threshold_child=0.36,
                  top_k_top=6, top_k_child=6, top_k_total=18,
                  use_priors=True, seed_prior_threshold=0.45):
    model = get_model(arts); L=arts["L"]; T=arts["T"]
    labels=arts["labels"]; tops=arts["tops"]; idx_by_top=arts["idx_by_top"]
    E = l2norm(model.encode(texts, convert_to_numpy=True))
    S_top = E @ T.T
    out=[]
    for i,txt in enumerate(texts):
        tscores = {tops[k]: float(S_top[i,k]) for k in range(len(tops))}
        chosen_tops = [t for t,_ in sorted(tscores.items(), key=lambda x:-x[1])[:top_k_top] if tscores[t]>=threshold_top]
        label_scores = {}
        for t in chosen_tops:
            idxs = idx_by_top[t]
            if not idxs: continue
            Lsub = L[idxs]
            S_child = float(S_top[i, tops.index(t)]) * (E[i:i+1] @ Lsub.T)
            order = np.argsort(-S_child.flatten())[:top_k_child]
            for o in order:
                lab = labels[idxs[o]]
                sc  = float(S_child.flatten()[o])
                if sc >= threshold_child:
                    label_scores[lab] = max(label_scores.get(lab, 0.0), sc)
        if use_priors:
            label_scores = apply_priors(label_scores, seed_threshold=seed_prior_threshold)
        ranked = sorted(label_scores.items(), key=lambda x:-x[1])[:top_k_total]
        out.append({"text": txt, "labels": [l for l,_ in ranked], "scores": {l: s for l,s in ranked},
                    "top_categories": [{"top": t, "score": tscores[t]} for t in chosen_tops]})
    return out

class PredictReq(BaseModel):
    texts: List[str]
    threshold_top: float = 0.30
    threshold_child: float = 0.36
    top_k_top: int = 6
    top_k_child: int = 6
    top_k_total: int = 18
    seed_prior_threshold: float = 0.45
    use_priors: bool = True
    return_providers: bool = True

app = FastAPI()
MODEL_DIR = os.environ.get("MODEL_DIR", "./api/artifacts_zs_hier_plus")
PROVIDERS = os.environ.get("PROVIDERS_JSON", "./api/providers_map.json")
arts = load_artifacts(MODEL_DIR)
providers_map = {}
if os.path.exists(PROVIDERS):
    with open(PROVIDERS) as f:
        providers_map = json.load(f)

# Optional CORS support for cross-origin frontend (e.g., Firebase Hosting)
cors_origins = os.environ.get("CORS_ORIGINS", "").strip()
if cors_origins:
    origins = [o.strip() for o in cors_origins.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

@app.get("/healthz")
def healthz():
    return {"ok": True, "labels": len(arts["labels"]), "tops": len(arts["tops"])}

@app.post("/predict")
def predict(req: PredictReq):
    preds = predict_batch(
        arts, req.texts,
        threshold_top=req.threshold_top,
        threshold_child=req.threshold_child,
        top_k_top=req.top_k_top,
        top_k_child=req.top_k_child,
        top_k_total=req.top_k_total,
        use_priors=req.use_priors,
        seed_prior_threshold=req.seed_prior_threshold
    )
    if req.return_providers and providers_map:
        for p in preds:
            bundle = {}
            for lab in p["labels"]:
                if lab in providers_map:
                    bundle[lab] = providers_map[lab]
            p["providers"] = bundle
    return {"predictions": preds}
