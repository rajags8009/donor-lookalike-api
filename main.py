from fastapi import FastAPI, Query
from typing import List, Optional
import os
import numpy as np
from scipy.spatial.distance import cosine, euclidean
from supabase import create_client, Client  # From official supabase-py package
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow CORS from anywhere (for dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load env vars (Render will inject them securely)
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get("SUPABASE_ANON_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


def compute_similarity(vec1: List[float], vec2: List[float], metric: str):
    if metric == "euclidean":
        return -euclidean(vec1, vec2)  # Negative so higher is better
    else:
        return 1 - cosine(vec1, vec2)   # Cosine similarity


@app.get("/lookalike")
def get_lookalikes(
    donor_id: int,
    top_n: int = 10,
    metric: str = "cosine",
    cluster_only: bool = False,
):
    # Fetch all donors w/ their embeddings + metadata.
    response = supabase.table("donor_vectors").select("*").execute()
    rows = response.data

    if not rows:
        return []

    # Find selected donor's vector and metadata first.
    target_donor = next((d for d in rows if d["donor_id"] == donor_id), None)
    
    if not target_donor or "embedding" not in target_donor:
        return {"error": "Donor not found or embedding missing"}

    target_vector = np.array(target_donor["embedding"])

    results = []

    for row in rows:
        if row["donor_id"] == donor_id:
            continue  # skip self
        
        if cluster_only and row.get("cluster_label") != target_donor.get("cluster_label"):
            continue
        
        emb_vec = np.array(row.get("embedding", []))
        if emb_vec.size == 0:
            continue
        
        sim_score = compute_similarity(target_vector, emb_vec, metric)

        row_copy = dict(row)  # Avoid mutating original row object from Supabase client
        row_copy["similarity"] = sim_score
        
        results.append(row_copy)

    sorted_results = sorted(results, key=lambda x: x["similarity"], reverse=True)
    
    return sorted_results[:top_n]