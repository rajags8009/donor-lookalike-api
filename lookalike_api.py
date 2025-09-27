from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import pandas as pd
import numpy as np
from typing import List, Optional
from sklearn.preprocessing import normalize

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Lookalike Search API")

# CORS settings so frontend can call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load donor vectors at startup from CSV file (local)
try:
    df_vectors = pd.read_csv("donor_vectors.csv")
    vectors = df_vectors.select_dtypes(include=[np.number]).values
    donor_ids = df_vectors["donor_id"].values
except Exception as e:
    raise RuntimeError(f"❌ Failed to load donor_vectors.csv: {e}")


class LookalikeResult(BaseModel):
    donor_id: str
    name: str = ""
    cluster_label: Optional[int] = None
    site_id: Optional[str] = None
    total_donated: Optional[float] = None
    avg_donation_size: Optional[float] = None
    donation_count: Optional[int] = None 
    campaign_count: Optional[int] = None 
    health_screening_count: Optional[int] = None 
    similarity: float


@app.get("/lookalike", response_model=List[LookalikeResult])
def lookalikey_search(
        donor_id: str,
        metric: str = Query("cosine", pattern="^(cosine|euclidean)$"),
        top_n: int = Query(10),
        cluster_only: bool = False):

    if donor_id not in donor_ids:
        raise HTTPException(status_code=404, detail=f"{donor_id=} not found")

    idx         = np.where(donor_ids == donor_id)[0][0]
    target_vec  = vectors[idx]

    mask        = np.ones(len(vectors), dtype=bool)

    if cluster_only and 'cluster_label' in df_vectors.columns:
        target_cluster_vals = df_vectors.loc[df_vectors['donor_id'] == donor_id, 'cluster_label'].values

        if len(target_cluster_vals) > 0 and pd.notna(target_cluster_vals[0]):
            target_cluster       = target_cluster_vals[0]
            mask &= (df_vectors['cluster_label'] == target_cluster)
        else:
            print(f"[WARN ⚠️ ] Donor {donor_id} has no valid cluster. Skipping filter.")

    masked_indices     = np.where(mask)[0]

    if metric == "cosine":
        normed_all     = normalize(vectors)
        norm_targetvec = normed_all[idx]
        sim_scores     = normed_all[masked_indices].dot(norm_targetvec.T).flatten()

        if idx in masked_indices:
            sim_scores[np.where(masked_indices == idx)] -= 1

    else:
        diff       = vectors[masked_indices] - target_vec.reshape(1, -1)
        dists      = np.linalg.norm(diff, axis=1)

        if idx in masked_indices:
            dists[np.where(masked_indices == idx)] += 1e9

        sim_scores  = -dists

    
    top_idx_local   = np.argsort(sim_scores)[::-1][:top_n]

    results : List[LookalikeResult]   = []

    for i in top_idx_local:
        global_i  = masked_indices[i]
        row       = df_vectors.iloc[global_i]

        try:
            results.append(LookalikeResult(
                donor_id=str(row["donor_id"]) if "donor_id" in row else "",

                name=str(row["name"]) if "name" in row and pd.notna(row["name"]) else "",

                cluster_label=int(row["cluster_label"]) 
                    if "cluster_label" in row and pd.notna(row["cluster_label"]) else None,

                site_id=str(row["site_id"])
                    if "site_id" in row and pd.notna(row["site_id"]) else None,

                total_donated=float(row["total_donated"])
                    if "total_donated" in row and pd.notna(row["total_donated"]) else None,

                avg_donation_size=float(row["avg_donation_size"])
                    if "avg_donation_size" in row and pd.notna(row["avg_donation_size"]) else None,

                donation_count=int(row["donation_count"])
                    if "donation_count" in row and pd.notna(row["donation_count"]) else None,

                campaign_count=int(row["campaign_count"])
                    if "campaign_count" in row and pd.notna(row["campaign_count"]) else None,

                health_screening_count=int(row["health_screening_count"])
                    if "health_screening_count" in row and pd.notna(row["health_screening_count"]) else None,

                similarity=float(sim_scores[i])
            ))

            print(f"[MATCH ✅ ] Added {row['donor_id']}")

        except Exception as e:
            print(f"[ERROR ❌ ] Could not process index={global_i}: {e}")

    
    print(f"[LOOKALIKE ✅ ] Returned {len(results)} matches for {donor_id}")
    
    return results


@app.get("/vector", response_model=dict)
def get_vector(donor_id: str):
    if donor_id not in donor_ids:
        raise HTTPException(status_code=404, detail='Not found')

    idx     = np.where(donor_ids == donor_id)[0][0]
    
    return {
       "vector": vectors[idx].tolist(),
       "dimension": len(vectors[idx]),
       "donor": df_vectors.iloc[idx].get("name", "")
   }


@app.get("/top_looklikeable", response_model=list)
def get_top_similars(top_n:int=10):
    normed_vecs=normalize(vectors)
results=[]



@app.get("/top_looklikeable", response_model=list)
def get_top_similars(top_n: int = 10):
    normed_vecs = normalize(vectors)
    results = []

    for i, did in enumerate(donor_ids):
        sims = (normed_vecs @ normed_vecs[i]).flatten()
        sims[i] = -1  # exclude self-match

        max_sim = max(sims)

        if max_sim > 0.9:
            results.append({
                "donor": did,
                "max_similarity": float(max_sim)
            })

    return sorted(results, key=lambda x: -x["max_similarity"])[:top_n]