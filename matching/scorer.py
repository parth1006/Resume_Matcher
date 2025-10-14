from .embedder import embed, cosine
from typing import Dict, List
import math

def rule_skill_overlap(c_skills: List[str], req: List[str], nice: List[str]) -> Dict[str,float]:
    s = set(s.lower() for s in c_skills)
    r = set(x.lower() for x in req)
    n = set(x.lower() for x in nice)
    req_cover = len(s & r) / (len(r) or 1)
    nice_cover = len(s & n) / (len(n) or 1)
    return {"req_cover": req_cover, "nice_cover": nice_cover}

def composite_score(cand_embedding, job_embedding, c_skills, req, nice, llm_score=None) -> Dict[str,float]:
    sim = cosine(cand_embedding, job_embedding)            # 0..1
    covers = rule_skill_overlap(c_skills, req, nice)       # 0..1 each
    #add try catch
    # Normalize llm_score -> 0–1
    llm_norm = 0.0
    if llm_score is not None:
        llm_norm = max(0.0, min(1.0, llm_score / 10.0))

    # Weights: documented in README
    w_sim, w_req, w_nice, w_llm = 0.45, 0.35, 0.1, 0.10
    base = w_sim*sim + w_req*covers["req_cover"] + w_nice*covers["nice_cover"] + w_llm*llm_norm

    return {
        "similarity": sim,
        "req_cover": covers["req_cover"],
        "nice_cover": covers["nice_cover"],
        "llm": llm_norm,
        "final": base
    }

