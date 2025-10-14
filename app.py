from __future__ import annotations
import os, re, uuid
from typing import List
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import Base, Candidate, Job
from schemas import CandidateOut, JobIn, MatchResult
from parsers.pdf import pdf_to_text
from parsers.extract import ResumeParser
from matching.embedder import embed
from matching.scorer import composite_score
from matching.llm_groq import llm_match_groq

IS_HF = os.environ.get("SPACE_ID") is not None
# ✅ Use /tmp for HF Spaces, data for local
BASE_DIR = "/tmp/data" if IS_HF else "data"
engine = None
Session = sessionmaker(autoflush=False, autocommit=False, future=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle — fully Hugging Face compatible."""
    global engine, Session

    # ✅ Use /tmp for writable storage in HF Spaces
    base_dir = BASE_DIR
    resumes_dir = os.path.join(base_dir, "resumes")

    try:
        os.makedirs(resumes_dir, exist_ok=True)
        print(f"[INFO] ✅ Created directory: {resumes_dir}")
    except Exception as e:
        print(f"[ERROR] Failed to create directories: {e}")
        raise

    db_path = os.path.join(base_dir, "app.db")
    print(f"[INFO] Using base directory: {base_dir}")
    print(f"[INFO] Database path: {db_path}")

    # ✅ Test write permissions
    try:
        test_path = os.path.join(base_dir, "write_test.txt")
        with open(test_path, "w") as f:
            f.write("ok")
        os.remove(test_path)
        print("[INFO] ✅ Write test passed.")
    except Exception as e:
        print(f"[ERROR] ❌ Write test failed: {e}")
        raise

    # ✅ Create SQLite engine
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
        future=True,
    )
    Session.configure(bind=engine)
    Base.metadata.create_all(engine)

    print("[INFO] Database and folders ready.")
    yield
    print("[INFO] Application shutting down.")


app = FastAPI(title="AI Resume Matcher (Groq Cloud)", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # OK for demo — restrict for prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _safe_filename(prefix: str, original: str) -> str:
    base = re.sub(r"[^A-Za-z0-9._-]", "_", original)
    unique = uuid.uuid4().hex[:8]
    return f"{prefix}_{unique}_{base}"


def _safe_embed(text: str) -> List[float] | None:
    try:
        return embed(text)
    except Exception:
        return None


# -------------------------------------------------------------------
# Routes
# -------------------------------------------------------------------
@app.post("/candidates/upload", response_model=CandidateOut)
async def upload_candidate(resume: UploadFile = File(...)):
    try:
        # ✅ Use the same BASE_DIR as lifespan
        base_dir = BASE_DIR
        resume_dir = os.path.join(base_dir, "resumes")
        os.makedirs(resume_dir, exist_ok=True)
        
        safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", resume.filename)
        unique_name = f"{uuid.uuid4().hex[:8]}_{safe_name}"
        save_path = os.path.join(resume_dir, unique_name)

        with open(save_path, "wb") as f:
            f.write(await resume.read())

        parser = ResumeParser()
        result = parser.parse(save_path)

        return CandidateOut(**result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to extract details: {e}")


@app.post("/jobs", response_model=dict)
def create_job(job: JobIn):
    """Create a job with embeddings for later matching."""
    if not job.jd_text or not job.jd_text.strip():
        raise HTTPException(status_code=400, detail="jd_text cannot be empty.")

    job_emb = _safe_embed(job.jd_text)

    with Session() as s:
        j = Job(
            title=job.title,
            jd_text=job.jd_text,
            required_skills=job.required_skills or [],
            nice_to_have_skills=job.nice_to_have_skills or [],
            embedding=job_emb,
        )
        s.add(j)
        s.commit()
        s.refresh(j)
        return {"job_id": j.id, "title": j.title}


@app.get("/jobs/list", response_model=list[dict])
def list_jobs():
    """Return all job IDs and titles for dropdown display."""
    with Session() as s:
        jobs = s.query(Job).all()
        return [{"id": j.id, "title": j.title} for j in jobs]


@app.get("/match/{job_id}", response_model=List[MatchResult])
def match_candidates(job_id: int, top_k: int = 5):
    """Rank candidates for a job."""
    if top_k <= 0:
        raise HTTPException(status_code=400, detail="top_k must be > 0")

    with Session() as s:
        job = s.get(Job, job_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")

        candidates = s.query(Candidate).all()
        if not candidates:
            return []

        prelim = []
        for c in candidates:
            comps = composite_score(
                c.embedding,
                job.embedding,
                c.skills or [],
                job.required_skills or [],
                job.nice_to_have_skills or [],
                llm_score=None,
            )
            prelim.append((c, comps.get("final", 0.0), comps))

        prelim.sort(key=lambda x: x[1], reverse=True)
        shortlist = prelim[: max(5, top_k * 2)]

        results: List[MatchResult] = []
        for c, _, comps in shortlist:
            try:
                llm = llm_match_groq(job.jd_text, c.raw_text, c.name)
                llm_fit = llm.get("fit_score", None)
            except Exception:
                llm = {"summary_bullets": ["LLM scoring unavailable"], "key_strengths": [], "concerns": [], "reasoning": ""}
                llm_fit = None

            final_comps = composite_score(
                c.embedding,
                job.embedding,
                c.skills or [],
                job.required_skills or [],
                job.nice_to_have_skills or [],
                llm_score=llm_fit,
            )

            justification_lines = []
            # ✅ Combine bullets, strengths, concerns, reasoning into a clear explanation
            if llm.get("summary_bullets"):
                justification_lines.append("**Summary:**")
                justification_lines.extend([f"- {b}" for b in llm["summary_bullets"]])
            if llm.get("key_strengths"):
                justification_lines.append("\n**Key Strengths:**")
                justification_lines.extend([f"- {s}" for s in llm["key_strengths"]])
            if llm.get("concerns"):
                justification_lines.append("\n**Concerns:**")
                justification_lines.extend([f"- {c}" for c in llm["concerns"]])
            if llm.get("reasoning"):
                justification_lines.append(f"\n**Reasoning:** {llm['reasoning']}")

            results.append(
                MatchResult(
                    candidate_id=c.id,
                    candidate_name=c.name,
                    score=round(100 * final_comps.get("final", 0.0), 2),
                    components={k: round(v, 3) for k, v in final_comps.items()},
                    justification="\n".join(justification_lines),
                )
            )
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]