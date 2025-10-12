from __future__ import annotations
import os, re, uuid
from typing import List
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Local imports
from models import Base, Candidate, Job
from schemas import CandidateOut, JobIn, MatchResult
from parsers.pdf import pdf_to_text
from parsers.extract import extract_skills, extract_experience_years, extract_education
from matching.embedder import embed
from matching.scorer import composite_score
from matching.llm_groq import llm_match_groq


# -------------------------------------------------------------------
# Config and lifespan
# -------------------------------------------------------------------
IS_HF = os.environ.get("SPACE_ID") is not None
BASE_DIR = "/tmp/data" if IS_HF else "data"
engine = None
Session = sessionmaker(autoflush=False, autocommit=False, future=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle for Hugging Face–safe initialization."""
    global engine, Session
    os.makedirs(BASE_DIR, exist_ok=True)
    os.makedirs(f"{BASE_DIR}/resumes", exist_ok=True)
    print(f"[INFO] Using base directory: {BASE_DIR}")

    # Initialize DB here (after /data is mounted)
    engine = create_engine(f"sqlite:///{BASE_DIR}/app.db", future=True)
    Session.configure(bind=engine)
    Base.metadata.create_all(engine)

    yield  # Application runs here

    print("[INFO] Application shutting down.")


app = FastAPI(title="AI Resume Matcher (Groq Cloud)", lifespan=lifespan)

# -------------------------------------------------------------------
# Middleware
# -------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # OK for demo — restrict for prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------
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
    """Upload and parse a resume — auto-extract name, email, phone."""
    data = await resume.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    filename = _safe_filename("resume", resume.filename)
    path = os.path.join(BASE_DIR, "resumes", filename)
    with open(path, "wb") as f:
        f.write(data)

    # Convert PDF/text
    text = (
        pdf_to_text(path)
        if resume.filename.lower().endswith(".pdf")
        else data.decode("utf-8", "ignore")
    )

    from parsers.extract import extract_contact_info
    contact = extract_contact_info(text)
    name = contact.get("name") or "Unknown"
    email = contact.get("email")
    phone = contact.get("phone")

    skills = extract_skills(text)
    exp_years = extract_experience_years(text)
    edu = extract_education(text)
    emb = embed(text)

    with Session() as s:
        c = Candidate(
            name=name,
            email=email,
            phone=phone,
            raw_text=text,
            skills=skills,
            experience_years=exp_years,
            education=edu,
            embedding=emb,
        )
        s.add(c)
        s.commit()
        s.refresh(c)

        return CandidateOut(
            id=c.id,
            name=c.name,
            skills=skills,
            experience_years=exp_years,
            education=edu,
        )


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
        shortlist = prelim[: max(10, top_k * 2)]

        results: List[MatchResult] = []
        for c, _, comps in shortlist:
            try:
                llm = llm_match_groq(job.jd_text, c.raw_text, c.name)
                llm_fit = llm.get("fit_score", None)
            except Exception:
                llm = {"bullets": ["LLM scoring unavailable"], "concerns": []}
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
            if llm.get("bullets"):
                justification_lines.extend([f"• {b}" for b in llm["bullets"]])
            if llm.get("concerns"):
                justification_lines.append(f"Concerns: {', '.join(llm['concerns'])}")

            results.append(
                MatchResult(
                    candidate_id=c.id,
                    candidate_name=c.name,
                    score=round(100 * final_comps.get("final", 0.0), 2),
                    components={k: round(v, 3) for k, v in final_comps.items()},
                    justification="\n".join(justification_lines),
                )
            )

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]
