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
    """Startup/shutdown lifecycle — fully Hugging Face + Windows compatible."""
    global engine, Session

    base_dir = BASE_DIR
    resumes_dir = os.path.join(base_dir, "resumes")

    # ✅ Ensure directories exist
    try:
        os.makedirs(resumes_dir, exist_ok=True)
        print(f"[INFO] ✅ Created directory: {resumes_dir}")
    except Exception as e:
        print(f"[ERROR] Failed to create directories: {e}")
        raise

    db_path = os.path.join(base_dir, "app.db")
    print(f"[INFO] Using base directory: {base_dir}")
    print(f"[INFO] Database path: {db_path}")

    # ✅ Test write permissions (safe on Windows too)
    test_path = os.path.join(base_dir, "write_test.txt")
    try:
        with open(test_path, "w") as f:
            f.write("ok")
        # Close before deleting
        if os.path.exists(test_path):
            try:
                os.remove(test_path)
            except PermissionError:
                # Windows can delay handle release — rename instead
                renamed = test_path + ".old"
                os.rename(test_path, renamed)
                os.remove(renamed)
        print("[INFO] ✅ Write test passed.")
    except Exception as e:
        print(f"[ERROR] ❌ Write test failed: {e}")

    # ✅ Create SQLite engine (safe threading)
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
        
        # ✅ Create embedding from the parsed text (add raw_text if missing)
        raw_text = result.get("raw_text", "")
        if not raw_text:
            # Fallback: use concatenated fields if raw_text is empty
            raw_text = f"{result.get('name', '')} {' '.join(result.get('skills', []))}"
        
        embedding = _safe_embed(raw_text) if raw_text else None
        
        # ✅ Extract data from parser result (matching your schema)
        email_list = result.get("email", [])
        phone_list = result.get("phone_numbers", [])
        skills = result.get("skills", [])
        education = result.get("education", [])
        work_history = result.get("work_history", [])
        
        # ✅ FIX: Deduplicate work history (catches reversed duplicates too)
        seen = set()
        unique_work_history = []
        for entry in work_history:
            company = entry.get("company", "").lower().strip()
            designation = entry.get("designation", "").lower().strip()
            
            # Skip empty entries
            if not company or not designation:
                continue
            
            # Create TWO keys - forward and reverse order
            # This catches cases where company/designation are swapped
            key1 = (company, designation)
            key2 = (designation, company)
            
            # Skip if either key variant has been seen
            if key1 in seen or key2 in seen:
                print(f"[DEBUG] Skipping duplicate: {designation} at {company}")
                continue
            
            # Add both variants to prevent future matches
            seen.add(key1)
            seen.add(key2)
            unique_work_history.append(entry)
        
        # ✅ Extract experience years
        experience_details = result.get("experience_details", {})
        experience_years = experience_details.get("total_years") if experience_details else None
        
        # ✅ FOR DATABASE: Convert lists to single values
        # Database expects single email/phone strings
        email_for_db = email_list[0] if email_list and len(email_list) > 0 else None
        phone_for_db = phone_list[0] if phone_list and len(phone_list) > 0 else None
        
        print(f"[DEBUG] Parsed candidate: {result.get('name')}")
        print(f"[DEBUG] Emails: {email_list}")
        print(f"[DEBUG] Phones: {phone_list}")
        print(f"[DEBUG] Skills count: {len(skills)}")
        print(f"[DEBUG] Education count: {len(education)}")
        print(f"[DEBUG] Work history: {len(unique_work_history)} (deduplicated from {len(work_history)})")
        print(f"[DEBUG] Embedding created: {embedding is not None}")
        
        # ✅ Save to database with single email/phone
        with Session() as s:
            candidate = Candidate(
                name=result.get("name", "Unknown"),
                email=email_for_db,  # ✅ Single string for database
                phone=phone_for_db,   # ✅ Single string for database
                skills=skills,        # ✅ List - JSONType handles serialization
                education=education,  # ✅ List - JSONType handles serialization
                experience_years=experience_years,
                companies=unique_work_history,  # ✅ Deduplicated work history
                raw_text=raw_text,
                embedding=embedding,  # ✅ JSONType handles serialization
                resume_path=save_path
            )
            s.add(candidate)
            s.commit()
            s.refresh(candidate)
            print(f"[INFO] ✅ Saved candidate {candidate.name} with ID {candidate.id}")
            
            # Store the ID for response
            candidate_id = candidate.id

        # ✅ Return full parser result (matching CandidateOut schema)
        # This preserves all the detailed parsing data for the frontend
        return CandidateOut(
            id=candidate_id,
            name=result.get("name"),
            email=email_list,  # ✅ Return list as per schema
            phone_numbers=phone_list,  # ✅ Return list as per schema
            skills=skills,
            skills_categorized=result.get("skills_categorized", {}),
            total_experience_years=experience_years,
            experience_details=experience_details,
            education=education,
            work_history=unique_work_history,  # ✅ Return deduplicated version
            parsed_at=result.get("parsed_at"),
            file_name=result.get("file_name")
        )

    except Exception as e:
        print(f"[ERROR] Failed to process resume: {e}")
        import traceback
        traceback.print_exc()
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
        print(f"[DEBUG] Found {len(candidates)} candidates in database")
        
        if not candidates:
            print("[DEBUG] No candidates found, returning empty list")
            return []

        prelim = []
        for candidate in candidates:
            # Debug: Check if embeddings exist
            if candidate.embedding is None:
                print(f"[WARNING] Candidate {candidate.name} has no embedding, skipping")
                continue
            
            if job.embedding is None:
                print(f"[WARNING] Job {job.title} has no embedding")
            
            comps = composite_score(
                candidate.embedding,
                job.embedding,
                candidate.skills or [],
                job.required_skills or [],
                job.nice_to_have_skills or [],
                llm_score=None,
            )
            print(f"[DEBUG] {candidate.name}: composite score = {comps}")
            prelim.append((candidate, comps.get("final", 0.0), comps))
        
        print(f"[DEBUG] Preliminary matches: {len(prelim)} candidates scored")

        prelim.sort(key=lambda x: x[1], reverse=True)
        shortlist = prelim[: max(5, top_k * 2)]

        results: List[MatchResult] = []
        for candidate, _, comps in shortlist:  # ✅ Renamed 'c' to 'candidate'
            try:
                llm = llm_match_groq(job.jd_text, candidate.raw_text, candidate.name)
                llm_fit = llm.get("fit_score", None)
            except Exception as e:
                print(f"[ERROR] LLM scoring failed for {candidate.name}: {e}")
                llm = {
                    "summary_bullets": ["LLM scoring unavailable"], 
                    "key_strengths": [], 
                    "concerns": [], 
                    "reasoning": ""
                }
                llm_fit = None

            final_comps = composite_score(
                candidate.embedding,
                job.embedding,
                candidate.skills or [],
                job.required_skills or [],
                job.nice_to_have_skills or [],
                llm_score=llm_fit,
            )

            justification_lines = []
            # ✅ Fixed variable names in list comprehensions
            if llm.get("summary_bullets"):
                justification_lines.append("**Summary:**")
                justification_lines.extend([f"- {bullet}" for bullet in llm["summary_bullets"]])
            if llm.get("key_strengths"):
                justification_lines.append("\n**Key Strengths:**")
                justification_lines.extend([f"- {strength}" for strength in llm["key_strengths"]])
            if llm.get("concerns"):
                justification_lines.append("\n**Concerns:**")
                justification_lines.extend([f"- {concern}" for concern in llm["concerns"]])
            if llm.get("reasoning"):
                justification_lines.append(f"\n**Reasoning:** {llm['reasoning']}")

            results.append(
                MatchResult(
                    candidate_id=candidate.id,
                    candidate_name=candidate.name,
                    score=round(100 * final_comps.get("final", 0.0), 2),
                    components={k: round(v, 3) for k, v in final_comps.items()},
                    justification="\n".join(justification_lines),
                )
            )
        
        results.sort(key=lambda x: x.score, reverse=True)
        print(f"[DEBUG] Returning {len(results[:top_k])} results")
        return results[:top_k]