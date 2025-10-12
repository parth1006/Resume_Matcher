from pydantic import BaseModel
from typing import List, Optional, Dict

class CandidateIn(BaseModel):
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    text: Optional[str] = None  # if resume text provided directly

class CandidateOut(BaseModel):
    id: int
    name: str
    skills: List[str]
    experience_years: float
    education: List[str]

class JobIn(BaseModel):
    title: str
    jd_text: str
    required_skills: List[str] = []
    nice_to_have_skills: List[str] = []

class MatchResult(BaseModel):
    candidate_id: int
    candidate_name: str
    score: float
    components: Dict[str, float]
    justification: str
