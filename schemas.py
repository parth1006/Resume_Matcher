from pydantic import BaseModel
from typing import List, Optional, Dict, Any


# Education entry (parsed structure)
class EducationEntry(BaseModel):
    degree: Optional[str] = None
    field: Optional[str] = None
    institution: Optional[str] = None
    year: Optional[str] = None


# Work history entry
class WorkHistoryEntry(BaseModel):
    designation: Optional[str] = None
    company: Optional[str] = None
    duration: Optional[str] = None


# Output model for parsed resume data
class CandidateOut(BaseModel):
    id: Optional[int] = None
    name: Optional[str] = None
    email: Optional[List[str]] = []
    phone_numbers: Optional[List[str]] = []
    skills: List[str] = []
    skills_categorized: Optional[Dict[str, List[str]]] = {}
    total_experience_years: Optional[float] = None
    experience_details: Optional[Dict[str, Any]] = {}
    education: List[Dict[str, Any]] = []
    work_history: List[Dict[str, Any]] = []
    parsed_at: Optional[str] = None
    file_name: Optional[str] = None


# Job posting model
class JobIn(BaseModel):
    title: str
    jd_text: str
    required_skills: List[str] = []
    nice_to_have_skills: List[str] = []


# Match score result
class MatchResult(BaseModel):
    candidate_id: Optional[int] = None
    candidate_name: Optional[str] = None
    score: float
    components: Dict[str, float]
    justification: str
