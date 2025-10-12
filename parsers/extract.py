import re, spacy
from typing import List

CANONICAL_SKILLS = { s.lower() for s in [
    # Core languages
    "python","java","c++","c","sql","javascript","typescript",
    # ML / Data
    "pandas","numpy","tensorflow","pytorch","scikit-learn","keras",
    "matplotlib","seaborn","opencv","huggingface","transformers",
    # Cloud & DevOps
    "docker","kubernetes","aws","gcp","azure",
    # Frameworks / APIs
    "fastapi","flask","react","node.js","express","spring","django",
    # Domains
    "nlp","computer vision","data analysis","machine learning"
]}

SKILL_SYNONYMS = {
    "tf": "tensorflow",
    "sklearn": "scikit-learn",
    "js": "javascript",
    "cv": "computer vision",
}

_nlp = None
def get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm", disable=["parser","ner","tagger"])
    return _nlp

def extract_skills(text: str) -> List[str]:
    """Detect skills from resume text using keyword + synonym mapping."""
    low = text.lower()
    found = set()

    # keyword matching
    for skill in CANONICAL_SKILLS:
        if re.search(rf"(?<![a-z0-9]){re.escape(skill)}(?![a-z0-9])", low):
            found.add(skill)

    # synonym mapping
    for short, full in SKILL_SYNONYMS.items():
        if re.search(rf"(?<![a-z0-9]){re.escape(short)}(?![a-z0-9])", low):
            found.add(full)

    # fallback NER
    nlp = get_nlp()
    doc = nlp(text)
    for token in doc:
        if token.text.lower() in CANONICAL_SKILLS:
            found.add(token.text.lower())

    return sorted(found)

def extract_experience_years(text: str) -> float:
    """Extract total experience years using regex heuristics."""
    years = re.findall(r"(\d+(?:\.\d+)?)\s*(?:\+?\s*)?(?:years|yrs|year)", text.lower())
    if years:
        return max(float(y) for y in years)
    # fallback heuristic: roles like 'Software Engineer', 'Intern', etc.
    role_count = len(re.findall(r"Engineer|Developer|Intern|Manager", text, re.I))
    return min(role_count * 0.5, 10.0)  # heuristic cap

def extract_education(text: str) -> List[str]:
    """Extract education details by matching degree keywords and institution lines."""
    edu_keywords = [
        "B.Tech","BSc","M.Tech","MSc","B.E.","BEng","MEng","PhD",
        "Bachelor","Master","Diploma","Computer Science","Engineering"
    ]
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    edu = [ln for ln in lines if any(k.lower() in ln.lower() for k in edu_keywords)]
    return edu[:5]
def extract_contact_info(text: str) -> dict:
    """
    Extract name, email, and phone number from resume text using regex & heuristics.
    Returns: {"name": str | None, "email": str | None, "phone": str | None}
    """

    # --- Email ---
    email_match = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    email = email_match.group(0) if email_match else None

    # --- Phone number ---
    phone_match = re.search(
        r"(\+?\d{1,3}[\s-]?)?(?:\(?\d{2,4}\)?[\s-]?)?\d{3,4}[\s-]?\d{4}",
        text
    )
    phone = phone_match.group(0) if phone_match else None

    # --- Name heuristic ---
    # Usually in the first few lines, before the email or "Resume"/"CV" keyword.
    first_lines = text.strip().split("\n")[:5]
    possible_names = [
        ln.strip() for ln in first_lines
        if 2 <= len(ln.split()) <= 4
        and not re.search(r"@|resume|curriculum|vitae|cv", ln, re.I)
    ]
    name = possible_names[0] if possible_names else None

    return {"name": name, "email": email, "phone": phone}