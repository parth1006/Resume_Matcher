import re

def extract_contact_info(text):
    name, email, phone = "", None, None
    lines = text.strip().split("\n")
    if lines:
        name_candidate = re.sub(r"[^A-Za-z\s]", "", lines[0]).strip()
        if len(name_candidate.split()) >= 2:
            name = name_candidate.title()
    email = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    phone = re.search(r"(\+?\d[\d\s\-]{8,}\d)", text)
    return {
        "name": name,
        "email": email.group(0) if email else None,
        "phone": phone.group(0).replace(" ", "") if phone else None
    }

def extract_skills(text):
    skills = ["python", "java", "sql", "aws", "docker", "git", "ai", "ml", "tensorflow", "flask"]
    found = [s for s in skills if re.search(rf"\b{s}\b", text, re.IGNORECASE)]
    return found

def extract_education(text): return [l.strip() for l in text.split("\n") if "bachelor" in l.lower() or "master" in l.lower()]
def extract_experience_companies(text): return re.findall(r"(?:at|for)\s+([A-Z][A-Za-z&.\s]+)", text)
def extract_experience_years(text):
    m = re.search(r"(\d{1,2})\s*(?:year|yr|years|yrs)", text, re.IGNORECASE)
    return int(m.group(1)) if m else 0
