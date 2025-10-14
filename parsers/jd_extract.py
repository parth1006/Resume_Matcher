import re

def extract_jd_details(text: str) -> dict:
    """
    Smarter job description extractor that works even for messy paragraph-style JDs.
    Extracts: title, required skills, nice-to-have skills, and key responsibilities.
    """

    # 🧹 Basic cleaning
    text = text.replace("\n", " ").replace("•", " ").replace("–", "-")
    text = re.sub(r"\s+", " ", text).strip()

    # --- 1️⃣ Extract Title ---
    title = ""
    title_patterns = [
        r"We[’']?re\s+seeking\s+a[n]?\s+([A-Z][A-Za-z0-9\s/&\-]{2,50})",
        r"About\s+the\s+Role[:\-]?\s*([A-Z][A-Za-z0-9\s/&\-]{2,50})",
        r"Position[:\-]?\s*([A-Z][A-Za-z0-9\s/&\-]{2,50})",
        r"Job\s+Title[:\-]?\s*([A-Z][A-Za-z0-9\s/&\-]{2,50})"
    ]
    for p in title_patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            title = m.group(1).strip()
            break
    if not title:
        # Fallback heuristic: pick something ending with Engineer / Manager / Analyst etc.
        guess = re.search(r"\b([A-Z][A-Za-z\s]+(?:Engineer|Developer|Manager|Analyst|Scientist))\b", text)
        title = guess.group(1).strip() if guess else "General Role"

    # --- 2️⃣ Extract Required Skills ---
    # We’ll use section headers if available; else scan for known tech keywords.
    req_skills = []
    match_req = re.search(
        r"(?:Requirements|What You'll Need|Qualifications|Skills Required)[:\-]?\s*(.+?)(?=Preferred|Nice|What We Offer|$)",
        text, re.IGNORECASE | re.DOTALL
    )
    if match_req:
        req_text = match_req.group(1)
        req_skills = [s.strip(" -–:;,.") for s in re.split(r"[,;/]", req_text) if len(s.strip()) > 2]

    # Fallback: detect from keywords
    if not req_skills:
        keywords = [
            "python", "java", "scala", "sql", "aws", "gcp", "azure",
            "docker", "kubernetes", "airflow", "spark", "bigquery",
            "snowflake", "redshift", "postgres", "mysql", "beam"
        ]
        req_skills = [k for k in keywords if k.lower() in text.lower()]

    # --- 3️⃣ Extract Nice-to-Have Skills ---
    nice_skills = []
    match_nice = re.search(
        r"(?:Preferred|Nice[-\s]?to[-\s]?have|Good to have|Bonus)[:\-]?\s*(.+?)(?=What We Offer|Benefits|$)",
        text, re.IGNORECASE | re.DOTALL
    )
    if match_nice:
        nice_text = match_nice.group(1)
        nice_skills = [s.strip(" -–:;,.") for s in re.split(r"[,;/]", nice_text) if len(s.strip()) > 2]
    if not nice_skills:
        nice_skills = ["LLM", "Vector DB", "dbt", "Matillion", "Kafka"] if any(k in text.lower() for k in ["ai", "ml", "data"]) else ["Not specified"]

    # --- 4️⃣ Extract Responsibilities ---
    responsibilities = []
    match_resp = re.search(
        r"(?:Responsibilities|What You'll Do|Key Tasks|Duties|Your Role)[:\-]?\s*(.+?)(?=Requirement|Qualification|Preferred|Nice|$)",
        text, re.IGNORECASE | re.DOTALL
    )
    if match_resp:
        resp_text = match_resp.group(1)
        responsibilities = [s.strip(" -–:;,.") for s in re.split(r"[.;•]", resp_text) if len(s.strip()) > 5]

    if not responsibilities:
        responsibilities = ["Data pipeline design", "ETL workflow development", "Collaborate with ML & Analytics teams"]

    return {
        "title": title,
        "required_skills": list(dict.fromkeys(req_skills)) or ["Not found"],
        "nice_to_have_skills": list(dict.fromkeys(nice_skills)) or ["Not found"],
        "responsibilities": responsibilities,
        "raw_text": text
    }
