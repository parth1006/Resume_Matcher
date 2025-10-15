# 🤖 AI Resume Matcher  
*An intelligent LLM-powered system for matching candidate resumes to job descriptions.*

---

## 🧭 Overview  

**AI Resume Matcher** automates the process of shortlisting candidates by analyzing resumes against job descriptions using a hybrid approach — **semantic embeddings** for similarity and **LLM reasoning** for interpretability.  

Recruiters can upload job descriptions and candidate resumes via a clean **Streamlit dashboard**, while a **FastAPI backend** handles text extraction, embedding, and scoring.  
The system produces both a **numerical match score** and a **recruiter-style justification** summarizing candidate fit.

---

## 🏗️ System Architecture  

    ┌──────────────────────┐
    │   Resume (PDF)       │
    └──────────┬───────────┘
               │ Text Extraction
               ▼
        ┌──────────────┐
        │ Parser Layer │ ← PDF → Text
        └──────┬───────┘
               │
               ▼
    ┌──────────────────────────┐
    │ Embedding Model (MiniLM) │
    └──────────┬───────────────┘
               │
               ▼
    ┌─────────────────────────────┐
    │   Similarity Scoring Logic  │
    └──────────┬──────────────────┘
               │
               ▼
    ┌──────────────────────────────┐
    │ LLM (Groq) — Fit Evaluation │
    └──────────┬──────────────────┘
               │
               ▼
      Streamlit Dashboard (UI)

---

## ⚙️ Tech Stack  

| Layer | Technology |
|-------|-------------|
| Backend | FastAPI |
| Frontend | Streamlit |
| Embedding Model | `sentence-transformers/all-MiniLM-L6-v2` |
| LLM | Groq API |
| Database | SQLite |
| Containerization | Docker |
| Language | Python 3.10 |

---


---

## 💡 Key Features  

✅ Resume parsing (PDF → Text extraction)  
✅ Embedding-based semantic similarity using SentenceTransformers  
✅ Groq-powered LLM scoring and justification  
✅ Recruiter-style insights with bullet points & fit score  
✅ Modular architecture (FastAPI + Streamlit separation)  
✅ Secure `.env` handling  
✅ Dockerized for easy deployment  

---

## 🧠 LLM Prompt Design  

**System Prompt:**
```python
SYSTEM_PROMPT = """You are an expert technical recruiter and hiring evaluator.

Your goal: analyze a candidate's resume for their fit to a specific job description.

EVALUATION FRAMEWORK:
1. Required Skills Match — Do they possess the must-have technical and soft skills?
2. Experience Level — Is their seniority (years, role type) aligned with the position?
3. Domain Relevance — Are their projects, industries, or technologies relevant?
4. Accomplishments — Do they demonstrate measurable, outcome-based impact?

SCORING GUIDE (fit_score 1-10):
1–3 → Poor fit (missing core requirements)
4–5 → Weak fit (partial match, lacks key experience)
6–7 → Moderate fit (meets many, gaps remain)
8–9 → Strong fit (solid alignment, relevant experience)
10 → Exceptional fit (direct and deep match across all aspects)

OUTPUT FORMAT (STRICT JSON):
{
  "summary_bullets": ["Concise factual statements (3–5)"],
  "fit_score": <integer 1–10>,
  "key_strengths": ["Specific strengths (2–3)"],
  "concerns": ["Specific gaps (0–3)"],
  "reasoning": "1–2 sentences explaining the score"
}

Guidelines:
- Be objective and evidence-based (no bias, no speculation).
- Focus on what the candidate *has*, not what’s missing.
- Output ONLY valid JSON—no markdown, text, or explanations."""


  
USER_TEMPLATE = """JOB DESCRIPTION:
{jd}

CANDIDATE RESUME:
Name: {name}
{resume}

Analyze the candidate-job fit and respond strictly in the required JSON schema."""
```
This structured prompting ensures deterministic JSON responses and concise recruiter-like summaries.

---
## ⚙️ Installation
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/parth1006/Resume_Matcher.git
cd Resume_Matcher
```
### 2️⃣ Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate       # Linux/Mac
venv\Scripts\activate          # Windows
```
### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
### 4️⃣ Configure Environment Variables
```bash
GROQ_API_KEY=<your_api_key_here>
MODEL_NAME=llama-3.3-70b-versatile
```

---
## Run the Dashboard
```bash
streamlit run ui/dashboard.py
```
---
## 🎥 Demo Video  

A short walkthrough demonstrating the complete workflow — from uploading resumes and job descriptions to generating match scores and recruiter-style insights.

📹 **Watch the Demo Video:**  
👉 [Google Drive Link – Click Here]([https://drive.google.com/placeholder](https://drive.google.com/file/d/1GOPbBV51icFYKaDzHtzgTprXR0XeMuZQ/view?usp=sharing))

---

## 👨‍💻 Author  

**Parth Maheshwari**  
Final Year B.Tech CSE @ VIT Vellore  

🔗 [LinkedIn](https://www.linkedin.com/in/parth1006) • [GitHub](https://github.com/parth1006)

---

