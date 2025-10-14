import os, json, requests
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")

def llm_match_groq(jd_text: str, resume_text: str, name: str) -> dict:
    """Compare resume and JD using Groq Llama 70B and return structured JSON."""
    system_prompt = """You are an objective technical recruiter assistant.
Compare a resume and a job description and return concise JSON with:
- 3–5 short bullet points on candidate fit
- 'fit_score' (1–10)
- 'concerns' if any"""

    user_prompt = f"""
JOB DESCRIPTION:
{jd_text}

RESUME ({name}):
{resume_text}

Respond strictly in JSON:
{{
  "bullets": ["..."],
  "fit_score": 7,
  "concerns": ["..."]
}}
"""

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.4,
        "response_format": {"type": "json_object"},
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        raw = data["choices"][0]["message"]["content"]

        # Handle rare cases of stringified JSON
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                parsed = {"bullets": [raw], "fit_score": 5, "concerns": []}
        else:
            parsed = raw

        # Normalize output
        bullets = parsed.get("bullets", [])
        fit = parsed.get("fit_score", 5)
        try:
            fit = float(fit)
        except ValueError:
            fit = 5.0
        fit = max(1.0, min(10.0, fit))  # clamp 1–10

        return {
            "bullets": bullets,
            "fit_score": fit,
            "concerns": parsed.get("concerns", []),
        }

    except Exception as e:
        print("Groq API Error:", e)
        return {"bullets": [], "fit_score": 5, "concerns": [str(e)]}
