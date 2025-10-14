import os, json, requests
from dotenv import load_dotenv
from matching.prompts import USER_TEMPLATE, SYSTEM_PROMPT

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")

def llm_match_groq(jd_text: str, resume_text: str, name: str) -> dict:
    """Calls Groq LLM to evaluate resume-job fit and return structured JSON."""

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(jd=jd_text, name=name, resume=resume_text)}
        ],
        "temperature": 0.3,
        "response_format": {"type": "json_object"},
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=90)
        response.raise_for_status()
        data = response.json()

        raw_content = data["choices"][0]["message"]["content"]
        if isinstance(raw_content, str):
            parsed = json.loads(raw_content)
        else:
            parsed = raw_content

        # --- Normalize output ---
        return {
            "summary_bullets": parsed.get("summary_bullets", []),
            "fit_score": float(parsed.get("fit_score", 5)),
            "key_strengths": parsed.get("key_strengths", []),
            "concerns": parsed.get("concerns", []),
            "reasoning": parsed.get("reasoning", "")
        }

    except Exception as e:
        print("Groq API Error:", e)
        return {
            "summary_bullets": [],
            "fit_score": 5.0,
            "key_strengths": [],
            "concerns": [str(e)],
            "reasoning": "Error during LLM evaluation."
        }
