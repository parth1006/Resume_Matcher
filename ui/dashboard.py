# ui/dashboard.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import streamlit as st
import requests
import os
import pandas as pd
from parsers.jd_extract import extract_jd_details
from parsers.pdf import pdf_to_text
# -------------------- CONFIG --------------------
API_URL = os.getenv("API_URL", "https://parth4384-ai-resume-matcher-api.hf.space")
st.set_page_config(page_title="AI Resume Matcher", page_icon="🧠", layout="wide")
st.title("🤖 AI-Powered Resume-JD Matcher")

st.markdown(
    "Upload candidate resumes and compare them with a Job Description to see semantic match scores, "
    "skill coverage, and model-generated justifications."
)

# -------------------- SESSION STATE --------------------
if "api_url" not in st.session_state:
    st.session_state.api_url = API_URL

# Persist the most recent parsed candidate so Tab 1 view survives tab switches / reruns
if "last_candidate" not in st.session_state:
    st.session_state.last_candidate = None

# Persist last match results (optional; does NOT affect Tab 1)
if "last_match_results" not in st.session_state:
    st.session_state.last_match_results = None

# Cache for jobs list
if "job_list_cache" not in st.session_state:
    st.session_state.job_list_cache = []

# -------------------- TABS --------------------
tab1, tab2, tab3 = st.tabs(["📤 Upload Resume", "🧾 Add Job Description", "📊 Match Candidates"])

# ==================== TAB 1: Upload Resume ====================
with tab1:
    st.subheader("Upload Candidate Resume")

    with st.form("upload_form", clear_on_submit=False):
        resume_file = st.file_uploader("Upload Resume (PDF, DOCX, or TXT)", type=["pdf", "docx", "txt"])
        submitted = st.form_submit_button("Upload & Parse Resume")

    if submitted:
        if not resume_file:
            st.warning("Please upload a resume first.")
        else:
            files = {"resume": (resume_file.name, resume_file, resume_file.type)}
            with st.spinner("⏳ Uploading and parsing resume..."):
                try:
                    r = requests.post(f"{st.session_state.api_url}/candidates/upload", files=files, timeout=90)
                    r.raise_for_status()
                except requests.exceptions.RequestException as e:
                    st.error(f"❌ Connection error: {e}")
                    st.stop()

            cand = r.json()
            st.session_state.last_candidate = cand
            st.success("✅ Resume uploaded and parsed successfully!")

    cand = st.session_state.get("last_candidate")
    if cand:
        with st.container(border=True):
            st.markdown("### 🧾 Extracted Details:")

            # --- Basic Info ---
            st.markdown(f"**👤 Name:** {cand.get('name', '—')}")
            st.markdown(f"**📧 Email:** {', '.join(cand.get('email', [])) or '—'}")
            st.markdown(f"**📱 Phone:** {', '.join(cand.get('phone_numbers', [])) or '—'}")

            # --- Skills ---
            skills = cand.get("skills", [])
            if skills:
                st.markdown(f"**🧠 Skills:** {', '.join(skills)}")
            else:
                st.markdown("**🧠 Skills:** —")

            # --- Experience ---
            total_exp = cand.get("total_experience_years") or cand.get("experience_details", {}).get("total_years")
            st.markdown(f"**⏱️ Experience (Years):** {total_exp if total_exp is not None else '—'}")

            # --- Work History ---
            work_history = cand.get("work_history", [])
            if work_history:
                st.markdown("**🏢 Work History:**")
                for job in work_history:
                    designation = job.get("designation", "N/A")
                    company = job.get("company", "N/A")
                    duration = job.get("duration", "")
                    st.markdown(f"- **{designation}**, *{company}*")
                    if duration:
                        st.caption(f"⏰ {duration}")
                    if job.get("location"):
                        st.caption(f"📍 {job['location']}")
                    if job.get("highlights"):
                        with st.expander("🔹 Highlights", expanded=False):
                            for h in job["highlights"]:
                                st.markdown(f"• {h}")
            else:
                st.markdown("**🏢 Work History:** —")

            # --- Education ---
            education_list = cand.get("education", [])
            if education_list:
                st.markdown("**🎓 Education:**")
                for edu in education_list:
                    degree = edu.get("degree", "N/A")
                    field = edu.get("field", "")
                    inst = edu.get("institution", "")
                    year = edu.get("year", "")
                    grade = edu.get("grade", "")
                    line = f"- **{degree}**"
                    if field:
                        line += f" in {field}"
                    if inst:
                        line += f", *{inst}*"
                    if year:
                        line += f" ({year})"
                    st.markdown(line)
                    if grade:
                        st.caption(f"📊 Grade: {grade}")
            else:
                st.markdown("**🎓 Education:** —")


# ==================== TAB 2: Add Job Description ====================
with tab2:
    st.subheader("Add Job Description")

    # JD PDF upload option
    jd_file = st.file_uploader("📄 Upload Job Description (PDF, optional)", type=["pdf"])

    # Initialize empty variables
    title = ""
    jd_text = ""
    req_skills = ""
    nice_skills = ""

    # --- Extract JD details if PDF uploaded ---
    if jd_file:
        with st.spinner("Extracting details from uploaded JD..."):
            try:
                # Convert PDF to text and extract structured details
                text = pdf_to_text(jd_file)
                extracted = extract_jd_details(text)

                title = extracted.get("title", "")
                jd_text = extracted.get("raw_text", "")

                st.success("✅ JD parsed successfully! You can review or edit the fields below.")
            except Exception as e:
                st.error(f"❌ Failed to parse JD: {e}")

    # --- Manual or pre-filled JD form ---
    with st.form("job_form"):
        title = st.text_input("Job Title", value=title)
        jd_text = st.text_area("Paste or Edit Job Description", value=jd_text, height=200)
        req_skills = st.text_input("Required Skills (comma-separated)", value=req_skills)
        nice_skills = st.text_input("Nice-to-have Skills (comma-separated)", value=nice_skills)
        submitted_jd = st.form_submit_button("Create Job")

    # --- On submission ---
    if submitted_jd:
        payload = {
            "title": title.strip(),
            "jd_text": jd_text.strip(),
            "required_skills": [s.strip() for s in req_skills.split(",") if s.strip()],
            "nice_to_have_skills": [s.strip() for s in nice_skills.split(",") if s.strip()],
        }

        with st.spinner("Creating job entry..."):
            try:
                r = requests.post(f"{st.session_state.api_url}/jobs", json=payload, timeout=60)
            except requests.exceptions.RequestException as e:
                st.error(f"Connection error: {e}")
                st.stop()

        if r.status_code == 200:
            job = r.json()
            st.success("✅ Job added successfully!")

            # Extract values safely
            job_id = job.get("id") or job.get("job_id")
            title = job.get("title", "(No title returned)")
            req = ", ".join(job.get("required_skills", [])) or "—"
            nice = ", ".join(job.get("nice_to_have_skills", [])) or "—"
            jd_preview = job.get("jd_text", "")

            # Display summary cleanly
            st.markdown("### 📁 Job Summary")
            if job_id:
                st.markdown(f"**Job ID:** {job_id}")
            st.markdown(f"**Title:** {title}")

            if jd_preview:
                with st.expander("📜 View Full Job Description"):
                    st.write(jd_preview)

            # Refresh job list cache for Tab 3
            st.session_state.job_list_cache = []
        else:
            st.error(f"❌ Job creation failed: {r.text}")

# ==================== TAB 3: Match Candidates ====================
with tab3:
    st.subheader("Match Candidates to a Job Description")
    with st.spinner("Fetching available jobs..."):
        try:
            if not st.session_state.job_list_cache:
                resp = requests.get(f"{st.session_state.api_url}/jobs/list", timeout=30)
                st.session_state.job_list_cache = resp.json() if resp.status_code == 200 else []
            job_list = st.session_state.job_list_cache
        except requests.exceptions.RequestException:
            job_list = []

    if not job_list:
        st.warning("⚠️ No jobs found. Please add a job in the previous tab first.")
    else:
        job_options = {f"{j['id']} – {j['title']}": j["id"] for j in job_list}
        selected_job = st.selectbox("Select Job", options=list(job_options.keys()))
        selected_job_id = job_options[selected_job]

        top_k = st.slider("Number of Top Candidates", 1, 10, 5)

        if st.button("🔍 Run Matching"):
            with st.spinner("Matching candidates..."):
                try:
                    r = requests.get(f"{API_URL}/match/{selected_job_id}", params={"top_k": top_k})
                except requests.exceptions.RequestException as e:
                    st.error(f"Connection error: {e}")
                    st.stop()

            if r.status_code == 200:
                results = r.json()
                if not results:
                    st.warning("No candidates found for this job yet.")
                else:
                    for res in results:
                        with st.expander(f"🧑 {res['candidate_name']} — {res['score']}% match"):
                            # === New recruiter-friendly view ===
                            st.markdown("### 🧩 Score Breakdown")
                            comps = res["components"]
                            score_table = pd.DataFrame({
                                "Metric": [
                                    "Textual Match (Resume ↔ JD)",
                                    "Required Skills Coverage",
                                    "Preferred Skills Coverage",
                                    "LLM-Based Fit Score",
                                    "Final Suitability Score"
                                ],
                                "Score (%)": [
                                    f"{comps.get('similarity', 0)*100:.2f}",
                                    f"{comps.get('req_cover', 0)*100:.2f}",
                                    f"{comps.get('nice_cover', 0)*100:.2f}",
                                    f"{comps.get('llm', 0)*100:.2f}",
                                    f"{res['score']:.2f}"
                                ]
                            })

                            st.table(score_table)
                            st.markdown(
                                f"**Final Suitability:** {res['score']:.2f}% — "
                                "this combines all metrics above for a balanced overall fit score."
                            )

                            st.markdown("### 📝 Justification")
                            justification = res["justification"]
                            lines = justification.split("\n")
                            current_section = None
                            for line in lines:
                                line = line.strip()
                                if not line:
                                    continue
                                if line.startswith("**") and line.endswith("**:"):
                                    # Section title (Summary, Strengths, Concerns)
                                    current_section = line.strip("*:").strip()
                                    st.markdown(f"#### {current_section}")
                                elif line.startswith("**Reasoning:**"):
                                    st.markdown(f"🧠 {line.replace('**Reasoning:**', '').strip()}")
                                elif line.startswith("- "):
                                    st.markdown(line)
                                else:
                                    st.markdown(line)
            else:
                st.error(f"Match request failed: {r.text}")