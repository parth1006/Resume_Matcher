import streamlit as st
import requests
import os
import pandas as pd
# -------------------- CONFIG --------------------
API_URL = os.getenv("API_URL", "https://parth4384-ai-resume-matcher-api.hf.space")

st.set_page_config(
    page_title="AI Resume Matcher",
    page_icon="🧠",
    layout="wide",
)
st.title("🤖 AI-Powered Resume-JD Matcher")

st.markdown("""
Upload candidate resumes and compare them with a Job Description to see semantic match scores, 
skill coverage, and model-generated justifications.
""")

# -------------------- SIDEBAR --------------------
api_url=API_URL
# -------------------- TABS --------------------
tab1, tab2, tab3 = st.tabs(["📤 Upload Resume", "🧾 Add Job Description", "📊 Match Candidates"])

# -------------------- TAB 1: Upload Resume --------------------
with tab1:
    st.subheader("Upload Candidate Resume")

    with st.form("upload_form"):
        resume_file = st.file_uploader("Upload Resume (PDF or TXT)", type=["pdf", "txt"])
        submitted = st.form_submit_button("Upload & Parse Resume")

    if submitted:
        if not resume_file:
            st.warning("⚠️ Please select a resume file first.")
        else:
            files = {"resume": (resume_file.name, resume_file, resume_file.type)}
            with st.spinner("Uploading and parsing resume..."):
                try:
                    r = requests.post(f"{api_url}/candidates/upload", files=files)
                except requests.exceptions.RequestException as e:
                    st.error(f"Connection error: {e}")
                    st.stop()

            if r.status_code == 200:
                st.success("✅ Resume uploaded and parsed successfully!")
                candidate = r.json()
                st.write("### Extracted Details:")
                st.markdown(f"- **Name:** {candidate.get('name', 'Not Found')}")
                st.markdown(f"- **Skills:** {', '.join(candidate.get('skills', [])) or 'Not Found'}")
                st.markdown(f"- **Experience (Years):** {candidate.get('experience_years', 0)}")
                st.markdown(f"- **Education:** {', '.join(candidate.get('education', [])) or 'Not Found'}")
                st.info("Details automatically extracted from the uploaded resume.")
            else:
                st.error(f"Upload failed: {r.text}")

# -------------------- TAB 2: Add Job Description --------------------
with tab2:
    st.subheader("Add Job Description")
    with st.form("job_form"):
        title = st.text_input("Job Title")
        jd_text = st.text_area("Paste Job Description", height=200)
        req_skills = st.text_input("Required Skills (comma-separated)")
        nice_skills = st.text_input("Nice-to-have Skills (comma-separated)")
        submitted_jd = st.form_submit_button("Create Job")

    if submitted_jd:
        if not title or not jd_text:
            st.warning("⚠️ Please enter both title and job description.")
        else:
            payload = {
                "title": title,
                "jd_text": jd_text,
                "required_skills": [s.strip() for s in req_skills.split(",") if s.strip()],
                "nice_to_have_skills": [s.strip() for s in nice_skills.split(",") if s.strip()]
            }
            with st.spinner("Creating job entry..."):
                r = requests.post(f"{api_url}/jobs", json=payload)
            if r.status_code == 200:
                st.success("✅ Job added successfully!")
                job_info = r.json()
                st.write("### Job Details:")
                st.markdown(f"- **Job ID:** {job_info.get('job_id', 'N/A')}")
                st.markdown(f"- **Title:** {job_info.get('title', 'N/A')}")
                st.info("The job is now stored in the system and available for matching.")
            else:
                st.error(f"Job creation failed: {r.text}")


# -------------------- TAB 3: Match Candidates --------------------
with tab3:
    st.subheader("Match Candidates to a Job Description")

    # Fetch jobs dynamically
    with st.spinner("Fetching available jobs..."):
        try:
            job_list_resp = requests.get(f"{api_url}/jobs/list")
            job_list = job_list_resp.json() if job_list_resp.status_code == 200 else []
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
                    r = requests.get(f"{api_url}/match/{selected_job_id}", params={"top_k": top_k})
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

                            # Build a DataFrame for clear display
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
                            # Add narrative summary below the table
                            st.markdown(
                                f"**Final Suitability:** {res['score']:.2f}% — "
                                "this combines all metrics above for a balanced overall fit score."
                            )

                            st.markdown("### 📝 Justification")
                            justification = res["justification"]
                            # Split on common bullet or period patterns for line separation
                            lines = [line.strip("• ").strip() for line in justification.replace("•", "\n•").split("\n") if line.strip()]
                            # Rebuild markdown-friendly bullet list
                            formatted = "\n".join([f"- {line}" for line in lines])
                            st.markdown(formatted)

            else:
                st.error(f"Match request failed: {r.text}")
