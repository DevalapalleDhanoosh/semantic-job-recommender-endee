import streamlit as st
from sentence_transformers import SentenceTransformer
from endee import Endee
import google.generativeai as genai

# Config
import os
GEMINI_API_KEY = os.environ.get("GROQ_API_KEY", "")

# Load model and index
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)

@st.cache_resource
def load_index():
    client = Endee()
    return client.get_index(name="job_listings")

def search_jobs(query, top_k=5):
    model = load_model()
    index = load_index()
    embedding = model.encode(query).tolist()
    results = index.query(vector=embedding, top_k=top_k)
    jobs = []
    for r in results:
        meta = r.get("meta", {}) if isinstance(r, dict) else r.meta
        score = r.get("similarity", 0) if isinstance(r, dict) else r.similarity
        jobs.append({
            "title": meta.get("title", "N/A"),
            "company": meta.get("company", "N/A"),
            "location": meta.get("location", "N/A"),
            "experience": meta.get("experience", "N/A"),
            "remote": meta.get("remote", "N/A"),
            "description": meta.get("description", "N/A"),
            "score": round(score, 3)
        })
    return jobs

def generate_summary(jobs, user_skills):
    from groq import Groq
    client = Groq(api_key=GEMINI_API_KEY)
    job_list = "\n".join([f"- {j['title']} at {j['company']} ({j['location']})" for j in jobs])
    prompt = f"""A candidate has these skills: {user_skills}

These are the top matching jobs found:
{job_list}

Write a short 3-4 sentence personalized summary explaining why these jobs are a good fit for the candidate."""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
# UI
st.set_page_config(page_title="Job Recommender", page_icon="💼", layout="wide")
st.title("💼 Semantic Job Recommendation Engine")
st.markdown("Powered by **Endee Vector DB** + **Sentence Transformers** + **Gemini AI**")
st.divider()

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Your Profile")
    user_skills = st.text_area(
        "Enter your skills and experience",
        placeholder="e.g. Python, machine learning, data analysis, pandas, scikit-learn, SQL",
        height=150
    )
    job_title = st.text_input("Desired job title (optional)", placeholder="e.g. Data Scientist")
    top_k = st.slider("Number of results", min_value=3, max_value=10, value=5)
    search_btn = st.button("🔍 Find Jobs", type="primary", use_container_width=True)

with col2:
    if search_btn and user_skills:
        query = f"{job_title} {user_skills}" if job_title else user_skills
        
        with st.spinner("Searching jobs..."):
            jobs = search_jobs(query, top_k=top_k)
        
        st.subheader("🤖 AI Summary")
        with st.spinner("Generating personalized summary..."):
            summary = generate_summary(jobs, user_skills)
        st.info(summary)
        
        st.subheader(f"Top {len(jobs)} Matching Jobs")
        for i, job in enumerate(jobs, 1):
            with st.expander(f"#{i} {job['title']} at {job['company']} — Score: {job['score']}"):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.write(f"📍 **Location:** {job['location']}")
                    st.write(f"🎯 **Experience:** {job['experience']}")
                with col_b:
                    st.write(f"🏠 **Remote:** {job['remote']}")
                st.write(f"📝 **Description:** {job['description'][:300]}...")
    
    elif search_btn and not user_skills:
        st.warning("Please enter your skills first.")
    else:
        st.markdown("### 👈 Enter your skills and click Find Jobs")
        st.markdown("This app uses **vector similarity search** via Endee to find jobs that semantically match your skills — not just keyword matching.")