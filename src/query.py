from sentence_transformers import SentenceTransformer
from endee import Endee

model = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
client = Endee()
index = client.get_index(name="job_listings")

def search_jobs(user_query: str, top_k: int = 5):
    embedding = model.encode(user_query).tolist()
    results = index.query(vector=embedding, top_k=top_k)
    
    jobs = []
    for r in results:
        meta = r.get("meta", {}) if isinstance(r, dict) else r.meta
        jobs.append({
            "title": meta.get("title", "N/A"),
            "company": meta.get("company", "N/A"),
            "location": meta.get("location", "N/A"),
            "experience": meta.get("experience", "N/A"),
            "remote": meta.get("remote", "N/A"),
            "description": meta.get("description", "N/A"),
            "score": round(r.get("similarity", 0) if isinstance(r, dict) else r.similarity, 3)
        })
    return jobs

if __name__ == "__main__":
    query = "Python data science machine learning pandas sklearn"
    print(f"Searching for: {query}\n")
    results = search_jobs(query)
    for i, job in enumerate(results, 1):
        print(f"{i}. {job['title']} at {job['company']} ({job['location']})")
        print(f"   Score: {job['score']} | Experience: {job['experience']}")
        print()