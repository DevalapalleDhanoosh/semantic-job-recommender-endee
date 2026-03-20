import pandas as pd
from sentence_transformers import SentenceTransformer
from endee import Endee, Precision
import time

# Connect to Endee
client = Endee()

# Create index
INDEX_NAME = "job_listings"
DIMENSION = 384

try:
    client.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        space_type="cosine",
        precision=Precision.INT8
    )
    print("Index created.")
except Exception as e:
    print(f"Index may already exist: {e}")

index = client.get_index(name=INDEX_NAME)

# Load dataset
print("Loading dataset...")
df = pd.read_csv("data/postings.csv", usecols=[
    "job_id", "title", "description", "location",
    "company_name", "formatted_experience_level", "remote_allowed"
])

# Clean
df = df.dropna(subset=["title", "description"])
df = df.head(5000)  # Use first 5000 for speed
df["text"] = df["title"] + " " + df["description"].str[:300]

# Embed
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding jobs (this takes a few minutes)...")
embeddings = model.encode(df["text"].tolist(), batch_size=64, show_progress_bar=True)

# Push to Endee in batches
print("Uploading to Endee...")
batch_size = 100
vectors = []

for i, (_, row) in enumerate(df.iterrows()):
    vectors.append({
        "id": str(row["job_id"]),
        "vector": embeddings[i].tolist(),
        "meta": {
            "title": str(row.get("title", "")),
            "company": str(row.get("company_name", "")),
            "location": str(row.get("location", "")),
            "experience": str(row.get("formatted_experience_level", "")),
            "remote": str(row.get("remote_allowed", "")),
            "description": str(row.get("description", ""))[:500]
        }
    })

    if len(vectors) == batch_size:
        index.upsert(vectors)
        print(f"Uploaded {i+1} jobs...")
        vectors = []

if vectors:
    index.upsert(vectors)

print(f"Done! {len(df)} jobs indexed in Endee.")