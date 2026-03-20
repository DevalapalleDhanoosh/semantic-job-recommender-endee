\#  Semantic Job Recommendation Engine



An AI-powered job recommendation system that uses \*\*vector similarity search\*\* via \[Endee](https://github.com/endee-io/endee) to semantically match candidates with relevant job listings — going beyond keyword matching to understand the \*meaning\* of skills and job descriptions.



\---



\##  Problem Statement



Traditional job search relies on keyword matching, which fails when:

\- A candidate writes "data analysis" but a job says "analytical insights"

\- Skills are described differently across resumes and job postings



This system solves that by converting both skills and job descriptions into \*\*semantic embeddings\*\* and finding the closest matches using vector search.



\---



\##  System Design

```

User Skills Input

&#x20;      │

&#x20;      ▼

Sentence Transformer (all-MiniLM-L6-v2)

&#x20;      │ generates 384-dim embedding

&#x20;      ▼

Endee Vector DB ──── cosine similarity search

&#x20;      │ returns top-K matching job vectors

&#x20;      ▼

Job Metadata (title, company, location, experience)

&#x20;      │

&#x20;      ▼

Groq LLM (llama-3.3-70b) ──── generates personalized summary

&#x20;      │

&#x20;      ▼

Streamlit UI

```



\---



\## 🔧 How Endee is Used



Endee serves as the core vector database powering semantic search:



\- \*\*Index Creation\*\*: A cosine similarity index with 384 dimensions is created in Endee

\- \*\*Vector Upsert\*\*: 5,000 LinkedIn job postings are embedded and stored in Endee with metadata (title, company, location, experience level)

\- \*\*Semantic Query\*\*: User's skills are embedded and queried against Endee to retrieve the top-K most semantically similar jobs using approximate nearest neighbor (ANN) search



\---



\##  Setup \& Installation



\### Prerequisites

\- Python 3.10+

\- Docker Desktop



\### 1. Clone the repository

```bash

git clone https://github.com/DevalapalleDhanoosh/semantic-job-recommender-endee.git

cd semantic-job-recommender-endee

```



\### 2. Start Endee vector database

```bash

docker compose up -d

```



\### 3. Create virtual environment

```bash

python -m venv venv

venv\\Scripts\\activate  # Windows

source venv/bin/activate  # Mac/Linux

```



\### 4. Install dependencies

```bash

pip install endee sentence-transformers pandas requests streamlit groq kaggle

```



\### 5. Download the dataset

```bash

kaggle datasets download -d arshkon/linkedin-job-postings -p data/ --unzip

```



\### 6. Set environment variables

Create a `.env` file:

```

GROQ\_API\_KEY=your\_groq\_api\_key\_here

```



\### 7. Index the jobs into Endee

```bash

python src/embed\_and\_index.py

```



\### 8. Run the app

```bash

streamlit run app.py

```



Open `http://localhost:8501` in your browser.



\---



\##  Project Structure

```

semantic-job-recommender-endee/

├── app.py                  # Streamlit UI + Groq AI summary

├── docker-compose.yml      # Endee vector DB setup

├── src/

│   ├── embed\_and\_index.py  # Embed jobs and push to Endee

│   └── query.py            # Query Endee for similar jobs

├── data/                   # LinkedIn job postings dataset

├── .gitignore

└── README.md

```



\---



\##  Tech Stack



| Component | Technology |

|---|---|

| Vector Database | Endee |

| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |

| LLM Summary | Groq (llama-3.3-70b-versatile) |

| Dataset | LinkedIn Job Postings (Kaggle) |

| UI | Streamlit |

| Language | Python |



\---



\##  Features



\-  \*\*Semantic Search\*\* — understands meaning, not just keywords

\-  \*\*AI Summary\*\* — personalized explanation of why each job fits

\-  \*\*Similarity Scores\*\* — ranked results with cosine similarity scores

\-  \*\*Adjustable Results\*\* — choose 3–10 job recommendations

\-  \*\*Remote Filter\*\* — see remote availability per job



\---



\##  Dataset



\[LinkedIn Job Postings](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings) by Arsh Kon on Kaggle (CC-BY-SA-4.0)

