# Intelligent Candidate Discovery & Ranking System

**India Runs Hackathon - Data & AI Challenge | Track 01 | Redrob AI x Hack2Skill**
**by Raja K C**

[![Python](https://img.shields.io/badge/Python-3.8+-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Why I Built This

Recruiters go through hundreds of profiles for a single role. Most tools just match keywords - they miss candidates who describe the same skill differently. I wanted to build something that actually understands the meaning behind a profile, not just the words. The goal was simple: help recruiters find the strongest candidates faster and analyse them better.

I built this in about 2-4 hours for the India Runs Hackathon.

---

## What It Does

You give it a job description and a list of candidate profiles. It ranks them by how well they actually fit the role - using AI embeddings, skill matching, and activity signals combined into one score.

It outputs three things:
- A ranked CSV file ready to submit
- A visual HTML report you can open in any browser
- A terminal breakdown showing exactly why each candidate ranked where they did

---

## Run It

```bash
git clone https://github.com/RAJA1404/hack2skill.git
cd hack2skill
pip install -r requirements.txt
python src/ranker.py
```

Show only top 3:
```bash
python src/ranker.py --top 3
```

No data files needed to test - demo data is built in. To use your own data, drop `job_description.json` and `candidates.json` into the `/data` folder.

---

## How the Scoring Works

I did not want to rely on just one signal. A candidate with a great profile but no real skills can fool a pure semantic system. So I combined three things:

| Signal | Weight | What it checks |
|---|---|---|
| Semantic Similarity | 55% | Meaning-level match using AI embeddings |
| Skill Match | 30% | Direct overlap with required and preferred skills |
| Activity Score | 15% | Platform engagement and behavioral signals |
| Experience Bonus | +5% | Small reward for meeting required years (score capped at 1.0) |

The semantic layer uses `all-MiniLM-L6-v2` from sentence-transformers. It converts both the job description and each candidate profile into 384-dimension vectors, then measures cosine similarity. This is how it catches things like "built ETL workflows" matching "data pipelines" - a keyword filter would miss that completely.

Skill matching handles real-world variations through a fuzzy alias system. `PySpark` correctly matches `Apache Spark`, `ML` matches `Machine Learning`, `Postgres` matches `PostgreSQL` and so on.

If sentence-transformers is not installed, the system automatically falls back to TF-IDF cosine similarity built from scratch - so it always runs, even in minimal environments.

---

## Sample Output

Terminal:
```text
-----------------------------------------------------------------
  TOP 5 CANDIDATES
-----------------------------------------------------------------
  # 1  Ananya Sharma          Score: 0.82  ################
       Matched : Python, SQL, Apache Spark, AWS, ETL
       Missing : Data Pipelines
       Preferred: Airflow

  # 2  Meera Iyer             Score: 0.80  ###############
       Matched : Python, SQL, Apache Spark, AWS, ETL
       Missing : Data Pipelines
       Preferred: Kafka, Airflow

  # 3  Priya Nair             Score: 0.79  ###############
       Matched : Python, SQL, Apache Spark
       Missing : AWS, ETL, Data Pipelines
       Preferred: Kafka, Machine Learning, GCP
-----------------------------------------------------------------
```

HTML report (`output/report.html`) — open in browser for a visual ranked table with score bars, matched skills in green, and missing skills in red.

---

## Project Structure

```text
hack2skill/
+-- src/
|   +-- ranker.py              <- main ranking script
+-- data/
|   +-- job_description.json   <- job description input
|   +-- candidates.json        <- candidate profiles
+-- output/
|   +-- ranked_candidates.csv  <- ranked CSV output
|   +-- report.html            <- visual HTML report
+-- requirements.txt
+-- README.md
```

---

## Input Format

**job_description.json**
```json
{
  "title": "Senior Data Engineer",
  "description": "Full role description...",
  "required_skills": ["Python", "SQL", "Spark"],
  "preferred_skills": ["Kafka", "Airflow"],
  "experience_years": 3
}
```

**candidates.json**
```json
[
  {
    "id": "C001",
    "name": "Candidate Name",
    "summary": "Brief profile summary...",
    "skills": ["Python", "SQL", "Spark"],
    "experience_years": 4,
    "activity_score": 75
  }
]
```

---

## Tech Stack

- Python 3.8+
- sentence-transformers - semantic embeddings
- all-MiniLM-L6-v2 - lightweight, fast, 80MB model
- scikit-learn - TF-IDF fallback
- pandas - CSV export

---

## About Me

**Raja K C** - CS student, building things that solve real problems.

- LinkedIn: [linkedin.com/in/raja-k-c-991b7a294](https://linkedin.com/in/raja-k-c-991b7a294)
- GitHub: [github.com/RAJA1404](https://github.com/RAJA1404)

*Submitted for India Runs Hackathon - Data & AI Challenge by Redrob AI x Hack2Skill*
