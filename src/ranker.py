"""
India Runs - Data & AI Challenge
Intelligent Candidate Discovery & Ranking System
Author: Raja K C
GitHub: https://github.com/RAJA1404/hack2skill
"""

import json
import csv
import math
import re
import sys
import argparse
from pathlib import Path

# -- Try to import AI libraries; fall back to TF-IDF if not installed ----------
try:
    from sentence_transformers import SentenceTransformer, util
    USE_SEMANTIC = True
    print("sentence-transformers found - using semantic embeddings")
except ImportError:
    USE_SEMANTIC = False
    print("sentence-transformers not found - using TF-IDF fallback")

# -- CONFIG --------------------------------------------------------------------
DATA_DIR   = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

JOB_DESC_FILE   = DATA_DIR / "job_description.json"
CANDIDATES_FILE = DATA_DIR / "candidates.json"
OUTPUT_FILE     = OUTPUT_DIR / "ranked_candidates.csv"
HTML_FILE       = OUTPUT_DIR / "report.html"

WEIGHT_SEMANTIC = 0.55
WEIGHT_SKILLS   = 0.30
WEIGHT_ACTIVITY = 0.15

SKILL_ALIASES = {
    "apache spark": "spark",
    "pyspark": "spark",
    "spark": "spark",
    "ml": "machine learning",
    "gcs": "gcp",
    "google cloud": "gcp",
    "postgres": "postgresql",
    "js": "javascript",
    "node": "node.js",
}

# -- STEP 1: LOAD DATA ---------------------------------------------------------

def load_job_description():
    if not JOB_DESC_FILE.exists():
        print("No job_description.json found in /data - using demo JD")
        return {
            "title": "Senior Data Engineer",
            "description": (
                "We are looking for a Senior Data Engineer to build and maintain "
                "scalable data pipelines. The role requires strong Python skills, "
                "experience with SQL, Apache Spark, and cloud platforms like AWS or GCP. "
                "The candidate should have 3+ years of experience in data engineering, "
                "understand ETL processes, and be comfortable working in agile teams. "
                "Bonus: experience with ML pipelines, Kafka, or Airflow."
            ),
            "required_skills": ["Python", "SQL", "Apache Spark", "AWS", "ETL", "Data Pipelines"],
            "preferred_skills": ["Kafka", "Airflow", "Machine Learning", "GCP"],
            "experience_years": 3
        }
    with open(JOB_DESC_FILE, "r", encoding="utf-8") as f:
        ext = JOB_DESC_FILE.suffix.lower()
        if ext == ".json":
            return json.load(f)
        else:
            return {"title": "Role", "description": f.read(),
                    "required_skills": [], "preferred_skills": [], "experience_years": 0}


def load_candidates():
    if not CANDIDATES_FILE.exists():
        print("No candidates.json found in /data - using demo candidates")
        return [
            {"id": "C001", "name": "Ananya Sharma",
             "summary": "Python data engineer with 5 years building ETL pipelines on AWS and Spark.",
             "skills": ["Python", "Apache Spark", "AWS", "SQL", "ETL", "Airflow"],
             "experience_years": 5, "activity_score": 85},
            {"id": "C002", "name": "Rahul Verma",
             "summary": "Backend developer with 2 years of Python and SQL experience. Learning cloud.",
             "skills": ["Python", "SQL", "Django", "PostgreSQL"],
             "experience_years": 2, "activity_score": 60},
            {"id": "C003", "name": "Priya Nair",
             "summary": "Data scientist experienced in ML pipelines, PySpark, and GCP infrastructure.",
             "skills": ["Python", "PySpark", "GCP", "Machine Learning", "SQL", "Kafka"],
             "experience_years": 4, "activity_score": 92},
            {"id": "C004", "name": "Arun Kumar",
             "summary": "DevOps engineer comfortable with cloud deployments and scripting.",
             "skills": ["AWS", "Docker", "Bash", "Terraform"],
             "experience_years": 3, "activity_score": 55},
            {"id": "C005", "name": "Meera Iyer",
             "summary": "Senior data engineer specializing in real-time streaming with Kafka and Spark.",
             "skills": ["Apache Spark", "Kafka", "Python", "SQL", "AWS", "Airflow", "ETL"],
             "experience_years": 6, "activity_score": 78},
            {"id": "C006", "name": "Vikram Singh",
             "summary": "Fresh graduate with Python and SQL knowledge from academic projects.",
             "skills": ["Python", "SQL", "pandas"],
             "experience_years": 0, "activity_score": 40},
        ]
    ext = CANDIDATES_FILE.suffix.lower()
    with open(CANDIDATES_FILE, "r", encoding="utf-8") as f:
        if ext == ".json":
            return json.load(f)
        elif ext == ".csv":
            return list(csv.DictReader(f))
    return []


# -- STEP 2: BUILD TEXT PROFILES -----------------------------------------------

def parse_skills(raw_skills) -> list:
    if not raw_skills:
        return []
    if isinstance(raw_skills, list):
        return [str(s).strip() for s in raw_skills if str(s).strip()]
    return [s.strip() for s in str(raw_skills).split(",") if s.strip()]


def normalize_skill(skill: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", str(skill).lower()).strip()
    return SKILL_ALIASES.get(normalized, normalized)


def skills_match(candidate_skill: str, target_skill: str) -> bool:
    cn = normalize_skill(candidate_skill)
    tn = normalize_skill(target_skill)
    if cn == tn:
        return True
    if cn in tn or tn in cn:
        return True
    ct = set(cn.split())
    tt = set(tn.split())
    return bool(ct and tt and ct & tt)


def count_skill_matches(target_skills: list, candidate_skills: list) -> int:
    return sum(
        1 for ts in target_skills
        if any(skills_match(cs, ts) for cs in candidate_skills)
    )


def get_matched_missing(candidate: dict, jd: dict):
    """Return (matched_required, missing_required, matched_preferred) skill lists."""
    required  = parse_skills(jd.get("required_skills", []))
    preferred = parse_skills(jd.get("preferred_skills", []))
    c_skills  = parse_skills(candidate.get("skills", []))

    matched_req  = [s for s in required  if any(skills_match(cs, s) for cs in c_skills)]
    missing_req  = [s for s in required  if not any(skills_match(cs, s) for cs in c_skills)]
    matched_pref = [s for s in preferred if any(skills_match(cs, s) for cs in c_skills)]
    return matched_req, missing_req, matched_pref


def build_candidate_text(candidate: dict) -> str:
    parts = []
    if candidate.get("summary"):
        parts.append(candidate["summary"])
    skills = parse_skills(candidate.get("skills", []))
    if skills:
        parts.append("Skills: " + ", ".join(skills))
    if candidate.get("experience"):
        parts.append(str(candidate["experience"]))
    if candidate.get("bio"):
        parts.append(candidate["bio"])
    return " | ".join(parts) if parts else candidate.get("name", "")


def build_job_text(jd: dict) -> str:
    parts = [jd.get("description", "")]
    if jd.get("required_skills"):
        parts.append("Required: " + ", ".join(jd["required_skills"]))
    if jd.get("preferred_skills"):
        parts.append("Preferred: " + ", ".join(jd["preferred_skills"]))
    if jd.get("title"):
        parts.insert(0, "Role: " + jd["title"])
    return " | ".join(parts)


# -- STEP 3: SCORING -----------------------------------------------------------

def compute_semantic_scores(jd_text: str, candidate_texts: list) -> list:
    model = SentenceTransformer("all-MiniLM-L6-v2")
    jd_emb   = model.encode(jd_text, convert_to_tensor=True)
    c_embs   = model.encode(candidate_texts, convert_to_tensor=True)
    scores   = util.cos_sim(jd_emb, c_embs)[0]
    return [float(s) for s in scores]


def tfidf_similarity(jd_text: str, candidate_texts: list) -> list:
    def tokenize(text):
        return re.findall(r'\w+', text.lower())
    def tf(tokens):
        freq = {}
        for t in tokens:
            freq[t] = freq.get(t, 0) + 1
        total = len(tokens) or 1
        return {t: c / total for t, c in freq.items()}
    def idf(docs):
        N = len(docs)
        df = {}
        for doc in docs:
            for t in set(doc):
                df[t] = df.get(t, 0) + 1
        return {t: math.log((N + 1) / (c + 1)) + 1 for t, c in df.items()}
    def tfidf_vec(tf_d, idf_d):
        return {t: tf_d.get(t, 0) * idf_d.get(t, 0) for t in idf_d}
    def cosine(v1, v2):
        keys = set(v1) | set(v2)
        dot  = sum(v1.get(k, 0) * v2.get(k, 0) for k in keys)
        m1   = math.sqrt(sum(x**2 for x in v1.values())) or 1
        m2   = math.sqrt(sum(x**2 for x in v2.values())) or 1
        return dot / (m1 * m2)
    all_docs = [tokenize(jd_text)] + [tokenize(t) for t in candidate_texts]
    idf_d    = idf(all_docs)
    jd_vec   = tfidf_vec(tf(all_docs[0]), idf_d)
    return [cosine(jd_vec, tfidf_vec(tf(doc), idf_d)) for doc in all_docs[1:]]


def compute_skill_score(candidate: dict, jd: dict) -> float:
    required  = parse_skills(jd.get("required_skills", []))
    preferred = parse_skills(jd.get("preferred_skills", []))
    c_skills  = parse_skills(candidate.get("skills", []))
    req_match  = count_skill_matches(required, c_skills)  / max(len(required), 1)
    pref_match = count_skill_matches(preferred, c_skills) / max(len(preferred), 1)
    return min(1.0, 0.7 * req_match + 0.3 * pref_match)


def compute_activity_score(candidate: dict) -> float:
    raw = candidate.get("activity_score", 50)
    try:
        return min(1.0, float(raw) / 100.0)
    except (TypeError, ValueError):
        return 0.5


def compute_experience_bonus(candidate: dict, jd: dict) -> float:
    req_years = jd.get("experience_years", 0)
    try:
        cand_years = float(candidate.get("experience_years", 0))
    except (TypeError, ValueError):
        cand_years = 0
    if req_years == 0:
        return 0.05
    ratio = cand_years / req_years
    if ratio >= 1.0:
        return 0.05
    elif ratio >= 0.7:
        return 0.02
    return 0.0


# -- STEP 4: RANK --------------------------------------------------------------

def rank_candidates(jd: dict, candidates: list) -> list:
    jd_text    = build_job_text(jd)
    cand_texts = [build_candidate_text(c) for c in candidates]

    print(f"\nRanking {len(candidates)} candidates for: {jd.get('title', 'Role')}")

    sem_scores = compute_semantic_scores(jd_text, cand_texts) if USE_SEMANTIC else tfidf_similarity(jd_text, cand_texts)

    results = []
    for i, candidate in enumerate(candidates):
        semantic  = sem_scores[i]
        skill     = compute_skill_score(candidate, jd)
        activity  = compute_activity_score(candidate)
        exp_bonus = compute_experience_bonus(candidate, jd)

        final_score = min(1.0,
            WEIGHT_SEMANTIC * semantic +
            WEIGHT_SKILLS   * skill    +
            WEIGHT_ACTIVITY * activity +
            exp_bonus
        )

        matched_req, missing_req, matched_pref = get_matched_missing(candidate, jd)

        results.append({
            "rank":             0,
            "candidate_id":     candidate.get("id", f"C{i+1:03d}"),
            "name":             candidate.get("name", "Unknown"),
            "final_score":      round(final_score, 4),
            "semantic_score":   round(float(semantic), 4),
            "skill_score":      round(skill, 4),
            "activity_score":   round(activity, 4),
            "experience_years": candidate.get("experience_years", "N/A"),
            "skills":           ", ".join(parse_skills(candidate.get("skills", []))),
            "summary":          candidate.get("summary", "")[:120],
            "matched_required": ", ".join(matched_req),
            "missing_required": ", ".join(missing_req),
            "matched_preferred": ", ".join(matched_pref),
        })

    results.sort(key=lambda x: x["final_score"], reverse=True)
    for i, r in enumerate(results, 1):
        r["rank"] = i
    return results


# -- STEP 5: EXPORT ------------------------------------------------------------

def export_csv(results: list):
    fieldnames = [
        "rank", "candidate_id", "name", "final_score",
        "semantic_score", "skill_score", "activity_score",
        "experience_years", "skills", "summary",
        "matched_required", "missing_required", "matched_preferred"
    ]
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"Ranked CSV saved to {OUTPUT_FILE}")


def print_top_results(results: list, n: int = 5):
    print(f"\n{'-'*65}")
    print(f"  TOP {min(n, len(results))} CANDIDATES")
    print(f"{'-'*65}")
    for r in results[:n]:
        bar = "#" * int(r["final_score"] * 20)
        print(f"  #{r['rank']:>2}  {r['name']:<22} Score: {r['final_score']:.2f}  {bar}")
        print(f"       Matched : {r['matched_required'] or 'None'}")
        print(f"       Missing : {r['missing_required'] or 'None'}")
        print(f"       Preferred: {r['matched_preferred'] or 'None'}")
        print()
    print(f"{'-'*65}\n")


def export_html(results: list, jd: dict, n: int):
    rows = ""
    for r in results[:n]:
        score_pct = int(r["final_score"] * 100)
        color = "#2e7d32" if r["rank"] <= 3 else "#555"
        bar_color = "#4caf50" if r["rank"] <= 3 else "#90a4ae"
        rows += f"""
        <tr>
          <td style="font-weight:bold;color:{color}">#{r['rank']}</td>
          <td><strong>{r['name']}</strong><br><small style="color:#666">{r['summary'][:80]}...</small></td>
          <td>
            <div style="background:#e0e0e0;border-radius:4px;height:16px;width:120px">
              <div style="background:{bar_color};width:{score_pct}%;height:16px;border-radius:4px"></div>
            </div>
            <small>{r['final_score']}</small>
          </td>
          <td>{r['semantic_score']}</td>
          <td>{r['skill_score']}</td>
          <td>{r['activity_score']}</td>
          <td>{r['experience_years']} yrs</td>
          <td><span style="color:#2e7d32">{r['matched_required'] or '-'}</span></td>
          <td><span style="color:#c62828">{r['missing_required'] or '-'}</span></td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Candidate Ranking Report</title>
<style>
  body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; color: #333; }}
  h1 {{ color: #0d3349; }} h2 {{ color: #555; font-weight: normal; }}
  table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
  th {{ background: #0d3349; color: white; padding: 12px 10px; text-align: left; font-size: 13px; }}
  td {{ padding: 12px 10px; border-bottom: 1px solid #eee; font-size: 13px; vertical-align: top; }}
  tr:nth-child(1) td {{ background: #e8f5e9; }}
  tr:nth-child(2) td {{ background: #f1f8e9; }}
  tr:nth-child(3) td {{ background: #f9fbe7; }}
  .formula {{ background: #0d3349; color: #4caf50; padding: 12px 20px; border-radius: 6px; font-family: monospace; margin: 20px 0; }}
  footer {{ margin-top: 30px; color: #999; font-size: 12px; }}
</style>
</head>
<body>
<h1>Intelligent Candidate Discovery & Ranking System</h1>
<h2>India Runs Hackathon &mdash; Data & AI Challenge | Track 01 | by Raja K C</h2>
<h3>Role: {jd.get('title', 'N/A')} &nbsp;|&nbsp; Required Experience: {jd.get('experience_years', 0)}+ years</h3>
<div class="formula">Final Score = 55% Semantic + 30% Skill Match + 15% Activity + 5% Experience Bonus (capped at 1.0)</div>
<table>
  <thead>
    <tr>
      <th>Rank</th><th>Candidate</th><th>Final Score</th><th>Semantic</th>
      <th>Skill</th><th>Activity</th><th>Experience</th>
      <th>Matched Skills</th><th>Missing Skills</th>
    </tr>
  </thead>
  <tbody>{rows}</tbody>
</table>
<footer>
  Generated by: github.com/RAJA1404/hack2skill &nbsp;|&nbsp; linkedin.com/in/raja-k-c-991b7a294
</footer>
</body>
</html>"""

    with open(HTML_FILE, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"HTML report saved to {HTML_FILE}")


# -- MAIN ----------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Intelligent Candidate Ranking System")
    parser.add_argument("--top", type=int, default=5, help="Show top N candidates (default: 5)")
    args = parser.parse_args()

    print("=" * 65)
    print("  INDIA RUNS - Intelligent Candidate Ranking System")
    print("  by Raja K C | github.com/RAJA1404/hack2skill")
    print("=" * 65)

    jd         = load_job_description()
    candidates = load_candidates()

    if not candidates:
        print("No candidates loaded. Check your /data folder.")
        sys.exit(1)

    results = rank_candidates(jd, candidates)
    print_top_results(results, n=args.top)
    export_csv(results)
    export_html(results, jd, n=args.top)

    print("Done! Files saved:")
    print(f"  CSV  -> output/ranked_candidates.csv")
    print(f"  HTML -> output/report.html")