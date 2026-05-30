"""
India Runs - Data & AI Challenge
Intelligent Candidate Discovery & Ranking System
Author: Raja K C
GitHub: https://github.com/RAJA1404/hack2skill.git
"""

import json
import csv
import math
import re
from pathlib import Path

# ── Try to import AI libraries; fall back to TF-IDF if not installed ──────────
try:
    from sentence_transformers import SentenceTransformer, util
    USE_SEMANTIC = True
    print("sentence-transformers found - using semantic embeddings")
except ImportError:
    USE_SEMANTIC = False
    print("sentence-transformers not found - using TF-IDF fallback")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG  (edit these paths to match your dataset filenames)
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR    = Path(__file__).parent.parent / "data"
OUTPUT_DIR  = Path(__file__).parent.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

JOB_DESC_FILE       = DATA_DIR / "job_description.json"   # or .txt
CANDIDATES_FILE     = DATA_DIR / "candidates.json"         # or .csv
OUTPUT_FILE         = OUTPUT_DIR / "ranked_candidates.csv"

# Scoring weights (must sum to 1.0)
WEIGHT_SEMANTIC  = 0.55   # semantic / meaning similarity
WEIGHT_SKILLS    = 0.30   # direct skill overlap
WEIGHT_ACTIVITY  = 0.15   # platform activity & behavioral signals

SKILL_ALIASES = {
    "apache spark": "spark",
    "pyspark": "spark",
    "spark": "spark",
}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

def load_job_description():
    """Load job description from JSON or plain text."""
    if not JOB_DESC_FILE.exists():
        # Demo job description if file not yet provided
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
    """Load candidates from JSON or CSV."""
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


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — BUILD CANDIDATE TEXT PROFILE
# ─────────────────────────────────────────────────────────────────────────────

def build_candidate_text(candidate: dict) -> str:
    """Combine all candidate fields into one rich text for AI embedding."""
    parts = []
    if candidate.get("summary"):
        parts.append(candidate["summary"])
    if candidate.get("skills"):
        parts.append("Skills: " + ", ".join(parse_skills(candidate["skills"])))
    if candidate.get("experience"):
        parts.append(str(candidate["experience"]))
    if candidate.get("education"):
        parts.append(str(candidate["education"]))
    if candidate.get("bio"):
        parts.append(candidate["bio"])
    return " | ".join(parts) if parts else candidate.get("name", "")


def build_job_text(jd: dict) -> str:
    """Combine job description fields into rich text for AI embedding."""
    parts = [jd.get("description", "")]
    if jd.get("required_skills"):
        parts.append("Required: " + ", ".join(jd["required_skills"]))
    if jd.get("preferred_skills"):
        parts.append("Preferred: " + ", ".join(jd["preferred_skills"]))
    if jd.get("title"):
        parts.insert(0, "Role: " + jd["title"])
    return " | ".join(parts)


def parse_skills(raw_skills) -> list:
    """Normalize list or CSV skill fields into a clean list of skill names."""
    if not raw_skills:
        return []
    if isinstance(raw_skills, list):
        return [str(skill).strip() for skill in raw_skills if str(skill).strip()]
    return [skill.strip() for skill in str(raw_skills).split(",") if skill.strip()]


def normalize_skill(skill: str) -> str:
    """Lowercase and simplify a skill name for partial/fuzzy comparison."""
    normalized = re.sub(r"[^a-z0-9]+", " ", str(skill).lower()).strip()
    return SKILL_ALIASES.get(normalized, normalized)


def skills_match(candidate_skill: str, target_skill: str) -> bool:
    """Return True for exact, alias, partial, or token-overlap skill matches."""
    candidate_norm = normalize_skill(candidate_skill)
    target_norm = normalize_skill(target_skill)
    if candidate_norm == target_norm:
        return True
    if candidate_norm in target_norm or target_norm in candidate_norm:
        return True

    candidate_tokens = set(candidate_norm.split())
    target_tokens = set(target_norm.split())
    return bool(candidate_tokens and target_tokens and candidate_tokens & target_tokens)


def count_skill_matches(target_skills: list, candidate_skills: list) -> int:
    """Count JD skills matched by at least one candidate skill."""
    return sum(
        1
        for target_skill in target_skills
        if any(skills_match(candidate_skill, target_skill) for candidate_skill in candidate_skills)
    )


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — SCORING COMPONENTS
# ─────────────────────────────────────────────────────────────────────────────

def compute_semantic_scores(jd_text: str, candidate_texts: list) -> list:
    """Use sentence-transformers to get cosine similarity scores."""
    model = SentenceTransformer("all-MiniLM-L6-v2")
    jd_embedding  = model.encode(jd_text, convert_to_tensor=True)
    cand_embeddings = model.encode(candidate_texts, convert_to_tensor=True)
    scores = util.cos_sim(jd_embedding, cand_embeddings)[0]
    return [float(s) for s in scores]


def tfidf_similarity(jd_text: str, candidate_texts: list) -> list:
    """Fallback TF-IDF cosine similarity (no external AI library needed)."""
    def tokenize(text):
        import re
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

    def tfidf_vec(tf_dict, idf_dict):
        return {t: tf_dict.get(t, 0) * idf_dict.get(t, 0) for t in idf_dict}

    def cosine(v1, v2):
        keys = set(v1) | set(v2)
        dot  = sum(v1.get(k, 0) * v2.get(k, 0) for k in keys)
        m1   = math.sqrt(sum(x**2 for x in v1.values())) or 1
        m2   = math.sqrt(sum(x**2 for x in v2.values())) or 1
        return dot / (m1 * m2)

    all_docs  = [tokenize(jd_text)] + [tokenize(t) for t in candidate_texts]
    idf_dict  = idf(all_docs)
    jd_vec    = tfidf_vec(tf(all_docs[0]), idf_dict)
    scores    = []
    for doc_tokens in all_docs[1:]:
        c_vec = tfidf_vec(tf(doc_tokens), idf_dict)
        scores.append(cosine(jd_vec, c_vec))
    return scores


def compute_skill_score(candidate: dict, jd: dict) -> float:
    """Overlap between candidate skills and JD required/preferred skills."""
    required  = parse_skills(jd.get("required_skills", []))
    preferred = parse_skills(jd.get("preferred_skills", []))
    c_skills  = parse_skills(candidate.get("skills", []))

    req_match  = count_skill_matches(required, c_skills) / max(len(required), 1)
    pref_match = count_skill_matches(preferred, c_skills) / max(len(preferred), 1)
    return min(1.0, 0.7 * req_match + 0.3 * pref_match)


def compute_activity_score(candidate: dict) -> float:
    """Normalize platform activity / behavioral signal (0–100 → 0.0–1.0)."""
    raw = candidate.get("activity_score", 50)
    try:
        return min(1.0, float(raw) / 100.0)
    except (TypeError, ValueError):
        return 0.5


def compute_experience_bonus(candidate: dict, jd: dict) -> float:
    """Small bonus for meeting or exceeding required experience."""
    req_years  = jd.get("experience_years", 0)
    cand_years = candidate.get("experience_years", 0)
    try:
        cand_years = float(cand_years)
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


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — RANK CANDIDATES
# ─────────────────────────────────────────────────────────────────────────────

def rank_candidates(jd: dict, candidates: list) -> list:
    jd_text     = build_job_text(jd)
    cand_texts  = [build_candidate_text(c) for c in candidates]

    print(f"\nRanking {len(candidates)} candidates for: {jd.get('title', 'Role')}")

    # Semantic scores
    if USE_SEMANTIC:
        sem_scores = compute_semantic_scores(jd_text, cand_texts)
    else:
        sem_scores = tfidf_similarity(jd_text, cand_texts)

    results = []
    for i, candidate in enumerate(candidates):
        semantic   = sem_scores[i]
        skill      = compute_skill_score(candidate, jd)
        activity   = compute_activity_score(candidate)
        exp_bonus  = compute_experience_bonus(candidate, jd)

        final_score = (
            WEIGHT_SEMANTIC * semantic +
            WEIGHT_SKILLS   * skill    +
            WEIGHT_ACTIVITY * activity +
            exp_bonus
        )
        # The experience bonus is additive, but the final score is capped at 1.0.
        final_score = min(1.0, final_score)

        results.append({
            "rank":            0,  # filled after sort
            "candidate_id":    candidate.get("id", f"C{i+1:03d}"),
            "name":            candidate.get("name", "Unknown"),
            "final_score":     round(final_score, 4),
            "semantic_score":  round(float(semantic), 4),
            "skill_score":     round(skill, 4),
            "activity_score":  round(activity, 4),
            "experience_years": candidate.get("experience_years", "N/A"),
            "skills":          ", ".join(parse_skills(candidate.get("skills", []))),
            "summary":         candidate.get("summary", "")[:120],
        })

    results.sort(key=lambda x: x["final_score"], reverse=True)
    for i, r in enumerate(results, 1):
        r["rank"] = i

    return results


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — EXPORT RESULTS
# ─────────────────────────────────────────────────────────────────────────────

def export_csv(results: list):
    fieldnames = [
        "rank", "candidate_id", "name", "final_score",
        "semantic_score", "skill_score", "activity_score",
        "experience_years", "skills", "summary"
    ]
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nRanked output saved to {OUTPUT_FILE}")


def print_top_results(results: list, n: int = 5):
    print(f"\n{'-'*65}")
    print(f"  TOP {min(n, len(results))} CANDIDATES")
    print(f"{'-'*65}")
    for r in results[:n]:
        bar = "#" * int(r["final_score"] * 20)
        print(f"  #{r['rank']:>2}  {r['name']:<22} Score: {r['final_score']:.2f}  {bar}")
    print(f"{'-'*65}\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("  INDIA RUNS - Intelligent Candidate Ranking System")
    print("  by Raja K C | github.com/raja-k-c")
    print("=" * 65)

    jd          = load_job_description()
    candidates  = load_candidates()

    if not candidates:
        print("No candidates loaded. Check your /data folder.")
        exit(1)

    results = rank_candidates(jd, candidates)
    print_top_results(results)
    export_csv(results)

    print("Done! Submit the file at: output/ranked_candidates.csv")
