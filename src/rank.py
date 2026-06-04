"""
Redrob Hackathon - Intelligent Candidate Ranking System
Author : Raja K C
GitHub : https://github.com/RAJA1404/hack2skill.git

Approach : Multi-signal hybrid ranker (no GPU, no API calls, runs in <5 min)
  1. Semantic similarity  - sentence-transformers all-MiniLM-L6-v2
  2. Skills signal        - required / preferred overlap with proficiency + duration weighting
  3. Career quality       - product-company history, title progression, relevant role detection
  4. Experience fit       - years in range, recency of ML/AI work
  5. Location fit         - Pune/Noida/Hyderabad/Mumbai/Delhi NCR preference
  6. Availability signal  - behavioral signals from redrob_signals (activeness, response, notice)
  7. Honeypot filter      - flags impossible profiles before scoring

All components are combined with tuned weights, then sorted to produce top-100.
"""

import gzip
import json
import csv
import math
import argparse
import sys
from datetime import date, datetime
from pathlib import Path

# ─── Try sentence-transformers; fall back to TF-IDF ──────────────────────────
USE_SEMANTIC = False  # Set True on local machine with internet to use all-MiniLM-L6-v2
try:
    from sentence_transformers import SentenceTransformer, util
except ImportError:
    pass

# ─── Constants ────────────────────────────────────────────────────────────────
TODAY = date.today()

# Scoring weights (sum = 1.0 before availability multiplier)
W_SEMANTIC   = 0.30
W_SKILLS     = 0.28
W_CAREER     = 0.20
W_EXPERIENCE = 0.12
W_LOCATION   = 0.10

# JD-derived signals
REQUIRED_SKILLS = [
    "embeddings", "sentence-transformers", "vector database", "faiss",
    "pinecone", "weaviate", "qdrant", "milvus", "elasticsearch", "opensearch",
    "retrieval", "ranking", "semantic search", "python", "evaluation",
    "ndcg", "mrr", "a/b testing", "bm25", "hybrid search", "rag",
    "information retrieval", "re-ranking", "llm", "fine-tuning", "pytorch",
]
PREFERRED_SKILLS = [
    "lora", "qlora", "peft", "learning to rank", "xgboost", "distributed systems",
    "kafka", "airflow", "spark", "open-source", "nlp", "transformers",
    "huggingface", "langchain", "vector search", "recommendation systems",
]
# Titles that suggest product-company AI/ML background
GOOD_TITLE_TOKENS = [
    "ml engineer", "machine learning", "ai engineer", "nlp engineer",
    "data scientist", "research engineer", "applied scientist",
    "search engineer", "ranking engineer", "retrieval", "recommendations",
]
# Titles that are red flags (keyword stuffers / wrong domain)
BAD_TITLES = [
    "marketing manager", "hr manager", "accountant", "sales executive",
    "graphic designer", "content writer", "civil engineer",
    "mechanical engineer", "customer support", "operations manager",
    "project manager",
]
# Consulting firms explicitly called out in JD
CONSULTING_FIRMS = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "mindtree", "mphasis", "hexaware", "tech mahindra",
}
# Preferred locations
PREFERRED_LOCATIONS = {
    "pune", "noida", "hyderabad", "mumbai", "delhi", "bangalore", "bengaluru",
    "gurgaon", "gurugram",
}
# Experience window the JD says it wants (years)
EXP_MIN, EXP_MAX = 5, 9

JD_TEXT = """
Senior AI Engineer role at Redrob AI. Need production experience with 
embeddings-based retrieval systems (sentence-transformers, OpenAI embeddings, 
BGE, E5). Production experience with vector databases (Pinecone, Weaviate, 
Qdrant, Milvus, OpenSearch, Elasticsearch, FAISS). Strong Python. 
Hands-on evaluation frameworks for ranking systems (NDCG, MRR, MAP, A/B testing).
Shipped end-to-end ranking, search or recommendation system to real users at scale.
5-9 years experience. Product company background, not pure consulting or research.
Scrappy product-engineering attitude. Hybrid retrieval, LLM re-ranking architecture.
Located in Pune or Noida preferred, open to Hyderabad, Mumbai, Delhi NCR.
"""


# ─── Honeypot detection ───────────────────────────────────────────────────────
def is_honeypot(candidate: dict) -> bool:
    """
    Detect impossible profiles planted as traps.
    Rules:
      - Company founded date implied by first career entry vs years of experience
      - Expert proficiency in 10+ skills with 0 total endorsements
      - years_of_experience > sum of all career_history duration_months / 12 + 5
    """
    profile = candidate.get("profile", {})
    skills  = candidate.get("skills", [])
    career  = candidate.get("career_history", [])

    # Rule 1 — too many expert skills with zero endorsements
    expert_zero = [s for s in skills
                   if s.get("proficiency") == "expert" and s.get("endorsements", 0) == 0]
    if len(expert_zero) >= 8:
        return True

    # Rule 2 — claimed years >> sum of career months
    claimed_yrs = profile.get("years_of_experience", 0)
    career_months = sum(r.get("duration_months", 0) for r in career)
    career_yrs = career_months / 12 if career_months else 0
    if claimed_yrs > career_yrs + 6 and claimed_yrs > 10:
        return True

    # Rule 3 — current company start_date inconsistent with experience
    for role in career:
        if role.get("is_current"):
            try:
                start = datetime.strptime(role["start_date"], "%Y-%m-%d").date()
                role_age_yrs = (TODAY - start).days / 365
                if claimed_yrs > role_age_yrs + career_yrs + 4:
                    return True
            except Exception:
                pass

    return False


# ─── Skills scoring ───────────────────────────────────────────────────────────
PROF_WEIGHT = {"expert": 1.0, "advanced": 0.85, "intermediate": 0.6, "beginner": 0.3}

def skills_score(candidate: dict) -> float:
    skills_list = candidate.get("skills", [])
    skill_map = {}
    for s in skills_list:
        name = s.get("name", "").lower()
        prof = PROF_WEIGHT.get(s.get("proficiency", "beginner"), 0.3)
        dur  = min(1.0, s.get("duration_months", 0) / 36)   # cap at 3 years
        end  = min(1.0, s.get("endorsements", 0) / 20)       # cap at 20 endorsements
        trust = 0.5 * prof + 0.3 * dur + 0.2 * end
        skill_map[name] = max(skill_map.get(name, 0), trust)

    # Also scan career descriptions for skill mentions
    for role in candidate.get("career_history", []):
        desc = role.get("description", "").lower()
        for req in REQUIRED_SKILLS:
            if req in desc and req not in skill_map:
                skill_map[req] = 0.45   # implied but not listed = partial credit

    req_scores  = [skill_map.get(r, 0) for r in REQUIRED_SKILLS]
    pref_scores = [skill_map.get(p, 0) for p in PREFERRED_SKILLS]

    req_score  = sum(req_scores)  / len(REQUIRED_SKILLS)
    pref_score = sum(pref_scores) / len(PREFERRED_SKILLS)

    # Also grab skill_assessment_scores from redrob signals if available
    assessment = candidate.get("redrob_signals", {}).get("skill_assessment_scores", {})
    if assessment:
        relevant_assessments = []
        for skill_name, score in assessment.items():
            sk = skill_name.lower()
            if any(r in sk for r in ["python", "ml", "nlp", "data", "ai", "retrieval"]):
                relevant_assessments.append(score / 100)
        if relevant_assessments:
            assessment_bonus = sum(relevant_assessments) / len(relevant_assessments) * 0.1
        else:
            assessment_bonus = 0
    else:
        assessment_bonus = 0

    return min(1.0, 0.7 * req_score + 0.3 * pref_score + assessment_bonus)


# ─── Career quality scoring ───────────────────────────────────────────────────
def career_score(candidate: dict) -> float:
    profile = candidate.get("profile", {})
    career  = candidate.get("career_history", [])

    current_title = profile.get("current_title", "").lower()
    score = 0.0

    # Penalize bad titles (keyword stuffers)
    if any(bad in current_title for bad in BAD_TITLES):
        score -= 0.35

    # Reward good titles
    if any(g in current_title for g in GOOD_TITLE_TOKENS):
        score += 0.5

    product_months = 0
    consulting_months = 0
    ai_relevant_months = 0

    for role in career:
        comp  = role.get("company", "").lower()
        title = role.get("title", "").lower()
        desc  = role.get("description", "").lower()
        dur   = role.get("duration_months", 0)
        ind   = role.get("industry", "").lower()

        # Consulting vs product
        if any(cf in comp for cf in CONSULTING_FIRMS):
            consulting_months += dur
        else:
            product_months += dur

        # AI/ML relevance of role
        ai_terms = ["retrieval", "ranking", "embedding", "recommendation",
                    "search", "nlp", "machine learning", "ml", "llm",
                    "vector", "rag", "fine-tun"]
        if (any(t in title for t in ai_terms) or
                any(t in desc  for t in ai_terms) or
                "ai" in ind or "ml" in ind):
            ai_relevant_months += dur

    total_months = max(1, product_months + consulting_months)

    product_ratio    = product_months / total_months
    ai_relevant_ratio = min(1.0, ai_relevant_months / max(1, total_months))

    # Penalize pure consulting history
    consulting_penalty = -0.2 if consulting_months / total_months > 0.8 else 0

    score += 0.4 * product_ratio + 0.4 * ai_relevant_ratio + consulting_penalty
    return max(0.0, min(1.0, score))


# ─── Experience fit ───────────────────────────────────────────────────────────
def experience_score(candidate: dict) -> float:
    yrs = candidate.get("profile", {}).get("years_of_experience", 0)

    if EXP_MIN <= yrs <= EXP_MAX:
        # Sweet spot — score based on where in range
        score = 0.8 + 0.2 * ((yrs - EXP_MIN) / (EXP_MAX - EXP_MIN))
    elif yrs < EXP_MIN:
        # Under-experienced — linear decay
        score = max(0.0, 0.8 * (yrs / EXP_MIN))
    else:
        # Over-experienced — mild penalty (JD said 5-9 but won't hard reject)
        excess = yrs - EXP_MAX
        score = max(0.4, 0.8 - 0.03 * excess)

    return score


# ─── Location fit ─────────────────────────────────────────────────────────────
def location_score(candidate: dict) -> float:
    profile  = candidate.get("profile", {})
    signals  = candidate.get("redrob_signals", {})

    loc     = profile.get("location", "").lower()
    country = profile.get("country", "").lower()
    relocate = signals.get("willing_to_relocate", False)

    if country not in ("india", "in", ""):
        return 0.1 if relocate else 0.0

    if any(city in loc for city in PREFERRED_LOCATIONS):
        return 1.0

    # India but not preferred city
    if relocate:
        return 0.55
    return 0.25


# ─── Availability / behavioral signal ─────────────────────────────────────────
def availability_score(candidate: dict) -> float:
    sig = candidate.get("redrob_signals", {})
    scores = []

    # Recency of login
    last_active_str = sig.get("last_active_date", "")
    if last_active_str:
        try:
            last_active = datetime.strptime(last_active_str, "%Y-%m-%d").date()
            days_inactive = (TODAY - last_active).days
            if days_inactive <= 7:
                recency = 1.0
            elif days_inactive <= 30:
                recency = 0.85
            elif days_inactive <= 90:
                recency = 0.6
            elif days_inactive <= 180:
                recency = 0.3
            else:
                recency = 0.05
            scores.append(recency)
        except Exception:
            pass

    # Open to work
    if sig.get("open_to_work_flag", False):
        scores.append(1.0)
    else:
        scores.append(0.3)

    # Recruiter response rate
    rr = sig.get("recruiter_response_rate", 0.5)
    scores.append(float(rr))

    # Notice period — JD wants sub-30 days ideally
    notice = sig.get("notice_period_days", 90)
    if notice <= 30:
        notice_score = 1.0
    elif notice <= 60:
        notice_score = 0.7
    elif notice <= 90:
        notice_score = 0.5
    else:
        notice_score = 0.2
    scores.append(notice_score)

    # Interview completion rate
    icr = sig.get("interview_completion_rate", 0.5)
    scores.append(float(icr))

    # Profile completeness
    completeness = sig.get("profile_completeness_score", 50) / 100
    scores.append(completeness)

    return sum(scores) / len(scores) if scores else 0.5


# ─── Semantic scoring (batch) ──────────────────────────────────────────────────
def build_candidate_text(candidate: dict) -> str:
    p = candidate.get("profile", {})
    parts = [
        p.get("headline", ""),
        p.get("summary", ""),
        p.get("current_title", ""),
        p.get("current_industry", ""),
    ]
    for role in candidate.get("career_history", [])[:3]:
        parts.append(role.get("title", ""))
        parts.append(role.get("description", "")[:200])
    skill_names = [s.get("name", "") for s in candidate.get("skills", [])
                   if s.get("proficiency") in ("advanced", "expert")]
    parts.append(", ".join(skill_names))
    return " | ".join(p for p in parts if p)


def tfidf_batch(jd: str, texts: list) -> list:
    import re
    def tok(t): return re.findall(r'\w+', t.lower())
    def tf(tokens):
        freq = {}
        for t in tokens: freq[t] = freq.get(t, 0) + 1
        n = len(tokens) or 1
        return {t: c/n for t, c in freq.items()}
    docs = [tok(jd)] + [tok(t) for t in texts]
    N = len(docs)
    df = {}
    for d in docs:
        for t in set(d): df[t] = df.get(t, 0) + 1
    idf = {t: math.log((N+1)/(c+1))+1 for t, c in df.items()}
    def vec(tokens):
        t = tf(tokens)
        return {k: t.get(k,0)*idf.get(k,0) for k in idf}
    def cos(v1, v2):
        dot = sum(v1.get(k,0)*v2.get(k,0) for k in idf)
        m1 = math.sqrt(sum(x**2 for x in v1.values())) or 1
        m2 = math.sqrt(sum(x**2 for x in v2.values())) or 1
        return dot / (m1*m2)
    jd_vec = vec(docs[0])
    return [cos(jd_vec, vec(d)) for d in docs[1:]]


# ─── Reasoning generator ──────────────────────────────────────────────────────
def build_reasoning(candidate: dict, components: dict, rank: int) -> str:
    p      = candidate.get("profile", {})
    sig    = candidate.get("redrob_signals", {})
    skills = candidate.get("skills", [])

    title = p.get("current_title", "Unknown")
    yrs   = p.get("years_of_experience", 0)
    loc   = p.get("location", "Unknown")
    notice= sig.get("notice_period_days", "?")
    rr    = sig.get("recruiter_response_rate", 0)
    otw   = sig.get("open_to_work_flag", False)

    top_skills = [s["name"] for s in skills
                  if s.get("proficiency") in ("expert", "advanced")][:4]

    if rank <= 10:
        skill_str = ", ".join(top_skills) if top_skills else "relevant skills"
        return (f"{title} with {yrs:.1f} yrs exp in {loc}; "
                f"strong match on {skill_str}; "
                f"notice {notice}d, response rate {rr:.0%}, "
                f"{'open to work' if otw else 'passive candidate'}.")
    elif rank <= 30:
        concern = ""
        if yrs < EXP_MIN:
            concern = f" Under-experienced ({yrs:.1f} yrs vs 5-9 req)."
        elif components.get("career", 0) < 0.4:
            concern = " Career history less aligned to AI/ML product work."
        return (f"{title}, {yrs:.1f} yrs, {loc}. "
                f"Reasonable skills overlap.{concern} "
                f"Response rate {rr:.0%}.")
    else:
        gap = "Skills gap vs JD requirements."
        if components.get("location", 0) < 0.3:
            gap += " Location outside preferred cities."
        if components.get("availability", 0) < 0.4:
            gap += f" Low engagement (response rate {rr:.0%})."
        return (f"{title}, {yrs:.1f} yrs. {gap} "
                f"Ranked {rank} - included as extended consideration.")


# ─── Main ranking pipeline ────────────────────────────────────────────────────
def rank_candidates(candidates_path: str, out_path: str):
    # Load candidates
    print(f"Loading candidates from {candidates_path} ...")
    candidates = []
    opener = gzip.open if candidates_path.endswith(".gz") else open
    mode   = "rt"
    with opener(candidates_path, mode, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
    print(f"Loaded {len(candidates):,} candidates")

    # Step 1 — Honeypot filter (flag but don't remove yet)
    print("Detecting honeypots ...")
    honeypot_flags = [is_honeypot(c) for c in candidates]
    n_honeypots = sum(honeypot_flags)
    print(f"   Flagged {n_honeypots:,} honeypot candidates")

    # Step 2 — Fast pre-filter: skip obvious non-fits to save time
    print("Pre-filtering non-fits ...")
    filtered = []
    for i, (c, is_hp) in enumerate(zip(candidates, honeypot_flags)):
        if is_hp:
            continue
        title = c.get("profile", {}).get("current_title", "").lower()
        if any(bad in title for bad in BAD_TITLES):
            continue
        filtered.append(c)
    print(f"   {len(filtered):,} candidates after pre-filter")

    # Step 3 — Component scoring
    print("Computing component scores ...")
    scored = []
    for c in filtered:
        sk  = skills_score(c)
        car = career_score(c)
        exp = experience_score(c)
        loc = location_score(c)
        avail = availability_score(c)
        components = {
            "skills":       sk,
            "career":       car,
            "experience":   exp,
            "location":     loc,
            "availability": avail,
        }
        scored.append((c, components))

    # Step 4 — Semantic scoring (on top-2000 by component score first)
    print("Semantic scoring ...")
    # Quick pre-rank by non-semantic score to focus semantic on top candidates
    for item in scored:
        c, comp = item
        pre = (W_SKILLS * comp["skills"] +
               W_CAREER * comp["career"] +
               W_EXPERIENCE * comp["experience"] +
               W_LOCATION * comp["location"])
        comp["pre_score"] = pre

    scored.sort(key=lambda x: x[1]["pre_score"], reverse=True)
    top_n_semantic = 3000
    top_pool = scored[:top_n_semantic]
    rest     = scored[top_n_semantic:]

    texts = [build_candidate_text(c) for c, _ in top_pool]

    if USE_SEMANTIC:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        jd_emb   = model.encode(JD_TEXT, convert_to_tensor=True)
        c_embs   = model.encode(texts, convert_to_tensor=True, batch_size=256,
                                show_progress_bar=False)
        sem_scores = [float(s) for s in util.cos_sim(jd_emb, c_embs)[0]]
    else:
        sem_scores = tfidf_batch(JD_TEXT, texts)

    for i, (c, comp) in enumerate(top_pool):
        comp["semantic"] = sem_scores[i]

    # Assign low semantic score to rest (they were pre-filtered out)
    for c, comp in rest:
        comp["semantic"] = 0.0

    all_scored = top_pool + rest

    # Step 5 — Final score
    print("Computing final scores ...")
    results = []
    for c, comp in all_scored:
        final = (W_SEMANTIC   * comp["semantic"]   +
                 W_SKILLS     * comp["skills"]     +
                 W_CAREER     * comp["career"]     +
                 W_EXPERIENCE * comp["experience"] +
                 W_LOCATION   * comp["location"])
        # Availability as multiplier (0.6 – 1.15 range)
        avail_mult = 0.6 + 0.55 * comp["availability"]
        final = min(1.0, final * avail_mult)
        comp["final"] = final
        results.append((c, comp))

    results.sort(key=lambda x: x[1]["final"], reverse=True)

    # Step 6 — Write top-100 CSV
    print("Writing submission CSV ...")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, (c, comp) in enumerate(results[:100], 1):
            cid      = c["candidate_id"]
            score    = round(comp["final"], 4)
            reasoning = build_reasoning(c, comp, rank)
            writer.writerow([cid, rank, score, reasoning])

    print(f"\n{'='*60}")
    print(f"  TOP 10 CANDIDATES")
    print(f"{'='*60}")
    for rank, (c, comp) in enumerate(results[:10], 1):
        p = c["profile"]
        print(f"  #{rank:>3}  {p['current_title']:<30} {p['years_of_experience']:>4.1f}y  "
              f"score={comp['final']:.3f}  loc={p['location']}")
    print(f"{'='*60}")
    print(f"\nSubmission saved to {out_path}")
    return results[:100]


# ─── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Redrob Candidate Ranker")
    parser.add_argument("--candidates", default="./data/candidates.jsonl",
                        help="Path to candidates.jsonl or candidates.jsonl.gz")
    parser.add_argument("--out", default="./output/submission.csv",
                        help="Output CSV path")
    args = parser.parse_args()
    rank_candidates(args.candidates, args.out)
