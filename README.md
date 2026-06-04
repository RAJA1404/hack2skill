# Redrob Hackathon - Intelligent Candidate Ranking System

India Runs Data & AI Challenge, Track 01  
Author: Raja K C

## What This Does

This project ranks candidates for the Redrob AI hiring challenge using a production-grade, multi-signal Python ranker. It combines semantic matching, skill evidence, career quality, experience fit, location preference, availability signals, and honeypot filtering to produce a clean top-100 `submission.csv`.

## Why I Built This

Recruiters review hundreds or thousands of profiles for one role. Keyword-only filters miss strong candidates who describe the same skill differently, and they can be fooled by keyword stuffing. I built this ranker to identify candidates who are genuinely aligned with the role, explain why they rank highly, and keep the process reproducible without API calls or GPU dependency.

Built in about 2-4 hours for the India Runs Hackathon.

## Quick Start

```bash
git clone https://github.com/RAJA1404/hack2skill.git
cd hack2skill
pip install -r requirements.txt
python src/rank.py --candidates ./data/candidates.jsonl --out ./output/submission.csv
```

Single reproduce command after dependencies are installed:

```bash
python src/rank.py --candidates ./data/candidates.jsonl --out ./output/submission.csv
```

## Full Dataset Result

The ranker was tested on the 100,000-candidate dataset and generated a valid top-100 submission file.

- 100,000 candidates loaded
- 25 honeypot profiles detected
- 63,004 obvious non-fits filtered before scoring
- 36,996 real candidates scored
- 100 ranked candidates exported to `output/submission.csv`

## Top 5 Candidates

| Rank | Candidate ID | Title | Years | Location | Score |
|---|---|---|---:|---|---:|
| 1 | CAND_0018499 | Senior Machine Learning Engineer | 7.2 | Noida, Uttar Pradesh | 0.6002 |
| 2 | CAND_0046525 | Senior Machine Learning Engineer | 6.1 | Pune, Maharashtra | 0.5791 |
| 3 | CAND_0081846 | Lead AI Engineer | 6.7 | Jaipur, Rajasthan | 0.5786 |
| 4 | CAND_0064326 | Search Engineer | 7.6 | Gurgaon, Haryana | 0.5531 |
| 5 | CAND_0068811 | Applied ML Engineer | 8.0 | Pune, Maharashtra | 0.5491 |

## Approach

The solution uses a multi-signal hybrid ranker. It does not require GPU inference, paid APIs, or network access during ranking.

| Component | Weight | What It Captures |
|---|---:|---|
| Semantic similarity | 30% | Meaning-level match to the JD using TF-IDF fallback or MiniLM |
| Skills | 28% | Required and preferred skills, proficiency, duration, endorsements |
| Career quality | 20% | Product-company background, AI/ML role relevance, consulting penalty |
| Experience fit | 12% | Fit to the 5-9 year experience range |
| Location | 10% | Pune, Noida, Hyderabad, Mumbai, Delhi NCR preference |
| Availability | multiplier | Response rate, recency, open-to-work, notice period |

## Key Design Decisions

- Honeypot detection flags impossible profiles before scoring.
- Wrong-domain titles such as marketing, HR, accounting, or support are filtered early.
- Career descriptions are mined for skills, not only the explicit skills list.
- Availability is applied as a multiplier so inactive candidates are down-weighted.
- The script can use sentence-transformers if available, but defaults to TF-IDF for reliable offline execution.

## Output Format

`output/submission.csv` contains:

```text
candidate_id,rank,score,reasoning
```

The generated file has 100 rows, unique ranks from 1 to 100, and non-increasing scores.

## Project Structure

```text
hack2skill/
+-- src/
|   +-- rank.py                 <- production full-dataset ranker
|   +-- ranker.py               <- earlier demo ranker
+-- data/
|   +-- candidates.jsonl        <- place full dataset here
+-- output/
|   +-- submission.csv          <- generated top-100 submission
+-- submission_metadata.yaml
+-- requirements.txt
+-- README.md
```

## Files To Submit

- `output/submission.csv`
- GitHub repository: `https://github.com/RAJA1404/hack2skill.git`
- PDF deck with final results

## Author

Raja K C  
LinkedIn: [linkedin.com/in/raja-k-c-991b7a294](https://linkedin.com/in/raja-k-c-991b7a294)  
GitHub: [github.com/RAJA1404](https://github.com/RAJA1404)
