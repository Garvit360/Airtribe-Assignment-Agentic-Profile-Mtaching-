"""
Part B: Job Matching Engine (LangChain).
Semantic retrieval via LangChain Chroma, hybrid filter, ranking, must-have filtering, same JSON output.
"""

import logging
import re
from pathlib import Path

from pydantic import BaseModel, Field
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from embedding_utils import CHROMA_COLLECTION_NAME, CHROMA_PERSIST_DIR, require_openai_key

logger = logging.getLogger(__name__)


class JobMatchResult(BaseModel):
    candidate_name: str = Field(..., description="Candidate full name")
    resume_path: str = Field(..., description="Path to resume file")
    match_score: int = Field(..., ge=0, le=100, description="Match score 0-100")
    matched_skills: list[str] = Field(default_factory=list, description="Skills that matched JD")
    relevant_excerpts: list[str] = Field(default_factory=list, description="Relevant text snippets")
    reasoning: str = Field(..., description="Why this candidate matches")


class JobMatchOutput(BaseModel):
    job_description: str = Field(..., description="Input job description")
    top_matches: list[JobMatchResult] = Field(default_factory=list, description="Top K matches")


def _semantic_search(
    job_description: str,
    k: int,
    persist_dir: str,
) -> list[dict]:
    """Use LangChain Chroma retriever; return list of {document, metadata, distance}."""
    persist_path = Path(persist_dir)
    if not persist_path.exists() or not list(persist_path.iterdir()):
        logger.error(
            "Vector database not found at %s\n"
            "  → Run step 1 first: python resume_rag.py\n"
            "  → This indexes PDFs from the Resume/ folder and creates the database.",
            persist_dir,
        )
        raise FileNotFoundError(
            f"Chroma DB not found at {persist_dir}. Run resume_rag.py first to index resumes."
        )

    embedding = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = Chroma(
        persist_directory=persist_dir,
        embedding_function=embedding,
        collection_name=CHROMA_COLLECTION_NAME,
    )
    hits = vector_store.similarity_search_with_score(job_description, k=k)
    return [
        {
            "document": doc.page_content,
            "metadata": doc.metadata,
            "distance": float(score),
        }
        for doc, score in hits
    ]


def extract_critical_skills(jd: str) -> list[str]:
    skills: list[str] = []
    for m in re.finditer(
        r"(?:\d+\+\s*years?\s+(?:of\s+)?|experience\s+(?:with|in)\s+|proficient\s+in\s+)([A-Za-z][A-Za-z0-9\s]+?)(?:\s*[,.]|\s+and\s+|\s*$)",
        jd,
        re.I,
    ):
        skill = m.group(1).strip()
        if len(skill) < 50 and skill not in skills:
            skills.append(skill)
    for m in re.finditer(r"(?:skills?|required|must\s+have)\s*[:\-]\s*([^\n.]+)", jd, re.I):
        for part in re.split(r"[,;]", m.group(1)):
            part = part.strip().strip("-")
            if part and len(part) < 40:
                skills.append(part)
    return list(dict.fromkeys(skills))[:30]


def apply_hybrid_filter(
    search_results: list[dict],
    critical_skills: list[str],
    k: int,
) -> list[dict]:
    def meta_skills(meta: dict) -> set[str]:
        s = meta.get("skills") or ""
        return set(x.strip() for x in s.split(",") if x.strip())

    critical_set = set(s.lower() for s in critical_skills)
    scored: list[tuple[float, dict]] = []
    for r in search_results:
        meta = r.get("metadata") or {}
        res_skills = meta_skills(meta)
        overlap = sum(1 for cs in critical_set if any(cs in rs.lower() for rs in res_skills))
        hybrid_score = overlap * 2.0 - r.get("distance", 0)
        scored.append((hybrid_score, r))
    scored.sort(key=lambda x: -x[0])
    return [r for _, r in scored[:k]]


def score_match(
    jd: str,
    resume_metadata: dict,
    chunks: list[dict],
    base_distance: float,
) -> tuple[int, str, list[str], list[str]]:
    similarity_score = max(0.0, 100.0 - base_distance * 50.0)
    jd_lower = jd.lower()
    meta_skills_str = resume_metadata.get("skills") or ""
    meta_skills_list = [x.strip() for x in meta_skills_str.split(",") if x.strip()]
    matched_skills: list[str] = []
    for sk in meta_skills_list:
        if sk.lower() in jd_lower or any(w in jd_lower for w in sk.split() if len(w) > 2):
            matched_skills.append(sk)
    boost = min(20, len(matched_skills) * 3)
    final_score = min(100, int(similarity_score) + boost)
    relevant_excerpts = [c.get("document", c.get("text", ""))[:200] for c in chunks[:3]]
    relevant_excerpts = [e for e in relevant_excerpts if e]
    reasoning_parts = []
    if matched_skills:
        reasoning_parts.append(f"Matched skills: {', '.join(matched_skills[:10])}.")
    exp = resume_metadata.get("experience_years")
    if exp is not None and exp >= 0:
        reasoning_parts.append(f"Experience: {exp} years.")
    edu = resume_metadata.get("education")
    if edu:
        reasoning_parts.append(f"Education: {edu[:80]}.")
    reasoning = " ".join(reasoning_parts) if reasoning_parts else "Semantic match based on resume content."
    return final_score, reasoning, matched_skills, relevant_excerpts


def parse_must_have_years(jd: str) -> int | None:
    m = re.search(r"(\d+)\s*\+\s*years?", jd, re.I)
    return int(m.group(1)) if m else None


def meets_must_have(
    resume_metadata: dict,
    required_years: int | None,
    critical_skills: list[str],
) -> bool:
    if required_years is not None:
        exp = resume_metadata.get("experience_years")
        if exp is None or exp < 0 or exp < required_years:
            return False
    if not critical_skills:
        return True
    meta_skills_str = resume_metadata.get("skills") or ""
    meta_skills_set = set(x.strip().lower() for x in meta_skills_str.split(",") if x.strip())
    for cs in critical_skills[:5]:
        if any(cs.lower() in ms for ms in meta_skills_set):
            return True
    return len(critical_skills) == 0


def run_job_matcher(
    job_description: str,
    k: int = 10,
    persist_dir: str = CHROMA_PERSIST_DIR,
) -> dict:
    require_openai_key()
    logger.info("Searching for matches (top %d) ...", k)
    raw_results = _semantic_search(job_description, k=k * 2, persist_dir=persist_dir)
    critical_skills = extract_critical_skills(job_description)
    filtered = apply_hybrid_filter(raw_results, critical_skills, k * 2)
    required_years = parse_must_have_years(job_description)

    by_path: dict[str, list[dict]] = {}
    for r in filtered:
        path = (r.get("metadata") or {}).get("path") or "unknown"
        if path not in by_path:
            by_path[path] = []
        by_path[path].append(r)

    candidates: list[JobMatchResult] = []
    for path, chunk_list in by_path.items():
        if len(candidates) >= k:
            break
        meta = (chunk_list[0].get("metadata")) or {}
        if not meets_must_have(meta, required_years, critical_skills):
            continue
        best_dist = min(c.get("distance", 2.0) for c in chunk_list)
        score, reasoning, matched_skills, excerpts = score_match(
            job_description, meta, chunk_list, best_dist
        )
        candidates.append(
            JobMatchResult(
                candidate_name=meta.get("name") or Path(path).stem.replace("_", " "),
                resume_path=path,
                match_score=min(100, score),
                matched_skills=matched_skills,
                relevant_excerpts=excerpts,
                reasoning=reasoning,
            )
        )

    candidates.sort(key=lambda x: -x.match_score)
    top = candidates[:k]
    logger.info("Found %d matching candidate(s).", len(top))
    output = JobMatchOutput(job_description=job_description, top_matches=top)
    return output.model_dump()


if __name__ == "__main__":
    import json
    import sys
    jd = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "Senior Python developer. 5+ years Python. Machine Learning experience required."
    )
    try:
        result = run_job_matcher(jd, k=10)
        print(json.dumps(result, indent=2))
    except FileNotFoundError as e:
        logger.error("%s", e)
        sys.exit(1)
    except RuntimeError as e:
        logger.error("%s", e)
        sys.exit(1)
