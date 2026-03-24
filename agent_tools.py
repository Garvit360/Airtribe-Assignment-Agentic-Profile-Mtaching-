import json
import re
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, ValidationError
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from embedding_utils import require_openai_key
from job_matcher import run_job_matcher

CHAT_MODEL: str = "gpt-4o-mini"


class RequirementSpec(BaseModel):
    must_have: list[str] = Field(default_factory=list, description="Must-have requirements")
    nice_to_have: list[str] = Field(default_factory=list, description="Nice-to-have requirements")
    years_experience: Optional[int] = Field(None, description="Minimum years of experience")


class CandidateAnalysis(BaseModel):
    summary: str = Field(..., description="Short summary of fit")
    strengths: list[str] = Field(default_factory=list, description="Key strengths")
    gaps: list[str] = Field(default_factory=list, description="Key gaps")
    fit_score: int = Field(..., ge=0, le=100, description="Fit score 0-100")


class CandidateShortlistItem(BaseModel):
    candidate_id: str = Field(..., description="Stable identifier")
    name: str = Field(..., description="Candidate name")
    resume_path: str = Field(..., description="Path to resume file")
    match_score: int = Field(..., ge=0, le=100, description="Initial match score")
    matched_skills: list[str] = Field(default_factory=list)
    relevant_excerpts: list[str] = Field(default_factory=list)
    reasoning: str = Field(..., description="Round 1 reasoning")
    round2_summary: str = Field(..., description="Round 2 summary")
    strengths: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    fit_score: int = Field(..., ge=0, le=100)
    combined_score: int = Field(..., ge=0, le=100)
    recommendation: str = Field(..., description="hire | borderline | no-hire")
    city: Optional[str] = Field(None, description="Best-effort inferred city")
    improvement_suggestions: list[str] = Field(default_factory=list)


class InterviewQuestions(BaseModel):
    questions: list[str] = Field(default_factory=list, description="Screening questions")


def _get_llm() -> ChatOpenAI:
    require_openai_key()
    return ChatOpenAI(model=CHAT_MODEL, temperature=0.2)


def infer_city_from_text(excerpts: list[str]) -> Optional[str]:
    patterns = [
        r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*,\s*[A-Z]{2})\b",
        r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*,\s*(?:USA|United States)\b",
    ]
    for text in excerpts:
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()
    return None


def _extract_json_text(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start = min([idx for idx in [text.find("{"), text.find("[")] if idx != -1], default=-1)
    if start > 0:
        text = text[start:]
    return text.strip()


def _load_json_payload(raw: str) -> dict:
    cleaned = _extract_json_text(raw)
    parsed = json.loads(cleaned)
    if not isinstance(parsed, dict):
        raise ValueError("Expected JSON object payload from model response.")
    return parsed


def extract_requirements(jd: str) -> RequirementSpec:
    llm = _get_llm()
    system = (
        "You extract structured requirements from job descriptions. "
        "Return ONLY JSON matching this schema: "
        "{\"must_have\": [string], \"nice_to_have\": [string], \"years_experience\": number|null}."
    )
    human = f"Job description:\n{jd}\n\nReturn JSON only."
    response = llm.invoke([SystemMessage(content=system), HumanMessage(content=human)]).content
    payload = _load_json_payload(response)
    return RequirementSpec.model_validate(payload)


def search_resumes(jd: str, k: int = 10) -> list[dict]:
    output = run_job_matcher(jd, k=k)
    return output.get("top_matches", [])


def analyze_candidate_fit(jd: str, candidate: dict) -> CandidateAnalysis:
    llm = _get_llm()
    system = (
        "You are a hiring assistant. Analyze candidate fit for the job description. "
        "Return ONLY JSON: {\"summary\": string, \"strengths\": [string], "
        "\"gaps\": [string], \"fit_score\": number 0-100}."
    )
    human = (
        "Job description:\n"
        f"{jd}\n\n"
        "Candidate info:\n"
        f"Name: {candidate.get('candidate_name', 'Unknown')}\n"
        f"Matched skills: {candidate.get('matched_skills', [])}\n"
        f"Relevant excerpts: {candidate.get('relevant_excerpts', [])}\n"
        f"Reasoning: {candidate.get('reasoning', '')}\n\n"
        "Return JSON only."
    )
    response = llm.invoke([SystemMessage(content=system), HumanMessage(content=human)]).content
    try:
        payload = _load_json_payload(response)
        return CandidateAnalysis.model_validate(payload)
    except (ValueError, json.JSONDecodeError, ValidationError):
        matched = candidate.get("matched_skills", [])
        return CandidateAnalysis(
            summary="Fit estimated from semantic match and matched skills.",
            strengths=[f"Matched skill: {s}" for s in matched[:3]],
            gaps=[],
            fit_score=int(candidate.get("match_score", 50)),
        )


def generate_improvement_suggestions(jd: str, candidate_name: str, gaps: list[str]) -> list[str]:
    llm = _get_llm()
    system = (
        "You suggest concise, actionable improvement ideas for borderline candidates. "
        "Return ONLY JSON: {\"suggestions\": [string]}."
    )
    human = (
        "Job description:\n"
        f"{jd}\n\n"
        f"Candidate: {candidate_name}\n"
        f"Gaps: {gaps}\n\n"
        "Provide 2-4 suggestions. Return JSON only."
    )
    response = llm.invoke([SystemMessage(content=system), HumanMessage(content=human)]).content
    payload = _load_json_payload(response)
    return [s for s in payload.get("suggestions", []) if isinstance(s, str)]


def compare_candidates(candidate_ids: list[str], shortlist: list[CandidateShortlistItem]) -> str:
    selected = [c for c in shortlist if c.candidate_id in set(candidate_ids)]
    if len(selected) < 2:
        return "Please provide at least two valid candidate IDs to compare."
    llm = _get_llm()
    system = (
        "You compare candidates side by side for a hiring decision. "
        "Return a short textual comparison with a final recommendation order."
    )
    lines = []
    for c in selected:
        lines.append(
            f"- {c.name} (id: {c.candidate_id}) | match_score={c.match_score} | "
            f"fit_score={c.fit_score} | strengths={c.strengths} | gaps={c.gaps}"
        )
    human = "Candidates:\n" + "\n".join(lines) + "\n\nProvide comparison."
    response = llm.invoke([SystemMessage(content=system), HumanMessage(content=human)]).content
    return response.strip()


def generate_interview_questions(candidate: CandidateShortlistItem) -> list[str]:
    llm = _get_llm()
    system = (
        "You generate interview screening questions tailored to a candidate and role. "
        "Return ONLY JSON: {\"questions\": [string]}."
    )
    human = (
        "Candidate info:\n"
        f"Name: {candidate.name}\n"
        f"Strengths: {candidate.strengths}\n"
        f"Gaps: {candidate.gaps}\n"
        f"Summary: {candidate.round2_summary}\n\n"
        "Provide 5-7 concise questions. Return JSON only."
    )
    response = llm.invoke([SystemMessage(content=system), HumanMessage(content=human)]).content
    payload = _load_json_payload(response)
    parsed = InterviewQuestions.model_validate(payload)
    return parsed.questions


def build_candidate_id(name: str, resume_path: str, index: int) -> str:
    base = Path(resume_path).stem if resume_path else name
    slug = "".join(ch.lower() for ch in base if ch.isalnum() or ch in ("-", "_"))
    slug = slug.replace("_", "-")
    if not slug:
        slug = f"candidate-{index + 1}"
    return f"{index + 1}-{slug}"
