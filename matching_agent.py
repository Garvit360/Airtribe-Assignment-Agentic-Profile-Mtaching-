import re
from typing import Optional

from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END

from agent_tools import (
    CandidateShortlistItem,
    RequirementSpec,
    analyze_candidate_fit,
    build_candidate_id,
    compare_candidates,
    extract_requirements,
    generate_improvement_suggestions,
    generate_interview_questions,
    search_resumes,
)


class Message(BaseModel):
    role: str = Field(..., description="user | assistant")
    content: str = Field(..., description="message content")


class AgentState(BaseModel):
    conversation_history: list[Message] = Field(default_factory=list)
    job_description: Optional[str] = None
    requirements: Optional[RequirementSpec] = None
    candidate_pool: list[dict] = Field(default_factory=list)
    shortlist: list[CandidateShortlistItem] = Field(default_factory=list)
    last_user_intent: Optional[str] = None
    last_user_message: Optional[str] = None
    last_response: Optional[str] = None


# =============================================================================
# INTENT PARSING
# =============================================================================

def infer_intent(user_msg: str, has_jd: bool) -> str:
    msg = user_msg.lower().strip()
    if any(k in msg for k in ["compare", " vs ", " versus "]):
        return "compare"
    if msg.startswith("why") or "rank higher" in msg or "why did" in msg:
        return "why"
    if "interview" in msg or "questions" in msg:
        return "questions"
    if any(k in msg for k in ["actually", "refine", "update", "change", "require "]):
        return "refine" if has_jd else "search"
    if any(k in msg for k in ["find", "search", "candidates", "match", "looking for"]):
        return "search"
    return "search" if not has_jd else "refine"


def parse_compare_targets(user_msg: str, shortlist: list[CandidateShortlistItem]) -> list[str]:
    msg = user_msg.lower()
    top_match = re.search(r"top\s+(\d+)", msg)
    if top_match:
        count = int(top_match.group(1))
        return [c.candidate_id for c in shortlist[:count]]

    if " vs " in msg:
        parts = msg.split(" vs ")
    elif " versus " in msg:
        parts = msg.split(" versus ")
    elif "compare" in msg and " and " in msg:
        parts = msg.split("compare", 1)[1].split(" and ")
    else:
        parts = []

    targets: list[str] = []
    for part in parts:
        name = part.strip().strip(".?")
        if not name:
            continue
        candidate = match_candidate_by_query(name, shortlist)
        if candidate:
            targets.append(candidate.candidate_id)
    return targets


def match_candidate_by_query(query: str, shortlist: list[CandidateShortlistItem]) -> Optional[CandidateShortlistItem]:
    q = query.lower().strip()
    for c in shortlist:
        if c.candidate_id.lower() == q:
            return c
    tokens = [t for t in re.split(r"\s+", q) if t]
    for c in shortlist:
        name = c.name.lower()
        if all(t in name for t in tokens):
            return c
    return None


def parse_why_targets(user_msg: str, shortlist: list[CandidateShortlistItem]) -> list[CandidateShortlistItem]:
    msg = user_msg.lower()
    m = re.search(r"why did (.+?) rank higher than (.+?)[?\.]?", msg)
    if m:
        first = match_candidate_by_query(m.group(1), shortlist)
        second = match_candidate_by_query(m.group(2), shortlist)
        return [c for c in [first, second] if c]
    return shortlist[:2]


def parse_question_target(user_msg: str, shortlist: list[CandidateShortlistItem]) -> Optional[CandidateShortlistItem]:
    msg = user_msg.lower()
    for c in shortlist:
        if c.name.lower() in msg or c.candidate_id.lower() in msg:
            return c
    return shortlist[0] if shortlist else None


# =============================================================================
# GRAPH NODES
# =============================================================================

def parse_input_node(state: dict) -> dict:
    st = AgentState.model_validate(state)
    user_msg = (st.last_user_message or "").strip()
    intent = infer_intent(user_msg, has_jd=bool(st.job_description))

    history = list(st.conversation_history)
    if user_msg:
        history.append(Message(role="user", content=user_msg))

    updates: dict = {
        "conversation_history": history,
        "last_user_intent": intent,
    }

    if intent == "search":
        updates["job_description"] = user_msg
    elif intent == "refine":
        base_jd = st.job_description or ""
        updates["job_description"] = (base_jd + "\nRefinement: " + user_msg).strip()

    return updates


def extract_requirements_node(state: dict) -> dict:
    st = AgentState.model_validate(state)
    if not st.job_description:
        return {"last_response": "Please provide a job description to extract requirements."}
    requirements = extract_requirements(st.job_description)
    return {"requirements": requirements}


def search_resumes_node(state: dict) -> dict:
    st = AgentState.model_validate(state)
    if not st.job_description:
        return {"last_response": "Please provide a job description to search resumes."}
    results = search_resumes(st.job_description, k=10)
    return {"candidate_pool": results}


def rank_candidates_node(state: dict) -> dict:
    st = AgentState.model_validate(state)
    if not st.job_description or not st.candidate_pool:
        return {"last_response": "No candidates available to rank yet."}

    shortlist: list[CandidateShortlistItem] = []
    for idx, candidate in enumerate(st.candidate_pool[:10]):
        analysis = analyze_candidate_fit(st.job_description, candidate)
        match_score = int(candidate.get("match_score", 0))
        combined = int((match_score * 0.6) + (analysis.fit_score * 0.4))
        if combined >= 80:
            recommendation = "hire"
        elif combined >= 60:
            recommendation = "borderline"
        else:
            recommendation = "no-hire"

        suggestions: list[str] = []
        if recommendation == "borderline":
            suggestions = generate_improvement_suggestions(
                st.job_description,
                candidate.get("candidate_name", "Unknown"),
                analysis.gaps,
            )

        shortlist.append(
            CandidateShortlistItem(
                candidate_id=build_candidate_id(
                    candidate.get("candidate_name", "Unknown"),
                    candidate.get("resume_path", ""),
                    idx,
                ),
                name=candidate.get("candidate_name", "Unknown"),
                resume_path=candidate.get("resume_path", ""),
                match_score=match_score,
                matched_skills=candidate.get("matched_skills", []),
                relevant_excerpts=candidate.get("relevant_excerpts", []),
                reasoning=candidate.get("reasoning", ""),
                round2_summary=analysis.summary,
                strengths=analysis.strengths,
                gaps=analysis.gaps,
                fit_score=analysis.fit_score,
                combined_score=combined,
                recommendation=recommendation,
                improvement_suggestions=suggestions,
            )
        )

    shortlist.sort(key=lambda c: c.combined_score, reverse=True)
    return {"shortlist": shortlist}


def generate_report_node(state: dict) -> dict:
    st = AgentState.model_validate(state)
    if not st.shortlist:
        return {"last_response": "No ranked candidates yet. Try searching first."}

    lines = ["Top candidates (ranked):"]
    for c in st.shortlist[:5]:
        lines.append(
            f"- {c.name} (id: {c.candidate_id}) | combined={c.combined_score} | "
            f"match={c.match_score} | fit={c.fit_score} | {c.recommendation}"
        )

    if st.requirements:
        lines.append("")
        lines.append("Must-have requirements:")
        lines.extend([f"- {req}" for req in st.requirements.must_have[:8]])
        if st.requirements.nice_to_have:
            lines.append("Nice-to-have requirements:")
            lines.extend([f"- {req}" for req in st.requirements.nice_to_have[:8]])

    lines.append("")
    lines.append(
        "You can say: 'compare top 3', 'why did X rank higher than Y', "
        "or 'generate interview questions for <name>'."
    )

    return {"last_response": "\n".join(lines)}


def compare_candidates_node(state: dict) -> dict:
    st = AgentState.model_validate(state)
    if not st.shortlist:
        return {"last_response": "No candidates to compare yet. Run a search first."}
    ids = parse_compare_targets(st.last_user_message or "", st.shortlist)
    if not ids:
        ids = [c.candidate_id for c in st.shortlist[:2]]
    comparison = compare_candidates(ids, st.shortlist)
    return {"last_response": comparison}


def explain_ranking_node(state: dict) -> dict:
    st = AgentState.model_validate(state)
    if len(st.shortlist) < 2:
        return {"last_response": "Not enough candidates to explain ranking. Run a search first."}
    cands = parse_why_targets(st.last_user_message or "", st.shortlist)
    if len(cands) < 2:
        cands = st.shortlist[:2]

    higher, lower = cands[0], cands[1]
    lines = [
        f"{higher.name} ranked higher than {lower.name} because:",
        f"- Higher combined score ({higher.combined_score} vs {lower.combined_score}).",
        f"- Strengths: {', '.join(higher.strengths[:4]) or 'N/A'}.",
        f"- Gaps for {lower.name}: {', '.join(lower.gaps[:4]) or 'N/A'}.",
    ]
    return {"last_response": "\n".join(lines)}


def interview_questions_node(state: dict) -> dict:
    st = AgentState.model_validate(state)
    if not st.shortlist:
        return {"last_response": "No candidates available yet. Run a search first."}
    target = parse_question_target(st.last_user_message or "", st.shortlist)
    if not target:
        return {"last_response": "Please specify a candidate for interview questions."}
    questions = generate_interview_questions(target)
    lines = [f"Interview questions for {target.name}:"]
    lines.extend([f"- {q}" for q in questions])
    return {"last_response": "\n".join(lines)}


def fallback_node(state: dict) -> dict:
    return {
        "last_response": (
            "I can help with: searching candidates, comparing top matches, "
            "explaining rankings, and generating interview questions. "
            "Try: 'Find me candidates with React and 3+ years experience.'"
        )
    }


def human_feedback_node(state: dict) -> dict:
    st = AgentState.model_validate(state)
    history = list(st.conversation_history)
    if st.last_response:
        history.append(Message(role="assistant", content=st.last_response))
    return {"conversation_history": history}


# =============================================================================
# GRAPH SETUP
# =============================================================================

def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("parse_input", parse_input_node)
    graph.add_node("extract_requirements", extract_requirements_node)
    graph.add_node("search_resumes", search_resumes_node)
    graph.add_node("rank_candidates", rank_candidates_node)
    graph.add_node("generate_report", generate_report_node)
    graph.add_node("compare_candidates", compare_candidates_node)
    graph.add_node("explain_ranking", explain_ranking_node)
    graph.add_node("interview_questions", interview_questions_node)
    graph.add_node("fallback", fallback_node)
    graph.add_node("human_feedback", human_feedback_node)

    graph.set_entry_point("parse_input")

    def route_from_intent(state: dict) -> str:
        st = AgentState.model_validate(state)
        intent = st.last_user_intent or ""
        if intent in {"search", "refine"}:
            return "extract_requirements"
        if intent == "compare":
            return "compare_candidates"
        if intent == "why":
            return "explain_ranking"
        if intent == "questions":
            return "interview_questions"
        return "fallback"

    graph.add_conditional_edges(
        "parse_input",
        route_from_intent,
        {
            "extract_requirements": "extract_requirements",
            "compare_candidates": "compare_candidates",
            "explain_ranking": "explain_ranking",
            "interview_questions": "interview_questions",
            "fallback": "fallback",
        },
    )

    graph.add_edge("extract_requirements", "search_resumes")
    graph.add_edge("search_resumes", "rank_candidates")
    graph.add_edge("rank_candidates", "generate_report")

    graph.add_edge("generate_report", "human_feedback")
    graph.add_edge("compare_candidates", "human_feedback")
    graph.add_edge("explain_ranking", "human_feedback")
    graph.add_edge("interview_questions", "human_feedback")
    graph.add_edge("fallback", "human_feedback")

    graph.add_edge("human_feedback", END)

    return graph.compile()


# =============================================================================
# CLI
# =============================================================================

def run_cli() -> None:
    graph = build_graph()
    state = AgentState()
    print("Matching Agent CLI. Type 'exit' to quit.")

    while True:
        user_msg = input("\nYou: ").strip()
        if not user_msg:
            continue
        if user_msg.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        state = AgentState.model_validate(
            graph.invoke({**state.model_dump(), "last_user_message": user_msg})
        )
        if state.last_response:
            print(f"\nAgent:\n{state.last_response}")


if __name__ == "__main__":
    run_cli()
