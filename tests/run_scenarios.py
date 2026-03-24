from matching_agent import build_graph, AgentState

SCENARIOS = [
    "Find me candidates with React and 3+ years experience.",
    "Compare top 3.",
    "Why did John rank higher than Jane?",
    "Generate interview questions for the top candidate.",
    "Actually require 5+ years and AWS.",
]


def main() -> None:
    graph = build_graph()
    state = AgentState()
    for msg in SCENARIOS:
        print("\nYou:", msg)
        state = AgentState.model_validate(
            graph.invoke({**state.model_dump(), "last_user_message": msg})
        )
        print("\nAgent:\n", state.last_response)


if __name__ == "__main__":
    main()
