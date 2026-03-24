# Agentic Profile Match (LangGraph)

LangGraph-based matching agent that ranks resumes against a job description, supports iterative refinement, and explains its decisions via a CLI.

## Quick Start

### 1. Install dependencies

```bash
cd "/Users/garvit/Airtribe Assignment/Agentic Profile Mtaching"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Set your OpenAI API key

```bash
export OPENAI_API_KEY="your-key-here"
```

### 3. Index resumes (RAG)

Unzip the bundled dataset and index locally:

```bash
unzip Resume.zip

# Uses local Resume/ by default
python resume_rag.py
```

### 4. Run the matching agent

```bash
python matching_agent.py
```

Example prompts:
- `Find me candidates with React and 3+ years experience`
- `Compare top 3`
- `Why did John rank higher than Jane?`
- `Generate interview questions for <candidate name>`

## State Machine Diagram

See `docs/state_machine.mmd` for the Mermaid diagram.

## Tests / Scenarios

Conversation flows are documented in `tests/conversation_flows.md`.

## Notes

- The agent uses `gpt-4o-mini` by default for extraction and analysis.
- The RAG pipeline uses Chroma + OpenAI embeddings as in the earlier milestone.
