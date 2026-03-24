"""
Part A: RAG System Setup (LangChain).
Load resumes via DirectoryLoader + PyPDFLoader, chunk with RecursiveCharacterTextSplitter,
extract metadata per file, store in Chroma via LangChain.
"""

import logging
from pathlib import Path
from typing import Optional
import re

from pydantic import BaseModel, Field
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from embedding_utils import CHROMA_COLLECTION_NAME, CHROMA_PERSIST_DIR, require_openai_key

logger = logging.getLogger(__name__)


SECTION_HEADERS = re.compile(
    r"^(?:Education|Experience|Work\s*Experience|Skills|Summary|Projects|Certifications|Contact)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


class ResumeMetadata(BaseModel):
    """Structured metadata extracted from a resume."""

    name: str = Field(..., description="Candidate full name")
    skills: list[str] = Field(default_factory=list, description="List of skills")
    experience_years: Optional[int] = Field(None, description="Years of experience if parseable")
    education: Optional[str] = Field(None, description="Education summary")


def extract_metadata(text: str) -> ResumeMetadata:
    """Parse name, skills, experience_years, education from full resume text."""
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    name = ""
    skills: list[str] = []
    experience_years: Optional[int] = None
    education: Optional[str] = None

    for i, line in enumerate(lines):
        if re.match(r"^(?:name|full\s*name)\s*[:.]?\s*", line, re.I):
            name = re.sub(r"^(?:name|full\s*name)\s*[:.]?\s*", "", line, flags=re.I).strip()
            break
        if i == 0 and len(line) < 80 and not line.lower().startswith(
            ("summary", "objective", "education", "experience")
        ):
            name = line
            break
    if not name:
        name = "Unknown"

    in_skills = False
    for line in lines:
        if re.match(r"^skills\s*$", line, re.I):
            in_skills = True
            continue
        if in_skills:
            if SECTION_HEADERS.match(line) and not line.lower().startswith("skill"):
                break
            for part in re.split(r"[,;]|\s+-\s+", line):
                part = part.strip().strip("•-")
                if part and len(part) < 80:
                    skills.append(part)
        if re.match(r"^(?:experience|work)\s*$", line, re.I):
            in_skills = False

    years_match = re.search(r"(\d+)\s*\+\s*years?", text, re.I) or re.search(
        r"(\d+)\s*years?\s+(?:of\s+)?(?:experience|exp\.?)", text, re.I
    )
    if years_match:
        experience_years = int(years_match.group(1))

    in_edu = False
    edu_parts: list[str] = []
    for line in lines:
        if re.match(r"^education\s*$", line, re.I):
            in_edu = True
            continue
        if in_edu:
            if SECTION_HEADERS.match(line) and not line.lower().startswith("edu"):
                break
            edu_parts.append(line)
        if re.match(r"^(?:experience|work|skills)\s*$", line, re.I):
            in_edu = False
    if edu_parts:
        education = " ".join(edu_parts[:5]).strip()

    return ResumeMetadata(
        name=name,
        skills=skills[:50],
        experience_years=experience_years,
        education=education or None,
    )


def run_resume_rag(
    resumes_dir: Path | str,
    vector_db_path: str = CHROMA_PERSIST_DIR,
) -> None:
    """Load PDFs, chunk, enrich metadata, embed and store in Chroma via LangChain."""
    require_openai_key()
    resumes_dir = Path(resumes_dir).resolve()

    if not resumes_dir.exists():
        logger.error(
            "Resumes directory does not exist: %s\n"
            "  → Create a folder (e.g. 'Resume') and add PDF resumes.\n"
            "  → Or pass a path: python resume_rag.py /path/to/your/resumes",
            resumes_dir,
        )
        raise FileNotFoundError(f"Resumes directory not found: {resumes_dir}")

    if not resumes_dir.is_dir():
        logger.error("%s exists but is not a directory.", resumes_dir)
        raise NotADirectoryError(f"Not a directory: {resumes_dir}")

    logger.info("Loading PDFs from %s ...", resumes_dir)
    loader = DirectoryLoader(
        str(resumes_dir),
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
    )
    documents = loader.load()
    if not documents:
        logger.error(
            "No PDF files found in %s\n"
            "  → Add one or more .pdf resume files to that folder and run again.",
            resumes_dir,
        )
        raise ValueError(f"No PDF resumes found in {resumes_dir}")

    n_files = len({d.metadata.get("source") for d in documents})
    logger.info("Found %d PDF(s), %d page(s). Chunking and extracting metadata ...", n_files, len(documents))

    by_source: dict[str, list] = {}
    for doc in documents:
        source = doc.metadata.get("source", "unknown")
        by_source.setdefault(source, []).append(doc)
    file_metadata: dict[str, ResumeMetadata] = {}
    for source, docs in by_source.items():
        full_text = "\n".join(d.page_content for d in docs)
        file_metadata[source] = extract_metadata(full_text)

    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    logger.info("Created %d chunks. Embedding and storing in Chroma ...", len(chunks))

    for chunk in chunks:
        source = chunk.metadata.get("source", "")
        if source in file_metadata:
            m = file_metadata[source]
            chunk.metadata["path"] = source
            chunk.metadata["name"] = m.name
            chunk.metadata["skills"] = ",".join(m.skills) if m.skills else ""
            chunk.metadata["experience_years"] = (
                m.experience_years if m.experience_years is not None else -1
            )
            chunk.metadata["education"] = m.education or ""

    embedding = OpenAIEmbeddings(model="text-embedding-3-small")
    Chroma.from_documents(
        chunks,
        embedding,
        persist_directory=vector_db_path,
        collection_name=CHROMA_COLLECTION_NAME,
    )
    logger.info(
        "Done. Stored %d chunks in %s. You can now run: python job_matcher.py \"your job description\"",
        len(chunks),
        vector_db_path,
    )


if __name__ == "__main__":
    import sys
    resumes_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("Resume")
    try:
        run_resume_rag(resumes_dir)
    except (FileNotFoundError, NotADirectoryError, ValueError) as e:
        logger.error("%s", e)
        sys.exit(1)
    except RuntimeError as e:
        logger.error("%s", e)
        sys.exit(1)
