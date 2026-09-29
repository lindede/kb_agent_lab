from __future__ import annotations

from pathlib import Path
import os

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = Path(os.getenv("DOCS_DIR", ROOT / "docs")).expanduser().resolve()
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", ROOT / "data" / "chroma")).expanduser().resolve()
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "kb_agent_lab")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small").strip()
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "1"))

# local = Chroma ONNX MiniLM (no remote embedding API)
# openai = OpenAI-compatible /embeddings (many China gateways do NOT open this)
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local").strip().lower()

NOTES_DIR = Path(os.getenv("NOTES_DIR", ROOT / "data" / "notes")).expanduser().resolve()
AGENT_MAX_ITERATIONS = int(os.getenv("AGENT_MAX_ITERATIONS", "4"))
ASK_MODE = os.getenv("ASK_MODE", "agent").strip().lower()  # agent | rag

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "600"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))
RETRIEVE_K = int(os.getenv("RETRIEVE_K", "4"))


def has_llm_key() -> bool:
    return bool(OPENAI_API_KEY)


def use_openai_embeddings() -> bool:
    return EMBEDDING_PROVIDER in {"openai", "remote"} and has_llm_key()
