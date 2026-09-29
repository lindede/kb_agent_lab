from pathlib import Path

from dotenv import load_dotenv
import os

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = Path(os.getenv("DOCS_DIR", ROOT / "docs"))
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", ROOT / "data" / "chroma"))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
