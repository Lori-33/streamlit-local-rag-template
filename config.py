"""Central configuration for the local RAG template."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DOCS_FOLDER = Path(os.getenv("LOCAL_RAG_DOCS_FOLDER", PROJECT_ROOT / "sample_docs"))
OBJECTION_DOCS_FOLDER = Path(os.getenv("LOCAL_RAG_OBJECTION_DOCS_FOLDER", PROJECT_ROOT / "sample_objections"))
DOCS_FOLDERS = [DOCS_FOLDER, OBJECTION_DOCS_FOLDER]
VECTOR_STORE_DIR = Path(os.getenv("LOCAL_RAG_VECTOR_STORE_DIR", PROJECT_ROOT / "vector_store"))

LMSTUDIO_BASE_URL = os.getenv("LMSTUDIO_URL", "http://localhost:1234")
LMSTUDIO_MODEL = os.getenv("LMSTUDIO_MODEL", "qwen3-8b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-bge-m3")

# Optional OpenAI-compatible cloud endpoint. Leave empty for fully local use.
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = "deepseek-chat"

APP_TITLE = "Local RAG Assistant"
APP_PORT = int(os.getenv("APP_PORT", "8501"))
