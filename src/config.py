import os
from pathlib import Path

from dotenv import load_dotenv

# Reads API keys from a .env file in the project folder, if there is one
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

# Project root, so scripts work no matter which folder you run them from
ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
URLS_FILE = DATA_DIR / "web" / "urls.txt"
SQL_DB = DATA_DIR / "sql" / "ml_models.db"
EVAL_DIR = DATA_DIR / "eval"
FAISS_DIR = ROOT / "faiss_index"
CHROMA_DIR = ROOT / "chroma_db"
REPORTS_DIR = ROOT / "reports"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# "ollama" runs locally; "groq" is fast but needs GROQ_API_KEY set
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq" if os.getenv("GROQ_API_KEY") else "ollama")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:9b")
GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
# The judge is a DIFFERENT model family from the generator: models grade their own answers too kindly
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "openai/gpt-oss-120b")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Retrieval: how many candidates each retriever fetches, and how many survive reranking
FETCH_K = 20
TOP_K = 5
