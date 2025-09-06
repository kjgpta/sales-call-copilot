import os
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "sbert")  # sbert|none
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "offline")          # offline|openai
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

HYBRID_DENSE_WEIGHT = float(os.getenv("HYBRID_DENSE_WEIGHT", "0.6"))
MAX_CHUNK_TOKENS = int(os.getenv("MAX_CHUNK_TOKENS", "256"))
CHUNK_OVERLAP_RATIO = float(os.getenv("CHUNK_OVERLAP_RATIO", "0.25"))

DB_PATH = os.getenv("DB_PATH", "./index/copilot.db")
INDEX_DIR = os.getenv("INDEX_DIR", "./index")

RERANKER = os.getenv("RERANKER", "cross-encoder")  # cross-encoder|none
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
AUTO_INGEST = os.getenv('AUTO_INGEST', '1') == '1'
