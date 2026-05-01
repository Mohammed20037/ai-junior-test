"""
Configuration for the NovaBite Multi-Agent RAG System.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ── LLM ──────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
USE_FAKE_LLM = False  # Set to False when you have valid API keys
LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0.0          # deterministic for tool-calling & RAG
CREATIVE_TEMPERATURE = 0.3     # slightly warmer for friendly responses

def get_llm(temperature=0.0):
    """Get LLM instance, either real or fake based on configuration."""
    if USE_FAKE_LLM:
        from langchain_community.fake import FakeListLLM
        # Return a fake LLM that provides deterministic responses
        # We'll use a single response for simplicity; in practice, you might want multiple
        return FakeListLLM(responses=["fake response"])
    else:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=LLM_MODEL,
            temperature=temperature,
            openai_api_key=OPENAI_API_KEY,
        )

# ── Embeddings ───────────────────────────────────────
USE_FAKE_EMBEDDINGS = False  # Set to False when you have valid API keys
EMBEDDING_MODEL = "text-embedding-3-small"  # Only used if USE_FAKE_EMBEDDINGS = False

def get_embeddings():
    """Get embedding instance, either real or fake based on configuration."""
    if USE_FAKE_EMBEDDINGS:
        from langchain_community.embeddings import FakeEmbeddings
        return FakeEmbeddings(size=384)  # Size must match the embedding dimension
    else:
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            openai_api_key=OPENAI_API_KEY,
        )

# ── Vector Store ─────────────────────────────────────
VECTOR_STORE_PATH = os.path.join(os.path.dirname(__file__), "data", "vector_store")

# ── RAG ──────────────────────────────────────────────
CHUNK_SIZE = 500               # tokens-ish characters per chunk
CHUNK_OVERLAP = 50             # overlap to preserve context at boundaries
RETRIEVAL_TOP_K = 4            # number of chunks to retrieve

# ── Memory ───────────────────────────────────────────
MEMORY_WINDOW_K = 10           # keep last 10 conversation turns

# ── Knowledge Base Path ──────────────────────────────
KNOWLEDGE_BASE_DIR = os.path.join(os.path.dirname(__file__), "data", "knowledge_base")