"""
Document Ingestion Pipeline
============================
Reads knowledge-base files, chunks them, embeds them, and persists
a FAISS vector store to disk.

Design Decisions
----------------
* **Chunking – RecursiveCharacterTextSplitter**
  Chosen because restaurant documents contain mixed formats (markdown
  headings, bullet lists, paragraphs). Recursive splitting first tries
  double-newlines (section boundaries), then single newlines, then
  sentences, preserving semantic coherence better than naive fixed-size
  splitting.

* **Chunk size = 500 chars, overlap = 50 chars**
  Menu items and policy sections are short (100-300 chars each).
  A 500-char chunk comfortably holds one complete item with its
  metadata (price, allergens, description) while staying small enough
  for precise retrieval.  50-char overlap prevents losing context at
  chunk boundaries.

* **Embedding model – text-embedding-3-small**
  Best balance of cost and quality for a restaurant-scale knowledge
  base.  1536-dim vectors keep the FAISS index compact.

* **Vector store – FAISS**
  Lightweight, runs in-process, zero infrastructure.  Perfect for a
  knowledge base of this size (<1 000 chunks).
"""

import os
import glob
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import config

def ingest_documents(force_rebuild: bool = False) -> FAISS:
    """
    Load knowledge-base documents, chunk, embed, and return a FAISS
    vector store.  Persists to disk so subsequent runs skip embedding.

    Parameters
    ----------
    force_rebuild : bool
        If True, re-ingest even if a persisted store already exists.

    Returns
    -------
    FAISS vector store instance.
    """
    embeddings = config.get_embeddings()

    # ── Return cached store if available ──────────────────────────
    if os.path.exists(config.VECTOR_STORE_PATH) and not force_rebuild:
        print("[Ingestion] Loading existing vector store from disk …")
        return FAISS.load_local(
            config.VECTOR_STORE_PATH,
            embeddings,
            allow_dangerous_deserialization=True,
        )

    # ── Load raw documents ────────────────────────────────────────
    kb_dir = config.KNOWLEDGE_BASE_DIR
    files = glob.glob(os.path.join(kb_dir, "*.txt"))
    if not files:
        raise FileNotFoundError(f"No .txt files found in {kb_dir}")

    all_docs = []
    for fpath in files:
        loader = TextLoader(fpath, encoding="utf-8")
        docs = loader.load()
        # Tag each doc with its source filename for traceability
        for doc in docs:
            doc.metadata["source_file"] = os.path.basename(fpath)
        all_docs.extend(docs)

    print(f"[Ingestion] Loaded {len(all_docs)} document(s) from {len(files)} file(s).")

    # ── Chunk ─────────────────────────────────────────────────────
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(all_docs)
    print(f"[Ingestion] Split into {len(chunks)} chunks (size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP}).")

    # ── Embed & persist ───────────────────────────────────────────
    vector_store = FAISS.from_documents(chunks, embeddings)
    os.makedirs(config.VECTOR_STORE_PATH, exist_ok=True)
    vector_store.save_local(config.VECTOR_STORE_PATH)
    print(f"[Ingestion] Vector store saved to {config.VECTOR_STORE_PATH}")

    return vector_store