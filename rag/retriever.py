"""
Retriever
=========
Wraps the FAISS vector store in a LangChain retriever with top-k
configuration and optional score-threshold filtering.
"""

from langchain_community.vectorstores import FAISS
import config


def get_retriever(vector_store: FAISS):
    """
    Build a retriever from the vector store.

    Uses similarity search with top-k retrieval.  The relatively small k
    (default 4) keeps the context window lean and reduces the chance of
    injecting irrelevant chunks that could cause hallucination.
    """
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": config.RETRIEVAL_TOP_K},
    )
    return retriever
