"""
NovaBite Multi-Agent RAG System – Main Entry Point
====================================================
Interactive CLI that boots all agents and runs a conversation loop.

Usage:
    python main.py              # Interactive chat mode
    python main.py --rebuild    # Force rebuild the vector store
"""

import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(__file__))

from rag.ingestion import ingest_documents
from rag.retriever import get_retriever
from agents.rag_agent import RAGAgent
from agents.operations_agent import OperationsAgent
from agents.orchestrator import Orchestrator
import config


BANNER = """
+======================================================+
|         NovaBite AI Restaurant Assistant             |
|                                                      |
|   Multi-Agent RAG System                             |
|   - Menu & Policy Knowledge (RAG)                    |
|   - Table Availability & Booking (Tools)             |
|   - Today's Specials (Tools)                         |
|                                                      |
|   Type 'quit' or 'exit' to end the conversation.     |
|   Type 'clear' to reset conversation memory.         |
+======================================================+
"""


def build_system(force_rebuild: bool = False) -> Orchestrator:
    """
    Initialize all components and return the orchestrator.

    Steps:
    1. Ingest documents -> FAISS vector store
    2. Create retriever from vector store
    3. Initialize RAG sub-agent with retriever
    4. Initialize Operations sub-agent with tools
    5. Initialize Orchestrator with both sub-agents
    """
    print("\n[...] Initializing NovaBite AI System [...]\n")

    # Step 1: Document ingestion
    print("[Step 1/4] Document Ingestion")
    vector_store = ingest_documents(force_rebuild=force_rebuild)

    # Step 2: Retriever
    print("[Step 2/4] Building Retriever")
    retriever = get_retriever(vector_store)
    print(f"[Retriever] Top-K = {config.RETRIEVAL_TOP_K}")

    # Step 3: RAG Agent
    print("[Step 3/4] Initializing RAG Agent")
    rag_agent = RAGAgent(retriever)
    print("[RAG Agent] Ready [OK]")

    # Step 4: Operations Agent
    print("[Step 4/4] Initializing Operations Agent")
    ops_agent = OperationsAgent()
    print("[Operations Agent] Ready [OK]")

    # Step 5: Orchestrator
    orchestrator = Orchestrator(rag_agent, ops_agent)
    print("\n[OK] All systems initialized. Ready to chat!\n")

    return orchestrator


def main():
    """Run the interactive chat loop."""
    force_rebuild = "--rebuild" in sys.argv

    # Check for API key
    if not config.OPENAI_API_KEY:
        print("❌ Error: OPENAI_API_KEY not set.")
        print("   Create a .env file with: OPENAI_API_KEY=your-key-here")
        print("   Or set the environment variable directly.")
        sys.exit(1)

    # Build the system
    orchestrator = build_system(force_rebuild=force_rebuild)

    # Print banner
    print(BANNER)

    # Chat loop
    while True:
        try:
            user_input = input("\n[You] ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye! Thanks for visiting NovaBite.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("\nGoodbye! Thanks for visiting NovaBite. 🍽️")
            break

        if user_input.lower() == "clear":
            orchestrator.memory.clear()
            print("🔄 Conversation memory cleared.")
            continue

        # Get response
        try:
            response = orchestrator.chat(user_input)
            print(f"\n[NovaBite] {response}")
        except Exception as e:
            print(f"\n[Error] {e}")
            print("   Please try again or rephrase your question.")


if __name__ == "__main__":
    main()
