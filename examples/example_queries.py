"""
Example Queries & Expected Outputs
====================================
This script demonstrates the NovaBite AI system with various query types.
Run: python examples/example_queries.py

These examples cover all required agent routing paths:
- RAG (knowledge base)
- Operations (tool calling)
- General conversation
- Memory continuity across turns
- Hallucination prevention
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from main import build_system


# ── Test Scenarios ────────────────────────────────────────────────

DEMO_QUERIES = [
    # ── RAG: Menu Queries ────────────────────────────────
    {
        "category": "RAG – Menu",
        "queries": [
            "Do you have vegan pasta?",
            "Is the chicken grilled or fried?",
            "What allergens are in the Classic Beef Burger?",
            "What desserts do you have?",
            "Do you serve sushi?",
        ],
    },
    # ── RAG: Policies & Hours ────────────────────────────
    {
        "category": "RAG – Policies & Hours",
        "queries": [
            "What are your opening hours on weekends?",
            "Do you host birthday events?",
            "What's included in the premium catering package?",
            "What's your refund policy for takeout orders?",
            "How does the loyalty program work?",
        ],
    },
    # ── Operations: Tool Calling ─────────────────────────
    {
        "category": "Operations – Tool Calling",
        "queries": [
            "Is there a table available at the downtown branch tonight at 7pm?",
            "What's today's special at the Brooklyn branch?",
            "Book a table for 4 at Midtown on 2025-07-15 at 19:00 under the name John",
        ],
    },
    # ── General Conversation ─────────────────────────────
    {
        "category": "General",
        "queries": [
            "Hi there!",
            "Thanks for your help!",
        ],
    },
    # ── Memory Continuity ────────────────────────────────
    {
        "category": "Memory – Follow-up",
        "queries": [
            "What vegan options do you have?",
            "Which of those is gluten-free?",         # requires memory of previous answer
            "How much does it cost?",                  # refers to item from previous context
        ],
    },
    # ── Hallucination Prevention ─────────────────────────
    {
        "category": "Hallucination Test",
        "queries": [
            "Do you have a wagyu steak on the menu?",     # NOT on the menu (only as daily special)
            "Can I get a margherita pizza?",               # NOT on the menu
            "What's the price of the chicken tikka masala?",  # does NOT exist
        ],
    },
]


def run_demo():
    """Run all demo queries and print results."""
    orchestrator = build_system()

    for scenario in DEMO_QUERIES:
        print(f"\n{'='*60}")
        print(f"  Category: {scenario['category']}")
        print(f"{'='*60}")

        for query in scenario["queries"]:
            print(f"\n👤 User: {query}")
            try:
                response = orchestrator.chat(query)
                print(f"🤖 NovaBite: {response}")
            except Exception as e:
                print(f"❌ Error: {e}")

        # Clear memory between scenarios (except the Memory test)
        if "Memory" not in scenario["category"]:
            orchestrator.memory.clear()

    print(f"\n{'='*60}")
    print("  Demo Complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_demo()
