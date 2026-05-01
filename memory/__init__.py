"""
Conversation Memory
===================
Provides a shared conversation memory that the orchestrator and
sub-agents can read from.  Uses LangChain's ConversationBufferWindowMemory
to keep the last K exchanges, preventing unbounded context growth.

Design Decision
---------------
ConversationBufferWindowMemory (window K=10) was chosen over:
- ConversationBufferMemory: unbounded growth → token limit risk
- ConversationSummaryMemory: loses specific details (table numbers,
  menu items) that matter for restaurant operations
- ConversationEntityMemory: over-engineered for this use case

K=10 keeps ~10 user/assistant turns, which is enough for multi-step
interactions like "check availability → book table → ask about menu"
while staying well within the context window.
"""

from langchain.memory import ConversationBufferWindowMemory
import config


def create_memory() -> ConversationBufferWindowMemory:
    """Create a new conversation memory instance."""
    return ConversationBufferWindowMemory(
        k=config.MEMORY_WINDOW_K,
        memory_key="chat_history",
        return_messages=True,
        input_key="input",
        output_key="output",
    )
