"""
Main Orchestrator Agent
========================
The central coordinator of the NovaBite multi-agent system.

Responsibilities
-----------------
1. Classify user intent into categories (knowledge, operations, general)
2. Route requests to the appropriate sub-agent
3. Maintain shared conversation memory across all interactions
4. Merge and validate sub-agent responses
5. Handle ambiguity by asking clarifying questions
6. Prevent hallucinated outputs by enforcing grounding rules

The orchestrator does NOT contain business logic.  All domain work is
delegated to sub-agents.

Intent Classification Strategy
-------------------------------
Uses a lightweight LLM call with a structured prompt to classify the
user's intent into one of three categories.  This is more flexible than
keyword matching and handles paraphrased or ambiguous queries gracefully.
"""

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.memory import ConversationBufferWindowMemory
from agents.rag_agent import RAGAgent
from agents.operations_agent import OperationsAgent
import config


CLASSIFIER_PROMPT = """\
You are an intent classifier for NovaBite Restaurant's AI assistant.

Classify the user's message into EXACTLY one of these categories:

KNOWLEDGE - Questions about the menu, food items, ingredients, allergens,
  prices, opening hours, branch locations, policies, refund policy,
  loyalty program, events, catering, dress code, parking, dietary
  accommodations, or any other information that would be found in the
  restaurant's knowledge base.
  Examples: "Do you have vegan options?", "What are your hours?",
  "Tell me about the birthday package", "Is the chicken grilled?"

OPERATIONS - Requests that require calling a live system / tool:
  checking table availability, booking a table, getting today's special.
  Examples: "Is there a table free tonight?", "Book a table for 4",
  "What's today's special at the downtown branch?"

GENERAL - Greetings, small talk, thanks, goodbyes, or questions not
  related to NovaBite at all.
  Examples: "Hi!", "Thanks for your help", "What's the weather?",
  "Tell me a joke"

CONVERSATION HISTORY (for context):
{chat_history}

USER MESSAGE: {input}

Respond with ONLY the category name (KNOWLEDGE, OPERATIONS, or GENERAL).
Nothing else.
"""


GENERAL_RESPONSE_PROMPT = """\
You are NovaBite's friendly AI restaurant assistant.  The user's message
is general conversation (greeting, thanks, small talk, or off-topic).

Respond warmly and briefly.  If the message is off-topic, gently steer
the conversation back to how you can help with NovaBite (menu info,
reservations, specials, etc.).

Do NOT make up any restaurant information.  If they ask something
specific, tell them you can help with menu questions, table availability,
bookings, and today's specials.

CONVERSATION HISTORY:
{chat_history}

USER MESSAGE: {input}
"""


class Orchestrator:
    """
    Main orchestrator that classifies intent, routes to sub-agents,
    and manages conversation memory.
    """

    def __init__(self, rag_agent: RAGAgent, ops_agent: OperationsAgent):
        self.rag_agent = rag_agent
        self.ops_agent = ops_agent

        self.llm = ChatOpenAI(
            model=config.LLM_MODEL,
            temperature=config.LLM_TEMPERATURE,
            openai_api_key=config.OPENAI_API_KEY,
        )

        self.friendly_llm = ChatOpenAI(
            model=config.LLM_MODEL,
            temperature=config.CREATIVE_TEMPERATURE,
            openai_api_key=config.OPENAI_API_KEY,
        )

        # Shared conversation memory
        self.memory = ConversationBufferWindowMemory(
            k=config.MEMORY_WINDOW_K,
            memory_key="chat_history",
            return_messages=True,
            input_key="input",
            output_key="output",
        )

        # Intent classifier chain
        self.classifier_chain = (
            ChatPromptTemplate.from_template(CLASSIFIER_PROMPT)
            | self.llm
            | StrOutputParser()
        )

        # General response chain
        self.general_chain = (
            ChatPromptTemplate.from_template(GENERAL_RESPONSE_PROMPT)
            | self.friendly_llm
            | StrOutputParser()
        )

    def _get_chat_history_str(self) -> str:
        """Get formatted chat history string."""
        messages = self.memory.chat_memory.messages
        if not messages:
            return "No previous conversation."

        lines = []
        for msg in messages[-config.MEMORY_WINDOW_K * 2:]:
            role = "User" if msg.type == "human" else "Assistant"
            lines.append(f"{role}: {msg.content}")
        return "\n".join(lines)

    def _classify_intent(self, user_input: str) -> str:
        """Classify the user's intent."""
        chat_history = self._get_chat_history_str()
        result = self.classifier_chain.invoke({
            "input": user_input,
            "chat_history": chat_history,
        })
        # Clean up classification
        classification = result.strip().upper()
        if "KNOWLEDGE" in classification:
            return "KNOWLEDGE"
        elif "OPERATIONS" in classification:
            return "OPERATIONS"
        else:
            return "GENERAL"

    def _handle_general(self, user_input: str) -> str:
        """Handle general / small-talk messages."""
        chat_history = self._get_chat_history_str()
        return self.general_chain.invoke({
            "input": user_input,
            "chat_history": chat_history,
        })

    def chat(self, user_input: str) -> str:
        """
        Process a user message end-to-end.

        1. Classify intent
        2. Route to the appropriate sub-agent
        3. Save the exchange to memory
        4. Return the response

        Parameters
        ----------
        user_input : str
            The user's message.

        Returns
        -------
        str
            The assistant's response.
        """
        # Step 1: Classify
        intent = self._classify_intent(user_input)
        print(f"\n[Orchestrator] Intent classified as: {intent}")

        # Step 2: Route to sub-agent
        if intent == "KNOWLEDGE":
            print("[Orchestrator] Routing to RAG Agent ...")
            chat_history = self._get_chat_history_str()
            response = self.rag_agent.run(
                question=user_input,
                chat_history=chat_history,
            )

        elif intent == "OPERATIONS":
            print("[Orchestrator] Routing to Operations Agent ...")
            chat_messages = self.memory.chat_memory.messages
            response = self.ops_agent.run(
                query=user_input,
                chat_history=chat_messages,
            )

        else:  # GENERAL
            print("[Orchestrator] Handling as general conversation ...")
            response = self._handle_general(user_input)

        # Step 3: Save to memory
        self.memory.save_context(
            {"input": user_input},
            {"output": response},
        )

        return response
