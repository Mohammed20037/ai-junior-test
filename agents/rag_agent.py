"""
Restaurant Knowledge RAG Agent
================================
Sub-agent responsible for answering questions grounded in the
NovaBite knowledge base (menu, policies, hours, events).

Hallucination Prevention Strategy
----------------------------------
1. System prompt explicitly instructs the LLM to ONLY use provided
   context and to say "I don't have information about that" when
   the context is insufficient.
2. Retrieved chunks are injected verbatim so the model can cite them.
3. Low temperature (0.0) eliminates creative generation.
4. The prompt asks the model to quote the source when possible.
"""

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema import StrOutputParser
import config

RAG_SYSTEM_PROMPT = """\
You are the NovaBite Restaurant Knowledge Assistant – a sub-agent in a
multi-agent system.  Your ONLY job is to answer questions using the
context retrieved from NovaBite's internal knowledge base.

STRICT RULES:
1. ONLY use the provided context to answer.  Do NOT invent menu items,
   prices, policies, or any other information.
2. If the context does not contain enough information to answer the
   question, respond EXACTLY with:
   "I don't have specific information about that in our knowledge base.
   You may want to contact us directly at info@novabite.com or call
   (212) 555-0000."
3. When stating facts (prices, allergens, hours), quote them precisely
   from the context.
4. Be friendly, concise, and helpful.
5. If a customer asks about a menu item that does NOT exist in the
   context, clearly state that it is not on the menu.  Never make up
   a dish.
6. Always mention relevant allergen information when discussing food
   items.

CONTEXT FROM KNOWLEDGE BASE:
{context}

CONVERSATION HISTORY:
{chat_history}
"""

RAG_HUMAN_PROMPT = "{question}"


class RAGAgent:
    """Sub-agent that answers knowledge-base questions via RAG."""

    def __init__(self, retriever):
        self.retriever = retriever
        self.llm = ChatOpenAI(
            model=config.LLM_MODEL,
            temperature=config.LLM_TEMPERATURE,
            openai_api_key=config.OPENAI_API_KEY,
        )

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", RAG_SYSTEM_PROMPT),
            ("human", RAG_HUMAN_PROMPT),
        ])

        # Build the RAG chain
        self.chain = (
            {
                "context": self._retrieve_and_format,
                "question": RunnablePassthrough(),
                "chat_history": lambda x: x.get("chat_history", ""),
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def _retrieve_and_format(self, inputs) -> str:
        """Retrieve relevant chunks and format them as context."""
        if isinstance(inputs, dict):
            query = inputs.get("question", inputs.get("input", ""))
        else:
            query = str(inputs)

        docs = self.retriever.invoke(query)

        if not docs:
            return "No relevant information found in the knowledge base."

        formatted = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source_file", "unknown")
            formatted.append(f"[Source: {source} | Chunk {i}]\n{doc.page_content}")

        return "\n\n---\n\n".join(formatted)

    def run(self, question: str, chat_history: str = "") -> str:
        """
        Answer a knowledge-base question.

        Parameters
        ----------
        question : str
            The user's question.
        chat_history : str
            Formatted conversation history for context.

        Returns
        -------
        str
            The grounded answer.
        """
        result = self.chain.invoke({
            "question": question,
            "chat_history": chat_history,
        })
        return result
