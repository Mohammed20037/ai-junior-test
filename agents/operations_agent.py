"""
Operations Agent (Tool-Based / MCP-Style)
==========================================
Sub-agent that handles live operational queries by calling restaurant
tools.  Uses LangChain's tool-calling agent pattern.

The agent is given access to:
- check_table_availability(date, time, branch)
- book_table(name, date, time, branch, party_size)
- get_today_special(branch)

It decides which tool to call based on the user's request, extracts
the required parameters from the conversation, and formats the tool
response into a human-friendly answer.
"""

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from tools import TOOLS
import config

OPERATIONS_SYSTEM_PROMPT = """\
You are the NovaBite Operations Assistant – a sub-agent responsible for
handling live restaurant operations using the tools available to you.

YOUR CAPABILITIES (via tools):
1. check_table_availability – Check if tables are free at a branch.
2. book_table – Make a reservation for a guest.
3. get_today_special – Look up today's chef's special at a branch.

RULES:
1. ALWAYS use the appropriate tool to get real data.  NEVER guess or
   make up availability, booking references, or specials.
2. If the user hasn't specified required parameters (date, time,
   branch), ask them politely before calling the tool.
3. Branch names must be one of: 'downtown', 'midtown', 'brooklyn'.
   If the user says a full name like "Brooklyn Heights", map it to
   'brooklyn'.
4. Format dates as YYYY-MM-DD and times as HH:MM (24-hour).
5. After getting tool results, present them in a friendly,
   conversational tone.
6. If a tool returns an error, explain it clearly to the user and
   suggest alternatives.
7. For bookings, always confirm the details back to the user and
   provide the booking reference number.

CONVERSATION HISTORY:
{chat_history}
"""


class OperationsAgent:
    """Sub-agent that handles operational tasks via tool calling."""

    def __init__(self):
        self.llm = ChatOpenAI(
            model=config.LLM_MODEL,
            temperature=config.LLM_TEMPERATURE,
            openai_api_key=config.OPENAI_API_KEY,
        )

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", OPERATIONS_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        agent = create_openai_tools_agent(
            llm=self.llm,
            tools=TOOLS,
            prompt=self.prompt,
        )

        self.executor = AgentExecutor(
            agent=agent,
            tools=TOOLS,
            verbose=True,
            max_iterations=5,
            handle_parsing_errors=True,
            return_intermediate_steps=True,
        )

    def run(self, query: str, chat_history: list = None) -> str:
        """
        Handle an operational query.

        Parameters
        ----------
        query : str
            The user's request (e.g. "Book a table for 2 at downtown
            branch tomorrow at 7pm").
        chat_history : list
            List of message objects for context.

        Returns
        -------
        str
            The agent's response after tool execution.
        """
        result = self.executor.invoke({
            "input": query,
            "chat_history": chat_history or [],
        })
        return result["output"]
