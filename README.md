# 🍽️ NovaBite Multi-Agent RAG System – Smart Restaurant Assistant

A production-style multi-agent AI system for **NovaBite Restaurants**, built with LangChain. The system uses **RAG** for grounded knowledge retrieval, **tool-calling agents** for live operations, **sub-agent delegation** for clean separation of concerns, and **conversation memory** for multi-turn continuity.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [RAG Design Decisions](#rag-design-decisions)
3. [Tool Simulation (MCP-Style)](#tool-simulation-mcp-style)
4. [Memory Design](#memory-design)
5. [Hallucination Prevention](#hallucination-prevention)
6. [Setup & Installation](#setup--installation)
7. [Usage](#usage)
8. [Example Queries & Outputs](#example-queries--outputs)
9. [Project Structure](#project-structure)
10. [Assumptions](#assumptions)

---

## Architecture Overview

```
                         ┌───────────────┐
                         │     User      │
                         └──────┬────────┘
                                │
                                ▼
                 ┌──────────────────────────────┐
                 │     Main Orchestrator Agent   │
                 │                                │
                 │  • Intent Classification       │
                 │  • Request Routing              │
                 │  • Conversation Memory          │
                 │  • Response Merging             │
                 │  • Ambiguity Handling            │
                 │                                │
                 │  ⚠️ Contains NO business logic  │
                 └────┬──────────┬──────────┬─────┘
                      │          │          │
            ┌─────────┘    ┌─────┘    ┌─────┘
            ▼              ▼          ▼
   ┌────────────────┐ ┌─────────┐ ┌──────────────────┐
   │  RAG Agent     │ │ General │ │ Operations Agent  │
   │                │ │ Handler │ │                    │
   │ • Retrieval    │ │         │ │ • Tool Calling     │
   │ • Grounding    │ │ • Small │ │ • check_table_     │
   │ • Menu Info    │ │   talk  │ │   availability     │
   │ • Policies     │ │ • Greet │ │ • book_table       │
   │ • Hours        │ │ • Bye   │ │ • get_today_       │
   │ • Events       │ │         │ │   special          │
   └───────┬────────┘ └─────────┘ └────────┬───────────┘
           │                                │
           ▼                                ▼
  ┌─────────────────┐             ┌─────────────────┐
  │  FAISS Vector   │             │ Simulated MCP   │
  │  Store          │             │ Server / Tools  │
  │  (Knowledge     │             │ (Restaurant     │
  │   Base)         │             │  Backend)       │
  └─────────────────┘             └─────────────────┘
```

### Flow

1. **User sends a message** → Orchestrator receives it.
2. **Intent Classification** – A lightweight LLM call classifies the intent as `KNOWLEDGE`, `OPERATIONS`, or `GENERAL`.
3. **Routing** – The orchestrator delegates to the appropriate sub-agent:
   - `KNOWLEDGE` → RAG Agent queries the vector store and generates a grounded answer.
   - `OPERATIONS` → Operations Agent selects and calls the appropriate tool, then formats the result.
   - `GENERAL` → A simple friendly-response chain handles greetings/small talk.
4. **Memory** – The exchange (input + output) is saved to `ConversationBufferWindowMemory`, available to all agents on the next turn.
5. **Response** – The orchestrator returns the sub-agent's response to the user.

### Why This Architecture?

- **Separation of concerns**: The orchestrator knows *how to route* but not *how to answer*. Business logic lives in sub-agents.
- **Testability**: Each sub-agent can be unit-tested independently.
- **Extensibility**: Adding a new capability (e.g., a "Delivery Agent") means creating a new sub-agent and adding one routing case — no changes to existing agents.
- **Hallucination control**: The RAG agent has strict grounding rules. The Operations agent only returns real tool outputs. Neither can "make things up."

---

## RAG Design Decisions

### Document Domains (2 of 7 chosen)

1. **Menu** – Full menu with prices, descriptions, allergens, dietary tags, and calories.
2. **Branch Policies & Information** – Opening hours, reservation policy, refund policy, loyalty program, event hosting/catering packages, dietary accommodations, dress code, and branch details.

### Chunking Strategy

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Method | `RecursiveCharacterTextSplitter` | Tries `\n\n` → `\n` → `. ` → ` ` in order, preserving section boundaries before falling back to sentence/word splits. Better than fixed-size for structured documents. |
| Chunk Size | 500 chars | Most menu items and policy sections are 100–300 chars. 500 comfortably holds one complete item with all metadata (price, allergens, description) without mixing unrelated items. |
| Overlap | 50 chars | Small overlap preserves context at boundaries (e.g., a heading that starts a new section) without excessive duplication. |

### Embedding Model

**`text-embedding-3-small`** (OpenAI)
- 1536-dimensional vectors
- Strong semantic accuracy for English text
- Cost-effective ($0.02 / 1M tokens) — appropriate for a knowledge base of ~50 chunks
- Chosen over `text-embedding-3-large` (3072-dim) because the marginal quality gain is unnecessary for a small, well-structured corpus

### Vector Database

**FAISS** (Facebook AI Similarity Search)
- Runs in-process — no server, no Docker, no external dependency
- Persists to disk with `save_local()` / `load_local()`, so embeddings are computed only once
- Supports exact L2/IP similarity — for <1000 vectors, approximate methods (HNSW, IVF) offer no benefit
- Chosen over Chroma because it has fewer dependencies and simpler persistence for small-scale use

### Retrieval Strategy

- **Method**: Similarity search (cosine via FAISS inner product on normalized vectors)
- **Top-K**: 4 — retrieves the 4 most relevant chunks per query
- **Why K=4**: Provides enough context to answer most questions (a menu item + its neighbors), while keeping the prompt lean to reduce noise and hallucination risk. Tested with K=2 (missed cross-section answers) and K=6 (introduced irrelevant chunks).

### Context Filtering & Grounding

The RAG agent's system prompt enforces strict rules:
1. **Only** use the retrieved context — never generate from parametric memory.
2. If context is insufficient, explicitly say "I don't have information about that."
3. Quote prices, allergens, and hours directly from the context.
4. If asked about a non-existent menu item, clearly state it's not on the menu.

---

## Tool Simulation (MCP-Style)

### Approach

The Operations Agent uses **LangChain tool-calling** (`create_openai_tools_agent`) with three tools decorated with `@tool`. Each tool simulates what a real MCP server or backend API would return:

- **Realistic inputs**: Typed parameters with validation (date format, branch name).
- **Realistic outputs**: JSON responses with `status`, structured data, and error messages.
- **Deterministic simulation**: Availability uses a seeded random generator based on `hash(branch + date + time)`, so the same query returns the same result within a session — mimicking a real database.

### Implemented Tools (3)

| Tool | Parameters | Behavior |
|------|------------|----------|
| `check_table_availability` | `date`, `time`, `branch` | Returns total/available tables. Busier on weekends and evenings (realistic simulation). |
| `book_table` | `name`, `date`, `time`, `branch`, `party_size` | Checks availability first, then creates a booking with a reference number. Returns an error if fully booked. |
| `get_today_special` | `branch` | Returns a different chef's special per branch per day-of-week (7 unique specials × 3 branches = 21 total). |

### Why Simulated vs. Real MCP

A simulated approach was chosen because:
1. The assessment can be evaluated without running an external server.
2. The tool interface (typed parameters, JSON responses, error handling) is **identical** to what a real MCP integration would use.
3. Swapping `_generate_availability()` for an HTTP call to a real MCP server requires changing only the function body — the agent, routing, and memory layers are untouched.

---

## Memory Design

### Implementation

**`ConversationBufferWindowMemory`** with `K = 10` (last 10 exchanges).

### Why This Approach

| Alternative | Why Not |
|-------------|---------|
| `ConversationBufferMemory` (unlimited) | Unbounded growth risks hitting the LLM's context window on long conversations. |
| `ConversationSummaryMemory` | Summaries lose specific details (table numbers, menu item names, booking references) that are critical for restaurant follow-ups. |
| `ConversationEntityMemory` | Over-engineered — entity tracking adds latency without clear benefit for short, task-oriented restaurant interactions. |

### How Memory Flows

1. **Shared memory**: The orchestrator owns a single `ConversationBufferWindowMemory` instance.
2. **Read by sub-agents**: Both the RAG agent and Operations agent receive conversation history as input, so they can resolve follow-up references ("Which of those is gluten-free?" requires knowing what "those" refers to).
3. **Written by orchestrator**: After each exchange, the orchestrator saves `{input, output}` to memory.
4. **Resettable**: The user can type `clear` to reset memory (useful for demos).

### Memory Continuity Example

```
User: "What vegan options do you have?"
→ RAG Agent lists: Penne Arrabbiata, Thai Green Curry, Vegan Pad Thai, etc.

User: "Which of those is gluten-free?"
→ Memory injects the previous exchange, so RAG Agent knows "those"
  = the vegan options listed above. Answers: Thai Green Curry and
  Vegan Pad Thai are gluten-free.

User: "How much does the Thai Green Curry cost?"
→ Memory carries forward. RAG Agent answers: $16.49.
```

---

## Hallucination Prevention

Hallucination prevention is enforced at **multiple layers**:

| Layer | Mechanism |
|-------|-----------|
| **RAG Agent Prompt** | Strict instruction: "ONLY use the provided context." If context is insufficient, return a scripted fallback — never invent information. |
| **LLM Temperature** | Set to `0.0` for all factual chains (classification, RAG, tool-calling). Only general conversation uses `0.3`. |
| **Knowledge Base Design** | The menu document explicitly lists what NovaBite does NOT serve (e.g., "NovaBite does NOT serve sushi, tacos, or pizza"). This gives the RAG agent explicit negative evidence. |
| **Operations Agent** | Only returns data from tool calls. Cannot generate availability numbers or booking references on its own. |
| **Orchestrator** | Does not answer questions directly — always delegates. No path exists for the orchestrator to hallucinate domain content. |

---

## Setup & Installation

### Prerequisites

- Python 3.10+
- An OpenAI API key

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/ai-junior-test.git
cd ai-junior-test

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # Linux/Mac
# venv\Scripts\activate       # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your API key
cp .env.example .env
# Edit .env and add your OpenAI API key

# 5. Run the system
python main.py
```

### First Run

On the first run, the system will:
1. Read the knowledge base files from `data/knowledge_base/`.
2. Chunk and embed them using OpenAI's embedding API.
3. Save the FAISS index to `data/vector_store/` (cached for future runs).

Subsequent runs load the cached index instantly. Use `python main.py --rebuild` to force re-ingestion.

---

## Usage

### Interactive Chat

```bash
python main.py
```

Type messages naturally. The system automatically classifies intent and routes to the correct agent.

### Run Demo Scenarios

```bash
python examples/example_queries.py
```

This runs predefined queries covering all categories: RAG, operations, memory continuity, and hallucination tests.

### Commands

| Command | Action |
|---------|--------|
| `quit` / `exit` | End the session |
| `clear` | Reset conversation memory |

---

## Example Queries & Outputs

### RAG – Menu

```
👤 You: Do you have vegan pasta?
🧠 [Orchestrator] Intent classified as: KNOWLEDGE
📚 [Orchestrator] Routing to RAG Agent …

🤖 NovaBite: NovaBite doesn't have an item specifically called "vegan pasta,"
   but we do have vegan pasta dishes! The **Penne Arrabbiata** ($14.99) is fully
   vegan — it's penne pasta in a spicy tomato sauce with garlic, chili flakes,
   and fresh basil (contains gluten). We also have the **Vegan Pad Thai** ($14.49)
   made with rice noodles, tofu, and tamarind sauce (gluten-free, but contains
   peanuts and soy). Additionally, any pasta dish can be made with gluten-free
   pasta for an extra $2.00.
```

### RAG – Policies

```
👤 You: What are your opening hours on weekends?
🧠 [Orchestrator] Intent classified as: KNOWLEDGE

🤖 NovaBite: On weekends (Saturday & Sunday):
   • Downtown & Midtown branches: 10:00 AM – 11:30 PM
   • Brooklyn Heights: 9:00 AM – 11:00 PM (brunch served 9:00 AM – 1:00 PM)
   The kitchen closes 30 minutes before closing time at all branches.
```

### RAG – Events

```
👤 You: What's included in the premium catering package?
🧠 [Orchestrator] Intent classified as: KNOWLEDGE

🤖 NovaBite: The Premium Catering Package is $65 per person (minimum 20 guests)
   and includes: Chef's tasting menu (5 courses), premium wine pairing
   (2 glasses per guest), dedicated wait staff, custom table setup and floral
   arrangement, and a complimentary dessert bar. It must be booked at least
   14 days in advance and is available at the Midtown and Downtown branches only.
```

### Operations – Table Availability

```
👤 You: Is there a table available at downtown tonight at 7pm?
🧠 [Orchestrator] Intent classified as: OPERATIONS
⚙️  [Orchestrator] Routing to Operations Agent …

🤖 NovaBite: I checked the Downtown branch for tonight at 7:00 PM. There are
   currently 3 tables available out of 16 total. It's a busy evening, so I'd
   recommend booking soon! Would you like me to reserve a table for you?
```

### Operations – Booking

```
👤 You: Book a table for 4 at Midtown on 2025-07-15 at 19:00 under John
🧠 [Orchestrator] Intent classified as: OPERATIONS

🤖 NovaBite: Your table has been booked! Here are the details:
   • Name: John
   • Branch: Midtown
   • Date: July 15, 2025
   • Time: 7:00 PM
   • Party Size: 4
   • Booking Reference: NB-47283
   Please remember that cancellations must be made at least 2 hours before
   your reserved time.
```

### Operations – Today's Special

```
👤 You: What's today's special at the Brooklyn branch?
🧠 [Orchestrator] Intent classified as: OPERATIONS

🤖 NovaBite: Today's special at the Brooklyn branch is the **Shrimp & Grits**
   ($20.99) — Gulf shrimp over creamy stone-ground grits with andouille sausage.
```

### Hallucination Prevention

```
👤 You: Do you serve sushi?
🧠 [Orchestrator] Intent classified as: KNOWLEDGE

🤖 NovaBite: No, NovaBite does not serve sushi. Our menu focuses on Western and
   Mediterranean cuisine. However, if you're in the mood for seafood, we have
   some great options like the Grilled Atlantic Salmon ($24.99), Lobster
   Linguine ($32.99), and the Spicy Tuna Tartare appetizer ($14.99).
```

### Memory Continuity

```
👤 You: What vegan options do you have?
🤖 NovaBite: We have several vegan options: Penne Arrabbiata ($14.99),
   Thai Green Curry ($16.49), Vegan Pad Thai ($14.49), Mediterranean
   Hummus Platter ($9.99), and Vegan Mango Sorbet ($7.99).

👤 You: Which of those is gluten-free?
🤖 NovaBite: Of the vegan options I mentioned, the Thai Green Curry ($16.49)
   and the Vegan Mango Sorbet ($7.99) are both gluten-free. The Vegan Pad
   Thai is also gluten-free but contains peanuts and soy.

👤 You: Great, how much is the curry?
🤖 NovaBite: The Thai Green Curry is $16.49.
```

---

## Project Structure

```
novabite-ai-assistant/
├── main.py                        # Entry point – interactive CLI
├── config.py                      # Centralized configuration
├── requirements.txt               # Python dependencies
├── .env.example                   # Environment variable template
│
├── agents/
│   ├── __init__.py
│   ├── orchestrator.py            # Main orchestrator (intent + routing + memory)
│   ├── rag_agent.py               # Knowledge RAG sub-agent
│   └── operations_agent.py        # Tool-calling operations sub-agent
│
├── rag/
│   ├── __init__.py
│   ├── ingestion.py               # Document loading, chunking, embedding
│   └── retriever.py               # FAISS retriever wrapper
│
├── tools/
│   ├── __init__.py
│   └── restaurant_tools.py        # MCP-style simulated tools
│
├── memory/
│   └── __init__.py                # Conversation memory factory
│
├── data/
│   ├── knowledge_base/
│   │   ├── menu.txt               # Full restaurant menu
│   │   └── policies.txt           # Branch info, hours, policies, events
│   └── vector_store/              # Persisted FAISS index (auto-generated)
│
└── examples/
    └── example_queries.py         # Demo script with test scenarios
```

---

## Assumptions

1. **LLM Provider**: OpenAI GPT-4o-mini is used. The system is designed so swapping to another LangChain-compatible LLM (Anthropic, Ollama, etc.) requires changing only `config.py` and the import in agent files.

2. **Single-user**: The system runs as a single-user CLI. In production, the memory and orchestrator would be instantiated per-session (e.g., per WebSocket connection).

3. **Tool simulation**: Tools simulate backend responses with deterministic randomization. In production, these would be HTTP calls to real MCP servers or internal APIs. The tool interfaces are designed to be swap-ready.

4. **Knowledge base scope**: Two domains were implemented (Menu + Policies/Hours) out of the seven listed. Adding more domains requires only dropping new `.txt` files into `data/knowledge_base/` and running with `--rebuild`.

5. **Embedding cost**: On first run, embedding ~50 chunks costs approximately $0.001. The FAISS index is cached to disk, so subsequent runs incur zero embedding cost.

6. **Date handling in tools**: The `book_table` and `check_table_availability` tools accept future dates without validation against opening hours. A production system would cross-reference holiday closures and operating hours.

7. **No authentication**: No user authentication is implemented. The `check_loyalty_points` tool was not implemented because it would require a user identity system. The three implemented tools (`check_table_availability`, `book_table`, `get_today_special`) cover the required minimum of two.

---

## License

Built as a technical assessment for the AI Engineer position at Fekra.
