import json
import os
from typing import List, TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from generate_response import generate_response, setup_vector_store
from utils import call_llm_contextualize, call_llm_extract_facts, call_llm_relevant_context
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Please set your GOOGLE_API_KEY environment variable.")
MAX_INTERACTIONS = 10


# --- 1. Define Interaction and Agent State --- #
class Interaction(TypedDict):
    query: str
    response: str
    summary_and_facts: str

class AgentState(TypedDict):
    current_input: str
    modified_query: str
    relevant_context: str
    current_response: str
    current_summary_facts: str
    short_term_window: Annotated[List[Interaction], "Latest 10 interactions"]
    long_term_memory: Annotated[List[str], "Summaries older than 10 interactions"]

# --- 2. Initialize LLM --- #
try:
    LLM = ChatGoogleGenerativeAI(temperature=0.0, model="gemini-2.5-flash", api_key=GOOGLE_API_KEY)
except Exception as e:
    print(f"Warning: Could not initialize LLM. Error: {e}")
    LLM = None

# --- 3. Node Definitions --- #
def node_judge(state: AgentState):
    relevant_context = call_llm_relevant_context(
        state['current_input'],
        state.get('short_term_window', []),
        state.get('long_term_memory', []),
        LLM
    )
    return {"relevant_context": relevant_context}  # plain string

def node_contextualize(state: AgentState):
    relevant_context = state.get('relevant_context', "")
    refined_query = call_llm_contextualize(
        state['current_input'],
        relevant_context,
        LLM
    )
    return {"modified_query": refined_query}

def node_generate(state: AgentState):
    query = state['modified_query']
    response = generate_response(query, vector_db)
    return {"current_response": response}

def node_analyze(state: AgentState):
    query = state['modified_query']
    response = state['current_response']
    analysis =call_llm_extract_facts(query, response, LLM)
    return {"current_summary_facts": analysis}

def node_manage_memory(state: AgentState):
    new_interaction: Interaction = {
        "query": state['modified_query'],
        "response": state['current_response'],
        "summary_and_facts": state['current_summary_facts']
    }

    short_term = state.get('short_term_window', [])
    long_term = state.get('long_term_memory', [])

    short_term.append(new_interaction)
    if len(short_term) > MAX_INTERACTIONS:
        oldest = short_term.pop(0)
        long_term.append(oldest['summary_and_facts'])
        print("📦 Archiving oldest interaction to long-term memory")

    return {
        "short_term_window": short_term,
        "long_term_memory": long_term
    }

# --- 4. Build Graph --- #
workflow = StateGraph(AgentState)
workflow.add_node("judge", node_judge)
workflow.add_node("contextualize", node_contextualize)
workflow.add_node("generate", node_generate)
workflow.add_node("analyze", node_analyze)
workflow.add_node("memory_manager", node_manage_memory)

workflow.set_entry_point("judge")
workflow.add_edge("judge", "contextualize")
workflow.add_edge("contextualize", "generate")
workflow.add_edge("generate", "analyze")
workflow.add_edge("analyze", "memory_manager")
workflow.add_edge("memory_manager", END)
app = workflow.compile()

# --- 5. Run Simulation --- #
if __name__ == "__main__":
    with open("demo.json", "r") as f:
        dataset = json.load(f)

    print("--- 📂 Initializing RAG ---")
    vector_db = setup_vector_store(dataset, reset_db=True)
    print("--- ✅ RAG Setup Complete ---")

    current_state = {"short_term_window": [], "long_term_memory": []}

    queries = [
        "Across all transcripts, which escalation triggers appear most frequently before a customer requests to speak with a manager?",
        "Which transcripts contain multiple failed callback commitments before the escalation occurred?",
        "In transcript ESC-010, what unresolved issue caused the customer to grow increasingly frustrated?",
        "List all transcripts that match the most common escalation trigger identified earlier.",
        "Compare the pattern of missed callbacks with the frustration drivers seen in ESC-010 to determine if they share a common escalation trajectory."
    ]

    results_dump = []

    for i, q in enumerate(queries):
        print(f"\n[{i+1}/{len(queries)}] USER: {q}")
        inputs = {
            "current_input": q,
            "short_term_window": current_state["short_term_window"],
            "long_term_memory": current_state["long_term_memory"]
        }
        result = app.invoke(inputs, config={"configurable": {"vector_store": vector_db}})

        current_state["short_term_window"] = result["short_term_window"]
        current_state["long_term_memory"] = result["long_term_memory"]

        print(f"   👉 Answer:\n{result['current_response']}")
        print(f"   📊 Memory: Short-Term[{len(current_state['short_term_window'])}/{MAX_INTERACTIONS}] | Long-Term[{len(current_state['long_term_memory'])}]")

        results_dump.append({
            "query_index": i + 1,
            "query": q,
            "modified_query": result.get("modified_query", ""),
            "response": result.get("current_response", ""),
            "relevant_context": result.get("relevant_context", "")
        })

    output_file = "result.json"
    with open(output_file, "w") as f:
        json.dump(results_dump, f, indent=4)

    print(f"\n📁 All interactions saved to: {output_file}")
