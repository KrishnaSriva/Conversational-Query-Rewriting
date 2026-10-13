# utils.py
from langchain_core.messages import HumanMessage

# ----------- LLM CALL HELPERS -----------

def call_llm_relevant_context(query, short_term_window, long_term_memory, llm_agent):
    prompt = f"""
You are a context-refinement agent.

Your job:
Given:
1. The **current user query**:
   {query}

2. The **short-term memory** (last 10 interactions, verbatim):
   {short_term_window}

3. The **long-term memory** (summary facts from earlier turns):
   {long_term_memory}

Produce a **single, concise, factual context block** that contains ONLY the information necessary to:
- understand the query fully,
- resolve references, pronouns, or unknown entities,
- restore missing constraints or facts,
- make the query a **fully independent, standalone question** with no need for conversation history.

STRICT RULES:
- Do NOT rewrite or modify the query.
- Do NOT output a modified query.
- Do NOT include irrelevant history, chit-chat, or unused details.
- Use ONLY information from memory that directly helps interpret the current query.
- If no memory is relevant, return an empty string.

OUTPUT:
Return ONLY the refined context as plain text. No labels, no prefixes, no explanations.
"""

    response = llm_agent.invoke([HumanMessage(content=prompt)])
    return response.content


def call_llm_contextualize(query, relevant_context, llm_agent):
    prompt = f"""
You are a query-contextualization agent.

Goal:
Rewrite the user's query so that it becomes a **fully standalone, self-contained question**, using ONLY the information provided in the relevant context.

Inputs:
- Original Query: {query}
- Relevant Context: {relevant_context}

Guidelines:
1. Preserve the user's original intent and meaning exactly.
2. Incorporate only the facts explicitly present in the relevant context.
3. Resolve all pronouns, references, missing details, and ambiguities using the context.
4. Do NOT invent, infer, or guess anything that is not stated in the context.
5. Do NOT add extra explanations or commentary.
6. Do NOT mention the process or the context in the output.
7. If the context is empty or irrelevant, return the original query unchanged.

Output:
Return ONLY the rewritten standalone query as a single sentence, with no prefix, no label, and no additional text.
"""

    response = llm_agent.invoke([HumanMessage(content=prompt)])
    return response.content


def call_llm_extract_facts(query, response_text, llm_agent):
    prompt = (
        "Extract factual statements from the following:\n\n"
        f"Query: {query}\n"
        f"Response: {response_text}\n\n"
        "Return only bullet-point facts."
    )

    response = llm_agent.invoke([HumanMessage(content=prompt)])
    return response.content
