import os
from dotenv import load_dotenv
from typing import List, Dict, Optional, Any
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
llm = ChatOpenAI(model = 'gpt-4o-mini',
                 temperature=0.6)

INTENT_DEFINITIONS = {
    "Clarifying Query": "The user explicitly seeks disambiguation of a specific term, phrase, or concept in the previous response, because the meaning is unclear or incomplete.",
    "Exploring Domain": "The user requests additional information beyond what was provided, expanding the scope of the conversation into related but previously unmentioned aspects.",
    "Verifying Info": "The user is checking the factual correctness or accuracy of the model’s previous answer.",
    "Queries for Agent": "The question specifically evaluates, inspects, critiques, or extracts insights about the customer-care agent’s behavior, performance, decisions, or compliance.",
    "Queries for Customer": "The user aims to analyze, interpret, or extract insights about the customer’s intentions, goals, behavior, or emotional state.",
    "Excluding Condition": "The user requests a result minus a specific condition previously used in the search.",
    "Adding/Specifying Condition": "The user adds a new constraint, filter, or condition on top of the existing search or retrieval request.",
    "Substituting Condition": "The user replaces one search/filter condition with a different one, altering the scope without adding or removing multiple constraints.",
    "Converting Format": "The user requests the same information in a different output format (tables, JSON, etc.) without questioning correctness.",
    "Criticizing Response": "The user explicitly states or implies that the previous answer is missing information, incorrect, incomplete, or flawed.",
    "Confirming Response": "The user seeks to validate their understanding of the previous answer.",
    "Requesting Opinion": "User asks for subjective judgment or evaluation, not factual retrieval.",
    "Requesting Unrelated Info": "The query does not depend on conversation context and requests information outside the customer-service domain."
}

# Model instance (kept as you had it)



def generate_modified_query(
    vector_db: Any,
    query: str,
    context: Optional[str],
    metadata: Dict[str, Any],
) -> str:
    """
    Build a standalone follow-up query that strictly matches the provided intent.
    Returns: single follow-up query string (no quotes, no explanation).
    """
    cand_ids: List[str] = metadata.get("context_point", []) or []
    model_context: Dict[str, Dict[str, Optional[str]]] = {}

    if cand_ids:
        try:
            db_record = vector_db.get(ids=cand_ids, include=["embeddings", "metadatas", "documents"])
        except Exception:
            db_record = {}
    else:
        db_record = {}

    ids = db_record.get("ids", []) or []
    documents = db_record.get("documents", []) or []

    for i, rid in enumerate(ids):
        model_context[rid] = {
            "id": rid,
            "document": documents[i] if i < len(documents) else [],
        }

   

    intent_name = metadata.get("intent") or metadata.get("intent_name") or "Exploring Domain"
    intent_definition = INTENT_DEFINITIONS.get(intent_name, "User requests information related to the previous interaction.")

    prompt_template = ChatPromptTemplate.from_template(
    """
        You are a Manager at a company analyzing customer service call transcripts.
        You are currently conversing with an AI assistant that has access to these transcripts.

        Previous Context ( the context on which the followup query we are generating is dependent): 
        {parent_context}

        User's current query or utterance which has to be modified:
        \"\"\"{query}\"\"\"

        ### Task
        Produce a single, standalone follow-up query that STRICTLY matches the specified intent below.
        - The output MUST be only the follow-up query string (no quotes, no explanations, no lists).
        - The follow-up query should be sufficiently specific and challenging for retrieval/QA systems.
        - The follow up query must carry enough information from the context to be able to get evaluationed by our system
        - Do not mention the intent name, internal notes, or the source documents in the output.
        Return only the follow-up query text.
    """)

    chain = (prompt_template | llm | StrOutputParser())

    prompt_inputs = {
        "query" :  query,
        "parent_context": model_context,
    }

    result = chain.invoke(prompt_inputs)

  
    return result

if __name__ == '__main__':
    metadata = {
    "intent": "Clarifying Query",
    "context_point": ["doc_44A", "doc_44B"]
    }
    class MockVectorDB:
        def get(self, ids, include):
            return {
                "ids": ["doc_A", "doc_B"],
                "documents": [
                    """
                        Generally how the escalations happen during a call 
                    """ , 
                        """**How escalations are handled during a call – what the transcripts show** | Step | What usually triggers the escalation | What the agent does to escalate | Example from the transcripts | |------|--------------------------------------|--------------------------------|------------------------------| | **1. Customer signals that the normal resolution path has failed** | • A promised callback never happened (ESC‑011, ESC‑010, ESC‑012) <br>• Repeated transfers without a solution (ESC‑005, ESC‑006, ESC‑004) <br>• Long hold times that the customer refuses to accept (ESC‑001, ESC‑003, ESC‑002) <br>• Multiple unanswered promises or “empty” scripts (ESC‑014, ESC‑015, ESC‑013) | The agent acknowledges the frustration, repeats the last offer (usually a callback or a new ticket) and asks the customer if that is acceptable. If the customer says “no” or explicitly asks for a manager, the agent moves to the next step. | **ESC‑011** – Customer says the 10:30 AM callback never came; the agent says a supervisor will call within 60 minutes, then the customer says “I must escalate to a manager.” | | **2. Agent flags the case as high‑priority / urgent** | The conversation has already involved missed promises or multiple hand‑offs. | The agent adds a note to the ticket (e.g., “high priority”, “urgent escalation”) and often mentions a specific time window for a supervisor callback. | **ESC‑011** – Agent “flagged the ticket as high priority” and arranged a supervisor follow‑up within 60 minutes. | | **3. Creation of an escalation ticket or internal request** | When the issue cannot be solved on the call (e.g., needs field tech, billing approval, engineering). | Agent opens an internal escalation ticket, notes the required response time, and tells the customer the expected turnaround (e.g., “within 2 hours”, “72 hours”, “7‑10 business days”). | **ESC‑012** – Agent “will escalate to the billing supervisor and request a callback within two hours.” <br>**ESC‑015** – Agent “will raise another internal escalation request… they’ll get in touch within 72 hours.” | | **4. Transfer to a supervisor/manager** | Customer explicitly asks for a supervisor, or the agent determines the issue is beyond their authority. | The agent informs the customer they will be transferred, sometimes asking the customer to hold briefly, but often honoring the request to avoid another hold. | **ESC‑001** – Customer says “connect me to your manager now”; agent says “I can escalate this for you.” <br>**ESC‑003** – Customer says “I want to speak to a supervisor right now”; agent says “I will connect you to a supervisor; please hold briefly while I transfer.” | | **5. Confirmation of contact details** | Before the supervisor or callback is scheduled, the agent confirms the best phone number or email so the follow‑up reaches the right person. | “May I confirm the best number?” and the customer repeats the number ending in … | **ESC‑011** – Agent asks for the best number; customer confirms “the number ending in 4321.” <br>**ESC‑012** – Agent confirms “the number ending in 9087.” | | **6. Closing the call (or handing off)** | Once the escalation is set, the agent ends the call, often apologizing again and stating the next steps. | “You should expect a callback within X minutes/hours,” or “I’ve logged a ticket; a supervisor will call you.” | **ESC‑010** – Agent says “You should expect a callback within two hours.” <br>**ESC‑019** – Agent says “I’ll schedule a field technician; the earliest slot is in five days.” (customer still escalates, but the pattern is the same). | ### Common patterns across the transcripts 1. **Missed or unreliable callbacks** – When a promised callback does not occur, the customer immediately asks for a higher‑level escalation (ESC‑011, ESC‑010, ESC‑012). 2. **Multiple transfers without resolution** – After being bounced between departments, the customer demands a supervisor (ESC‑005, ESC‑006, ESC‑004). 3. **Long hold times** – If the hold time is deemed unreasonable, the customer refuses to wait and asks for a manager (ESC‑001, ESC‑003, ESC‑002). 4. **Repeated “same‑script” promises** – When the agent repeats a solution that has already failed (e.g., “we’ll call you in 2 hours” or “3‑5 days”), the customer escalates (ESC‑014, ESC‑015, ESC‑013). 5. **Escalation time‑frames** – Agents usually give a concrete window for the next step (60 minutes, 2 hours, 72 hours, 7‑10 days). This is meant to give the customer a measurable expectation. 6. **Supervisor/manager hand‑off** – The final step is a direct transfer to a supervisor or a promise that a supervisor will call back. The agent often tries to keep the hold brief, but many customers explicitly say “don’t put me on hold again.” ### In short - **Escalation is triggered by the customer’s loss of confidence** (missed callbacks, endless transfers, long holds, repeated ineffective promises). - **The agent first tries to offer a concrete next step** (new callback, urgent ticket, higher‑priority flag). - **If the customer refuses or explicitly asks for a manager, the agent creates an urgent escalation ticket and transfers the call** (or promises a supervisor callback within a stated time). - **Contact details are reconfirmed**, and the agent ends the call with an apology and the agreed‑upon follow‑up timeline. All of the above steps are reflected in the provided call transcripts; no other escalation procedures are mentioned in the data. -------------------------------------------------- generate a follow up query for this query and response based on what is missing or ambiguous in the context , the query may also ask an exploring domain query for generation of content that is not actually present in the query
                """
                ],
                  
            }
    vector_db = MockVectorDB()
    output = generate_modified_query(
    vector_db=vector_db,
    query="Is there anything in these escalation steps that could work differently depending on the company’s internal process?",
    context=None,
    metadata=metadata,
)

    print(output)
#output-> :

#Which of the six escalation steps outlined (customer trigger, high‑priority flag, escalation ticket creation, supervisor transfer, contact‑detail confirmation, and call closing) are most likely to differ across companies, and what alternative practices do organizations commonly use for those steps?
