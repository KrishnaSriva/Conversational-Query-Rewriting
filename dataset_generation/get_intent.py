import numpy as np
import os
import json
import logging
from typing import List, Dict, Any, Union # Added Union for flexible type hints
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
# LangChain / RAG Core
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser # Added JsonOutputParser for structured output reliability
from langchain_openai import ChatOpenAI
# from langchain_groq import ChatGroq # Only necessary if you plan to use GROQ
from pydantic import BaseModel, Field
load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')
llm = ChatOpenAI(model = 'gpt-4o-mini',
                 temperature=0.7)


with open('intent_definitions.json' , "r" , encoding="utf-8") as f:
    INTENT_DEFINITIONS = json.load(f)


class IntentPrediction(BaseModel):
    label: str = Field(..., description="The predicted Intent Category, exactly one of the provided Intent Categories.")

parser = PydanticOutputParser(pydantic_object=IntentPrediction)

def predict_tag(
    cand_ids: List[str],
    vector_db: Chroma
) ->str:
    """
    Predict the query type (one of LABELS) for each candidate id in cand_ids.

    Returns a dict mapping cand_id -> { "label":..., "confidence":..., "explanation":... }

    - cand_ids : list of record ids to predict for
    - vector_db : Chroma instance (must support get(ids=..., include=[...]) returning dict with keys 'ids','embeddings','metadatas','documents')
    - embeddings_model : embeddings object (used only as fallback if embeddings missing)
    - llm_model : optional ChatGroq instance; if None, uses global `llm` variable.

    The function asks the LLM for a JSON response:
      { "label": "<one of LABELS>", "confidence": 0.0-1.0, "explanation": "short text" }

    """

    results : str
    # fetch candidate records (bulk)
    db_record = vector_db.get(ids=cand_ids, include=["embeddings", "metadatas", "documents"])
    ids = db_record.get("ids", [])
    embeddings = db_record.get("embeddings", [])
    metadatas = db_record.get("metadatas", [])
    documents = db_record.get('documents' , [])
    if not ids:
        # logging.error("No records returned for candidate ids: %s", cand_ids)
        results = ''
        return results

    # index by id for convenience

    model_context = {}
    for i , rid in enumerate(ids):
        meta =metadatas[i] if i < len(metadatas) else {}

    for i , rid in enumerate(ids):
        meta =metadatas[i] if i < len(metadatas) else {}
        model_context[rid] = {
            "id": rid,
            "relevancy": meta.get("relevancy", None),
            "completeness": meta.get("completeness", None),
            "complexity": meta.get("complexity", None),
            "ambiguity": meta.get("ambiguity", ""),
            "correctness": meta.get("correctness", ""),
            "novelty": meta.get("novelty", ""),
            "documents": (documents[i] if i < len(documents) else None),
        }
    
        
   
    prompt = ChatPromptTemplate.from_template("""
            You are a precise classification engine. Your task is to map a set of query dependency metrics/metadata to a specific Intent Label.

            ### INPUT DATA:
            You have to predict the label of the follow up query and you don't know what the query is .
            what you are given is the parent query on the basis of which the follow up query has to be generated 
            your task right now is to use the metrics , the context and your intuit you predict the label of the upcoming query without knowing the query
            ### STRICT RULES:
            1. **Output Format:** You must return ONLY a valid JSON object. Do not include markdown formatting (like ```json), distinct preambles, or postscripts.
            2. **Schema:**
             "label": a single string label that should not be out of the {labels}
            3. **Whitelisting:** You are strictly forbidden from inventing new labels. You must choose from the 'INTENT_DEFINITIONS' list provided below.
            4. **Hallucination Prevention:**
            - use only the given definitions to find the intent . Do not use your own reasoning to or remove any labels 
            -look at the metrics and context of the parent queries provided to predict the intent .

            ### INTENT DEFINITIONS:
            Use these definitions as your ground truth.
            {INTENT_DEFINITIONS}
             Analyze the following input metrics and categorize the intent.
               Given Model context :
                {model_context}
                Determine the label and explanation.
            return only a json with the given schema and nothing lese  , note the label must be only the key from the label definition
            output format must be string only"""

    )
    # llm_model = llm_model.with_structured_output(output)
    chain = (  
        prompt
        | llm|parser
    )
    
    result = chain.invoke({
        'INTENT_DEFINITIONS' : INTENT_DEFINITIONS ,
        'labels' : INTENT_DEFINITIONS.keys(),
        'model_context':model_context 
        
    })

   

    return result.label


if __name__ == '__main__':
    from langchain_community.embeddings import FakeEmbeddings
    
    embeddings = FakeEmbeddings(size=768)

    # Initialize Chroma (In-memory for testing)
    vector_db = Chroma(
        collection_name="manager_chat_history", 
        embedding_function=embeddings
    )

    # 3. Add Context (The Parent Interaction)
    # This represents the conversation history before the new query
    parent_id = "interaction_101"
    transcript_context = "A billing dispute call where the customer was overcharged by $50."
    vector_db.add_texts(
    texts=[
        "Manager: Did the agent verify the hardware connections before scheduling the technician?\n\n"
        "AI: No, the verification process was incomplete. While the agent did instruct the customer to perform a standard power cycle (router restart) at the 3-minute mark, they failed to request a physical inspection of the coaxial or ethernet cables to ensure they were securely plugged in. Consequently, the agent escalated the issue to a Tier 2 technician visit without ruling out simple loose cabling, which is a violation of the standard Step 1 troubleshooting protocol."
    ],
    metadatas=[{
        "id": parent_id,
        "role": "history",
        "relevancy": 0.95,     # The AI answered the specific process question directly
        "completeness": 0.9,  # Provided specific details on what was missed
        "complexity": 0.4,     # Standard procedure, low complexity
        "ambiguity": 0.1,    # The failure to check cables is clearly stated
        "correctness": 0.95,
        "timestamp": "2023-11-15 14:30:00"
    }],
    ids=[parent_id]
)

    try:
    # We pass the target_id as the candidate to be classified
        prediction = predict_tag(
            cand_ids=[parent_id], 
            vector_db=vector_db 
        )
        
        print("\nResult:")
        print(prediction)

    except Exception as e:
        print(f"Error occurred: {e}")