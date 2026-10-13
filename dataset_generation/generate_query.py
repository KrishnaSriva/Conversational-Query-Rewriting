import os
from dotenv import load_dotenv
from typing import List, Dict, Optional
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import json
from langchain_openai import ChatOpenAI

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
llm = ChatOpenAI(model = 'gpt-4o-mini',
                 temperature=0.5)

with open('intent_definitions.json' , "r" , encoding="utf-8") as f:
    INTENT_DEFINITIONS = json.load(f)

def get_parent_interactions(dependency_ids: List[str], vector_db: Chroma) -> str:
    """
    Retrieves the parent interaction(s) from ChromaDB based on IDs 
    and formats them as a conversation string.
    """
    if not dependency_ids:
        return "No previous interaction context."

    formatted_context = ""
    
    results = vector_db.get(ids=dependency_ids, include=["metadatas", "documents"])
    
    for i, doc_id in enumerate(results['ids']):
        content = results['documents'][i] 
        
        # If you stored specific fields in metadata, you can reconstruct nicely:
        # meta = results['metadatas'][i]
        # query = meta.get('query_text', '')
        # response = meta.get('response_text', '')
        
        formatted_context += f"--- Interaction ID: {doc_id} ---\n{content}\n\n"

    return formatted_context


def generate_followup_query(
    dependency_ids: List[str], 
    target_intent_name: str, 
    transcript_summary: str,
    vector_db: Chroma
) -> str:
    """
    Generates a follow-up query based on retrieved context and a specific intent.
    
    Args:
        dependency_ids: List of interaction IDs that this new query depends on.
        target_intent_name: Key from INTENT_DEFINITIONS.
        transcript_summary: Context of the call transcript being analyzed.
        vector_db: The initialized ChromaDB instance.
    """
    
    if target_intent_name not in INTENT_DEFINITIONS and target_intent_name != 'Original':
        raise ValueError(f"Intent '{target_intent_name}' not defined.")
    
    definition = INTENT_DEFINITIONS[target_intent_name]
    
    parent_context_str = get_parent_interactions(dependency_ids, vector_db)

    prompt_template = ChatPromptTemplate.from_template(
        """
        You are a Manager at a company analyzing customer service call transcripts.
        You are currently conversing with an AI assistant that has access to these transcripts.
        
        ### Context
        **Transcript Summary/Topic:** {transcript_summary}

        **Previous Interaction(s) between You and the AI:**
        {parent_context}

        ### Your Task
        Generate a **single follow-up query** to the AI based on the Previous Interaction.
        
        **Constraint:** Your query MUST match the following intent strictly:
        **Intent Name:** {intent_name}
        **Definition:** {intent_definition}

        ### Output
        Return ONLY the follow-up query text. Do not add quotes or explanations.
        """
    )

    chain = (
        prompt_template 
        | llm 
        | StrOutputParser()
    )

    result = chain.invoke({
        "transcript_summary": transcript_summary,
        "parent_context": parent_context_str,
        "intent_name": target_intent_name,
        "intent_definition": definition
    })

    return result


if __name__ == "__main__":
    from langchain_community.embeddings import FakeEmbeddings
    
    embeddings = FakeEmbeddings(size=768)
    vector_db = Chroma(collection_name="manager_chat_history", embedding_function=embeddings)

    parent_id = "interaction_101"
    vector_db.add_texts(
        texts=[
            "Manager: How did the agent handle the customer's frustration?\n\n"
            "AI: The agent remained calm and used empathy statements like 'I understand how frustrating this is,' but failed to offer a concrete solution immediately."
        ],
        metadatas=[{
            "id": parent_id,
            "query_text": "How did the agent handle the customer's frustration?",
            "response_text": "The agent remained calm and used empathy statements...",
            "timestamp": "2023-10-27 10:00:00"
        }],
        ids=[parent_id]
    )

    transcript_context = "A billing dispute call where the customer was overcharged by $50."
    target_intent = "Adding/Specifying Condition" 
    
    try:
        generated_query = generate_followup_query(
            dependency_ids=[parent_id],
            target_intent_name=target_intent,
            transcript_summary=transcript_context,
            vector_db=vector_db
        )

        print(f"--- GENERATION RESULT ---")
        print(f"Parent Context: (Retrieved internally from ID {parent_id})")
        print(f"Selected Intent: {target_intent}")
        print(f"Generated Follow-up: {generated_query}")
        
        
    except Exception as e:
        print(f"Error: {e}")