import os
import json
import time
from get_dependency import sample_next
from get_parent import resolve_dependency_chain
from get_intent import predict_tag
from generate_query import generate_followup_query
from query_modifier import generate_modified_query
from generate_response import generate_response, setup_vector_store
from generate_metircs import calculate_metrics

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from chromadb import PersistentClient

# --- Constants ---
NUM_OF_INTERACTIONS = 3
NEW_INDEPENDENT_QUERY = 10 
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# --- Setup Embeddings ---
embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# --- Setup ChromaDB (Hybrid Approach) ---
client = PersistentClient(path="./chroma_session_db")
collection = client.get_or_create_collection(name="session")

# LangChain Wrapper
langchain_vector_store = Chroma(
    client=client,
    collection_name="session",
    embedding_function=embedding
)

with open("demo.json" , "r" , encoding= "utf-8") as f:
    dataset = json.load(f)

transcript_db = setup_vector_store(dataset)

# --- Load Data ---
with open("query.json" , "r", encoding="utf-8") as f:
    data = json.load(f)

it = 0
conversation = []
# Renamed global variable to current_evidence for clarity
current_evidence = "" 

for i in range(NUM_OF_INTERACTIONS):
    interaction_id = str(i)
    parent_ids = [] # Initialize list for all parent IDs (even if only one is found)

    # --- Logic to determine Query Type ---
    if (i % NEW_INDEPENDENT_QUERY == 0):
        # NEW INDEPENDENT QUERY
        gold_interaction = data[it]
        it += 1
        query = gold_interaction.get("question")
        evidence_chunks = gold_interaction.get("evidence") 
        # Use evidence variable
        current_evidence = ("\n").join(evidence_chunks) 
        dependency_type = "NoDep"
        parent_ids = [] # No parent IDs
        intent = "Original"
        modified_query = query
        
        # FIX 1: Initialize context as an empty DICT for new independent queries
        context = {} 
        
        # For metrics calculation
        context_for_metrics = current_evidence

    else:
        # FOLLOW-UP QUERY
        prev_interaction = conversation[-1]
        prev_dependency = prev_interaction["metadata"].get("dependency_type", "NoDep")
        
        dependency_type = sample_next(prev_dep=prev_dependency)
        
        # NOTE: resolve_dependency_chain usually returns one ID. We treat it as one for now.
        parent_ids = resolve_dependency_chain(dependency_type, conversation, collection, embedding)
        
        # Ensure parent_ids is a list, even if single_parent_id is None or a single value
    

        intent = predict_tag(parent_ids[0] if parent_ids else None, collection)

        aggregated_parent_evidence = []
        context = {}
        # --- Multi-Parent Logic ---
        if parent_ids:
            for pid in parent_ids:
                parent_interaction = next(
                    (item for item in conversation if item["interaction_id"] == str(pid)), 
                    None
                )
                if parent_interaction:
                    # Aggregate evidence from all parents
                    parent_evidence_chunk = parent_interaction["metadata"].get("evidence", "")
                    aggregated_parent_evidence.append(parent_evidence_chunk)
                    context[f'{pid}'] = parent_interaction['metadata'].get('modified_query' , '') + " " + parent_interaction.get('response' , '') 


            # Update global evidence string by joining all aggregated chunks
            if aggregated_parent_evidence:
                current_evidence = ("\n").join(aggregated_parent_evidence)
        
        # Use the aggregated evidence for generating the follow-up query
        query = generate_followup_query(parent_ids, intent, current_evidence, collection)
        
        # Use the aggregated evidence for query modification
        modified_query = generate_modified_query(vector_db=collection, query=query, parent_context=current_evidence ,intent=intent)
        
        # For metrics calculation
        context_for_metrics = current_evidence

    # --- Generate Response ---
    response = generate_response(modified_query, transcript_db)
    
    # Calculate metrics using the aggregated evidence
    metrics = calculate_metrics(query=modified_query, response=response, context=context_for_metrics)

    # --- FIX: Ensure metrics is a Dict before processing ---
    if isinstance(metrics, str):
        try:
            metrics = json.loads(metrics)
        except json.JSONDecodeError:
            metrics = {} 

    # --- Store Data (Renamed context to evidence) ---
    interaction_entry = {
        "interaction_id": interaction_id,
        "query": query,
        "response": response,
        "metadata": {
            "modified_query": modified_query,
            "evidence": current_evidence,
            "dependency_type": dependency_type,
            "context" : context ,
            "parent_id": parent_ids, 
            "intent": intent,
            "metrics": metrics,
        }
    }
    conversation.append(interaction_entry)

    # --- Embed and Store (Native) ---
    document_content = f"{modified_query}\n\n{response}"
    embedding_vector = embedding.embed_query(document_content)

    chroma_metadata = {
        "query_text": query,
        "response_text": response,
        "modified_query": modified_query,
        "context_point": json.dumps(parent_ids) if parent_ids else "None", 
        # FIX 2: Serialize the Python dict 'context' to a JSON string for ChromaDB
        "context": json.dumps(context), 
        "evidence": current_evidence if current_evidence else "",
        "intent": intent,
        "dependency_type": dependency_type,
        "timestamp": int(time.time())
    }

    if metrics:
        for key, value in metrics.items():
            chroma_metadata[key] = value

    # We use the native collection to add manually constructed embeddings
    collection.add(
        ids=[interaction_id],
        embeddings=[embedding_vector],
        documents=[document_content],
        metadatas=[chroma_metadata]
    )

    print(f"Interaction {i} processed and stored.")


with open("conversation_dataset.json", "w" , encoding="utf-8") as f:
    json.dump(conversation, f, indent=4)