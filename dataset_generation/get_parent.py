import numpy as np
from typing import List, Dict, Any
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from get_dependency import sample_next

def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Calculates cosine similarity between two vectors."""
    if vec_a is None or vec_b is None:
        return 0.0
    a = np.array(vec_a)
    b = np.array(vec_b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def bell_curve_score(target_index: int, total_length: int) -> float:
    """
    Calculates a score based on a Split Gaussian (Asymmetric Bell Curve).
    
    Behavior:
    - Middle indices: Highest score (~1.0).
    - Recent indices (Right tail): Steep decay -> Very small score.
    - Old indices (Left tail): Gradual decay -> Moderate score (higher than recent).
    """
    if total_length <= 1:
        return 1.0

    mu = total_length / 2 
    
    sigma_left = total_length * 0.5
    sigma_right = total_length * 0.2
    
    if target_index <= mu:
        sigma = sigma_left
    else:
        sigma = sigma_right
        
    if sigma < 1e-6: 
        sigma = 1.0

    score = np.exp(-((target_index - mu) ** 2) / (2 * sigma ** 2))
    
    return float(score)


def calculate_utility(metrics: Dict[str, float]) -> float:
    """
    Aggregates metrics into a single utility score.
    Adjust weights based on what you value most (e.g., correctness).
    """
    if not metrics:
        return 0.0
    
    return (
        metrics.get("relevancy", 0) * 0.3 +
        metrics.get("completeness", 0) * 0.2 +
        metrics.get("correctness", 0) * 0.3 +
        metrics.get("novelty", 0) * 0.2
    )

# --- Dependency Resolution Functions ---

def resolve_direct_dependency(conversation: List[Dict]) -> str:
    """
    Resolves Direct Dependency: Returns the interaction_id of the immediately preceding query.
    """   
    return conversation[-1].get("interaction_id")

def resolve_long_range_dependency(
    conversation: List[Dict], 
    vector_db: Chroma, 
    embeddings_model: Embeddings,
    alpha: float = 0.5, # Weight for Similarity
    beta: float = 0.3,  # Weight for History/Recency
    gamma: float = 0.2  # Weight for Utility
) -> str:
    """
    Resolves Long-Range Dependency using the formula:
    Score = α·SimScore + β·DepHistoryScore + γ·Utility
    """
    if len(conversation) < 2:
        return resolve_direct_dependency(conversation)

    current_interaction = conversation[-1]
    current_id = current_interaction["interaction_id"]
    
    current_text = current_interaction.get("query", "")
    if current_interaction.get("response"):
        current_text += "\n\n" + current_interaction.get("response")
        
    current_embedding = vector_db.get(
        ids=[current_id],
        include=["embeddings"]
    )
    current_embedding = current_embedding["embeddings"][0]
    # print(current_embedding)
    

    candidate_interactions = conversation[:-1] 
    
    best_candidate_id = None
    max_utility_estimate = -float('inf')
    
    for idx, candidate in enumerate(candidate_interactions):
        cand_id = candidate["interaction_id"]

        db_record = vector_db.get(
            ids=[cand_id], 
            include=["embeddings", "metadatas", "documents"]
        )
        
        if not db_record['ids']:
            continue

        cand_embedding = db_record['embeddings'][0]
        cand_metadata = db_record['metadatas'][0]
        sim_score = cosine_similarity(current_embedding, cand_embedding)
        
        hist_score = bell_curve_score(idx, len(candidate_interactions))

        cand_metrics = {
            "relevancy": cand_metadata.get("relevancy", 0),
            "completeness": cand_metadata.get("completeness", 0),
            "correctness": cand_metadata.get("correctness", 0),
            "novelty": cand_metadata.get("novelty", 0)
        }
        util_score = calculate_utility(cand_metrics)
        
        candidate_utility_estimate = (alpha * sim_score) + (beta * hist_score) + (gamma * util_score)
        
        if candidate_utility_estimate > max_utility_estimate:
            max_utility_estimate = candidate_utility_estimate
            best_candidate_id = cand_id

    return best_candidate_id


def resolve_dependency_chain(
    dependency_type: str,
    conversation: List[Dict[str, Any]], 
    vector_db: Chroma, 
    embeddings_model: Embeddings
) -> str:
    """
    Main entry point. Determines the dependency type of the latest interaction
    and calls the appropriate resolution function.
    
    Args:
        conversation: List of interaction dictionaries.
        vector_db: The initialized Langchain Chroma object.
        embeddings_model: The Langchain Embeddings model used for the current query.
        
    Returns:
        interaction_id: The ID of the parent interaction.
    """
    if not conversation:
        raise ValueError("Conversation list cannot be empty.")

    # current_interaction = conversation[-1]

    # print(f"Resolving dependency for: {current_interaction.get('interaction_id')}")
    # print(f"Detected Type: {dep_type}")

    if dependency_type == "Direct":
        parent_id = [resolve_direct_dependency(conversation)]
    
    elif dependency_type == "LongRange":
        parent_id = [resolve_long_range_dependency(conversation, vector_db, embeddings_model)]
    
    elif dependency_type == "Multi":
        parent_id = [resolve_direct_dependency(conversation)]
        parent_id.append(resolve_long_range_dependency(conversation, vector_db, embeddings_model))
        
    else:
        parent_id = [resolve_direct_dependency(conversation)]

    return parent_id


if __name__ == "__main__":
    # --- Setup for Demonstration ---
    from langchain_community.embeddings import FakeEmbeddings
    from langchain_chroma import Chroma
    import uuid

    # 1. Mock Embeddings
    embeddings = FakeEmbeddings(size=768)

    # 2. Initialize ChromaDB (Client side)
    vector_db = Chroma(
        collection_name="chat_history",
        embedding_function=embeddings
    )

    # 3. Create Dummy Data (Past Interactions)
    history_data = [
        {
            "interaction_id": "msg_1", 
            "query": "What is Python?", 
            "response": "Python is a language...", 
            "metrics": {"relevancy": 0.9, "correctness": 1.0}
        },
        {
            "interaction_id": "msg_2", 
            "query": "How do I install it?", 
            "response": "pip install python...", 
            "metrics": {"relevancy": 0.8, "correctness": 0.9}
        },
        {
            "interaction_id": "msg_3", 
            "query": "Tell me a joke", 
            "response": "Why did the chicken...", 
            "metrics": {"relevancy": 0.2, "correctness": 1.0} # Low relevancy to coding
        }
    ]

    print("Adding the Embeddings")

    # 4. Add to VectorDB
    for item in history_data:
        vector_db.add_texts(
            texts=[item["query"] + "\n\n" + item["response"]],
            metadatas=[{
                "interaction_id": item["interaction_id"],
                "relevancy": item["metrics"]["relevancy"], 
                "correctness": item["metrics"]["correctness"],
                "completeness": 0.8,
                "novelty": 0.5
            }],
            ids=[item["interaction_id"]]
        )

    # 5. The Conversation Context (including the new query)
    conversation_context = [
        {"interaction_id": "msg_1", "query": "What is Python?", "dependency_type": "Directly Dependent"},
        {"interaction_id": "msg_2", "query": "How do I install it?", "dependency_type": "Directly Dependent"},
        {"interaction_id": "msg_3", "query": "Tell me a joke", "dependency_type": "Directly Dependent"},
        # The New Query: Long-range dependent on msg_1 (Python) or msg_2
        {
            "interaction_id": "msg_4",
            "query": "What are its main features?", # "Its" implies Python, not the joke
            "dependency_type": "Long-range Dependency",
            "metrics": {}
        }
    ]

    print("Resolving Dependency Type")

    # 6. Run Resolution
    resolved_id = resolve_dependency_chain( sample_next(prev_dep="NoDep" ,temperature= 1.2), conversation_context, vector_db, embeddings)

    print(f"\nResolved Parent ID: {resolved_id}")
    # Expected: msg_1 or msg_2 (High Semantic Similarity), definitely not msg_3 (Joke)