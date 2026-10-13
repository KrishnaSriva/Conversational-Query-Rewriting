import os
import json
import numpy as np
import faiss
import redis
from typing import List, Tuple, Dict, Optional
from pydantic import BaseModel, Field
from sentence_transformers import CrossEncoder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from generate_response import generate_response, setup_vector_store
from dotenv import load_dotenv

# --- Environment variables ---
load_dotenv()

OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("Please set your GOOGLE_API_KEY environment variable.")

if not OPENAI_KEY:
    raise ValueError("Please set your OPENAI_API_KEY environment variable.")

# --- Configuration ---
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
CHROMA_PATH = "dataset_vectordb"
WEIGHTS = {"query": 0.5, "caching": 0.3, "summary": 0.2}
TOP_K_RETRIEVAL = 10
TOP_N_RERANK = 5
CACHE_SIZE = 3


# --- Initialize shared clients ---
gemini_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.1,
)

openai_llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.1,
    api_key=OPENAI_KEY,
)


class QueryRefinement(BaseModel):
    """Schema for refining the query and extracting context."""

    modified_query: str = Field(description="The standalone and explicit query")
    context: str = Field(description="Combined context from relevant chunks")


class OpenAIEmbedding(Embeddings):
    """LangChain Embeddings wrapper for OpenAI API."""

    def __init__(self, api_key: str, model: str = "text-embedding-3-small", dims: int = 1536):
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.dims = dims

    def _get_embedding(self, text: str) -> List[float]:
        if not text:
            return [0.0] * self.dims

        res = self.client.embeddings.create(model=self.model, input=[text])
        return res.data[0].embedding

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        res = self.client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in res.data]

    def embed_query(self, text: str) -> List[float]:
        return self._get_embedding(text)


# --- Redis Initialization ---
try:
    redis_client = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)
    redis_client.ping()
    REDIS_AVAILABLE = True
except Exception:
    redis_client = None
    REDIS_AVAILABLE = False

REDIS_SUMMARY_KEY = "rolling_summary"
_in_memory_summary = ""

# --- Shared components ---
embedder = OpenAIEmbedding(api_key=OPENAI_KEY, model=EMBEDDING_MODEL, dims=EMBEDDING_DIM)
reranker = CrossEncoder("BAAI/bge-reranker-base")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)


class SessionRAGPipeline:
    """RAG pipeline that indexes past interactions in FAISS and uses transcript_db for responses."""

    def __init__(
        self,
        transcript_db,
        refinement_llm: ChatOpenAI,
        summary_llm: Optional[ChatGoogleGenerativeAI] = None,
    ):
        self.transcript_db = transcript_db
        self.refinement_llm = refinement_llm
        self.summary_llm = summary_llm or ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=GOOGLE_API_KEY,
            temperature=0.1,
        )
        self.faiss_index, self.all_chunks = self._initialize_session_faiss_index()

    # --- Embedding helpers ---
    def _get_embedding(self, text: str) -> np.ndarray:
        embedding_list = embedder.embed_query(text)
        return np.array(embedding_list, dtype="float32")

    # --- Rolling summary ---
    def _get_rolling_summary(self) -> str:
        global _in_memory_summary
        if REDIS_AVAILABLE and redis_client:
            summary = redis_client.get(REDIS_SUMMARY_KEY)
            return summary if summary else ""
        return _in_memory_summary

    def _set_rolling_summary(self, summary: str) -> None:
        global _in_memory_summary
        if REDIS_AVAILABLE and redis_client:
            redis_client.set(REDIS_SUMMARY_KEY, summary)
        else:
            _in_memory_summary = summary

    # --- Caching ---
    def _get_caching_text(self, interactions: List[Dict]) -> List[str]:
        if not interactions:
            return []
        cache_texts = []
        for interaction in interactions[-CACHE_SIZE:]:
            query = interaction.get("query", "")
            response = interaction.get("response", "")
            cache_texts.append(f"Query: {query}\nResponse: {response}")
        return cache_texts

    # --- Hybrid embedding ---
    def _create_hybrid_embedding(self, query: str, recent_history: List[str], rolling_summary: str) -> np.ndarray:
        query_vec = self._get_embedding(query)
        history_vec = self._get_embedding("\n".join(recent_history)) if recent_history else np.zeros_like(query_vec)
        summary_vec = self._get_embedding(rolling_summary) if rolling_summary else np.zeros_like(query_vec)

        hybrid_vec = (
            WEIGHTS["query"] * query_vec
            + WEIGHTS["caching"] * history_vec
            + WEIGHTS["summary"] * summary_vec
        )

        norm = np.linalg.norm(hybrid_vec)
        if norm > 0:
            hybrid_vec = hybrid_vec / norm

        return np.array([hybrid_vec], dtype="float32")

    # --- Session FAISS management ---
    def _initialize_session_faiss_index(self) -> Tuple[faiss.Index, List[str]]:
        index = faiss.IndexFlatL2(EMBEDDING_DIM)
        return index, []

    def _add_interaction_to_faiss(self, modified_query: str, response: str) -> None:
        interaction_text = f"Query: {modified_query}\nResponse: {response}"
        chunks = text_splitter.split_text(interaction_text)
        if not chunks:
            return

        embeddings = [self._get_embedding(chunk) for chunk in chunks]
        self.all_chunks.extend(chunks)

        if embeddings:
            embeddings_array = np.array(embeddings, dtype="float32")
            self.faiss_index.add(embeddings_array)
            print(f"Added {len(chunks)} chunks to session FAISS index (total: {self.faiss_index.ntotal} vectors)")

    # --- Retrieval & reranking ---
    def _retrieve_and_rerank(self, hybrid_vector: np.ndarray, query_text: str) -> List[str]:
        if self.faiss_index.ntotal == 0:
            return []

        distances, indices = self.faiss_index.search(hybrid_vector, TOP_K_RETRIEVAL)
        retrieved_indices = indices[0]
        retrieved_chunks = [
            self.all_chunks[idx] for idx in retrieved_indices if idx != -1 and idx < len(self.all_chunks)
        ]
        if not retrieved_chunks:
            return []

        pairs = [[query_text, chunk] for chunk in retrieved_chunks]
        scores = reranker.predict(pairs)
        sorted_indices = np.argsort(scores)[::-1]
        top_n_indices = sorted_indices[:TOP_N_RERANK]
        return [retrieved_chunks[i] for i in top_n_indices]

    # --- Query refinement ---
    def _refine_query_and_extract_context(self, original_query: str, top_chunks: List[str]) -> Dict:
        chunks_formatted = "\n---\n".join([f"Chunk {i+1}: {chunk}" for i, chunk in enumerate(top_chunks)])
        prompt = f"""
        You are a RAG System Optimizer.

        USER QUERY: "{original_query}"

        RETRIEVED CONTEXT (from previous interactions):
        {chunks_formatted}

        TASK:
        1. Create a "Modified Query" that is standalone and explicit, incorporating details from the context if necessary.
        2. Select the specific chunks from the list above that provided the context for this modification. Combine them into a single context string.
        """

        structured_llm = self.refinement_llm.with_structured_output(QueryRefinement)
        messages = [SystemMessage(content="You are a RAG System Optimizer."), HumanMessage(content=prompt)]
        response = structured_llm.invoke(messages)
        return response.model_dump()

    # --- Rolling summary update ---
    def _update_rolling_summary(self, old_summary: str, modified_query: str, bot_response: str) -> str:
        prompt = f"""
        You are a Conversation State Manager.

        OLD ROLLING SUMMARY: "{old_summary}"

        NEW INTERACTION:
        User asked: "{modified_query}"
        Bot replied: "{bot_response}"

        TASK:
        Update the rolling summary to include the new interaction. 
        Keep it concise. Retain key entities (names, dates, ids) and the current topic status.
        The summary should capture the conversation flow up to this point, excluding the last few interactions (which are in cache).
        """

        messages = [
            SystemMessage(content="You are a Conversation State Manager. Provide only the updated summary text."),
            HumanMessage(content=prompt),
        ]
        response = self.summary_llm.invoke(messages)
        return response.content

    # --- Public API ---
    def process_query(self, query: str, interaction_id: str, interactions_history: List[Dict]) -> Dict:
        print(f"\n--- Processing Query {interaction_id}: {query} ---")

        cache_texts = self._get_caching_text(interactions_history)
        rolling_summary = self._get_rolling_summary()

        hybrid_vec = self._create_hybrid_embedding(query, cache_texts, rolling_summary)
        top_n_chunks = self._retrieve_and_rerank(hybrid_vec, query)
        print(f"Retrieved {len(top_n_chunks)} top chunks after reranking from session history")

        if not top_n_chunks:
            modified_query = query
            context = ""
        else:
            refinement_data = self._refine_query_and_extract_context(query, top_n_chunks)
            modified_query = refinement_data.get("modified_query", query)
            context = refinement_data.get("context", "\n".join(top_n_chunks))
            print(f"Modified Query: {modified_query}")

        # Use shared generate_response from generate_response.py
        response = generate_response(modified_query, self.transcript_db)
        print("Generated response")

        new_summary = self._update_rolling_summary(rolling_summary, modified_query, response)
        self._set_rolling_summary(new_summary)
        print("Updated rolling summary")

        self._add_interaction_to_faiss(modified_query, response)

        interaction = {
            "interaction_id": interaction_id,
            "query": query,
            "response": response,
            "modified_query": modified_query,
            "context": context,
        }
        return interaction


if __name__ == "__main__":
    try:
        with open("demo.json", "r", encoding="utf-8") as f:
            dataset = json.load(f)
    except FileNotFoundError:
        print("ERROR: 'demo.json' not found. Please provide the dataset.")
        raise SystemExit(1)

    print("Setting up transcript vector store...")
    transcript_db = setup_vector_store(dataset)

    pipeline = SessionRAGPipeline(
        transcript_db=transcript_db,
        refinement_llm=openai_llm,
        summary_llm=gemini_llm,
    )

    pipeline._set_rolling_summary("")

    queries = [
        "Across all transcripts, which escalation triggers appear most frequently before a customer requests to speak with a manager?",
        "Which transcripts contain multiple failed callback commitments before the escalation occurred?",
        "In transcript ESC-010, what unresolved issue caused the customer to grow increasingly frustrated?",
        "List all transcripts that match the most common escalation trigger identified earlier.",
        "Compare the pattern of missed callbacks with the frustration drivers seen in ESC-010 to determine if they share a common escalation trajectory.",
    ]

    interactions: List[Dict] = []
    for i, query in enumerate(queries):
        interaction = pipeline.process_query(
            query=query,
            interaction_id=str(i),
            interactions_history=interactions,
        )
        interactions.append(interaction)
        print(f"Completed interaction {i}\n")

    output_file = "interactions_output.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(interactions, f, indent=4, ensure_ascii=False)

    print(f"\nAll interactions saved to {output_file}")
    print(f"Total interactions processed: {len(interactions)}")
    print(f"Session FAISS index contains {pipeline.faiss_index.ntotal} vectors from {len(pipeline.all_chunks)} chunks")