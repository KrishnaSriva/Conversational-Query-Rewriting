import os
import shutil
from dotenv import load_dotenv
from typing import List, Dict
import json
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_openai import ChatOpenAI
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
llm = ChatOpenAI(model = 'gpt-4o-mini',
                 temperature=0.7)
CHROMA_PATH = "dataset_vectordb"

with open('intent_definitions.json' , "r" , encoding="utf-8") as f:
    INTENT_DEFINITIONS = json.load(f)
def create_documents_from_json(data: List[Dict]) -> List[Document]:
    """
    Converts the structured JSON conversation log into LangChain Documents.
    It flattens the conversation flow into a single string for the AI to read.
    """
    documents = []
    
    for entry in data:
        dialogue_text = ""
        for turn in entry.get("conversation", []):
            dialogue_text += f"{turn['speaker']}: {turn['normalized']}\n"
            
        page_content = (
            f"Transcript ID: {entry['transcript_id']}\n"
            # f"Date: {entry['time_of_interaction']}\n"
            f"Domain: {entry['domain']}\n"
            f"Intent: {entry['intent']}\n"
            f"Reason for Call: {entry['reason_for_call']}\n"
            f"Summary: {entry['summary']}\n\n"
            f"--- Full Conversation ---\n"
            f"{dialogue_text}"
        )
        
        metadata = {
            "transcript_id": entry['transcript_id'],
            "domain": entry['domain'],
            "intent": entry['intent']
        }
        
        documents.append(Document(page_content=page_content, metadata=metadata))
        
    return documents

def setup_vector_store(data: List[Dict], reset_db: bool = True):
    """
    Ingests data, creates embeddings, and builds the Chroma index.
    """
    if reset_db and os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    print("Initializing ChromaDB and processing documents...")
    docs = create_documents_from_json(data)
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vector_store = Chroma.from_documents(
        documents=docs, 
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    return vector_store

def generate_response(query: str, vector_store) -> str:
    """
    Takes a query, retrieves the most relevant transcript context, 
    and generates an answer using Groq.
    """
    prompt = ChatPromptTemplate.from_template("""
    You are an expert customer service analyst. Use the provided context from customer care transcripts to answer the question at the end.

    Answer the user's question based strictly on the provided context (Customer Transcripts).
    If the answer is not in the context, say "I cannot find that information in the transcript." Don't make up information.

    When relevant, include specific details like:
    - Customer names
    - Agent names
    - Dates and times
    - Issues or complaints
    - Resolutions provided
    - Booking details
    - Loyalty program information

    Context:
    {context}

    Question: {input}

    Detailed Answer:
    """)

    document_chain = create_stuff_documents_chain(llm, prompt)
    
    retriever = vector_store.as_retriever(search_kwargs={"k": 20})
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    response = retrieval_chain.invoke({"input": query})
    
    return response["answer"]



if __name__ == "__main__":
    import json

    with open("demo.json") as f:
        dataset = json.load(f)

    vector_db = setup_vector_store(dataset)

    print("-" * 50)
    user_query = "Generally how the escalations happen during a call"
    print(f"Query: {user_query}")
    
    result = generate_response(user_query, vector_db)
    
    print(f"\nResponse:\n{result}")
    print("-" * 50)