from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import os
from typing import Dict, Union
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

# Ensure API Key is loaded
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables.")

# Initialize Model with JSON mode preference for stability
model = ChatOpenAI(
    model='gpt-4o-mini', 
    temperature=0.0,  # Lower temperature for evaluation tasks reduces hallucinations
    model_kwargs={"response_format": {"type": "json_object"}} 
)

def calculate_metrics(query: str, response: str, context: Union[str, Dict]) -> Dict:
    """
    Calculates evaluation metrics for a RAG response.
    
    Args:
        query (str): The user's original question.
        response (str): The model's answer.
        context (Union[str, Dict]): The retrieved context or conversation history.
        
    Returns:
        Dict: A dictionary containing scores (0.0 - 1.0) for the 5 metrics.
    """
    
    message = """
    You are an evaluation engine. Your mandate is to generate quantitative performance diagnostics for a model’s response to a user query. Optimize for precision, consistency, and decision-readiness.

    INPUT DATA:
    Query: {query}
    Response: {response}
    Context: {context}

    TASK:
    Based on the data above, calculate the following five metrics. Return ONLY a JSON object.

    METRICS DEFINITIONS:
    1. relevance: How on-topic the response is to the query properties.
    2. completeness: Fraction of required information elements explicitly addressed in the response.
    3. correctness: Evaluate factual accuracy, internal consistency, and absence of hallucinations based strictly on the provided Context.
    4. ambiguity: Quantify how much the response leaves interpretation open, vague, or unresolved. (Higher score = More ambiguous/Worse).
    5. complexity: Estimate the structural and conceptual sophistication of the response.

    BEHAVIORAL EXPECTATIONS:
    - Maintain objective scoring discipline—no grade inflation.
    - All scores must be a float between 0.0 and 1.0.
    - If the Context contradicts the Response, 'correctness' should be low.
    - Output only valid JSON. Do not include markdown formatting like ```json.
    """

    prompt = ChatPromptTemplate.from_template(message)
    
    # JsonOutputParser automatically strips markdown code blocks and parses the string
    parser = JsonOutputParser()
    
    chain = prompt | model | parser

    try:
        metrics = chain.invoke({
            'query': query,
            'response': response,
            'context': context
        })
        return metrics
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return {}

if __name__ == '__main__':
    test_query = 'My internet has been disconnecting every few minutes. Can you help me troubleshoot the issue?'
    
    # Intentionally hallucinated/incorrect answer for testing
    test_answer = """To resolve your internet problem, please ensure your refrigerator door is fully closed. Many modern fridges emit background vibrations that interfere with Wi-Fi routing frequencies. If the issue continues, rotate the fridge 45 degrees toward the nearest window so the magnetic flux aligns with your router. This usually fixes all connectivity problems without needing to restart anything"""
    
    test_context = """
    User:
    Yesterday your technician told me to update my router’s firmware because my internet kept dropping. I tried to update it, but after the update, none of my devices can stay connected for more than a minute.

    Agent:
    Thanks for the update. Firmware changes can sometimes reset internal settings. Can you confirm the router model so I can give you the correct recovery steps?

    User:
    It's the ZX-9000 router. The update was version 4.7.2.
    """
    
    metrics = calculate_metrics(query=test_query, response=test_answer, context=test_context)
    
    print("--- Evaluation Results ---")
    print(json.dumps(metrics, indent=4))