import os
import re
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Conversational Query Rewriter Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Turn(BaseModel):
    speaker: str  # 'user' or 'system'
    text: str

class RewriteRequest(BaseModel):
    query: str
    history: List[Turn] = []
    summary: Optional[str] = ""

class RewriteResponse(BaseModel):
    original_query: str
    rewritten_query: str
    extracted_context: str
    is_modified: bool
    resolved_entities: List[str]
    updated_summary: str

def get_groq_client():
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY is not configured.")
    from groq import Groq
    return Groq(api_key=groq_api_key)

def clean_text(text: str) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    cleaned = cleaned.strip().strip('"').strip("'")
    return cleaned

@app.get("/api/health")
def health():
    return {"status": "ok", "engine": "Conversational Query Rewriter (Qwen 3.8 27B on Groq)"}

@app.post("/api/rewrite", response_model=RewriteResponse)
def rewrite_query(req: RewriteRequest):
    client = get_groq_client()
    model = "qwen/qwen3.8-27b"

    # Build dialogue history
    recent_history_str = ""
    for turn in req.history[-6:]:
        recent_history_str += f"- {turn.text}\n"

    if not recent_history_str:
        recent_history_str = "(First turn - no prior history)"

    summary_context = req.summary.strip() if req.summary else "(None)"

    # Stage 1: Context & Entity Extraction
    context_prompt = f"""You are a conversational context-refinement analyzer.
Analyze the user's current query given the prior conversation history.

Summary of Prior Context:
{summary_context}

Previous Conversation Turns:
{recent_history_str}

Current Follow-up Query:
{req.query}

Identify:
1. Missing antecedent entities, topics, or reference resolutions needed to understand the query independently.
2. If the current query is already standalone and does not depend on prior context, output "Standalone query - no external context needed."
Keep it to 1-2 brief bullet points."""

    try:
        ctx_resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": context_prompt}],
            temperature=0.1,
            max_completion_tokens=200,
        )
        extracted_context = clean_text(ctx_resp.choices[0].message.content)
    except Exception:
        extracted_context = "Context inferred from prior interaction"

    # Stage 2: Standalone Query Reformulation
    rewrite_prompt = f"""You are a Conversational Query Rewriting (CQR) engine.
Goal: Transform the user's follow-up query into a single, unambiguous, fully self-contained search query.

Prior Turns:
{recent_history_str}

Extracted Context:
{extracted_context}

User Follow-up Query:
{req.query}

Instructions:
- If the query has pronouns ('it', 'they', 'those') or missing subjects/conditions ('what were the common reasons?'), resolve them using the prior context.
- If the query is already self-contained, return it unchanged.
- Return ONLY the final rewritten search query string. NO explanations, NO quotes, NO bullet points."""

    try:
        rw_resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": rewrite_prompt}],
            temperature=0.1,
            max_completion_tokens=150,
        )
        rewritten_query = clean_text(rw_resp.choices[0].message.content)
        if not rewritten_query:
            rewritten_query = req.query
    except Exception:
        rewritten_query = req.query

    # Check if modified
    norm_orig = re.sub(r"[^\w\s]", "", req.query.lower()).strip()
    norm_rew = re.sub(r"[^\w\s]", "", rewritten_query.lower()).strip()
    is_modified = norm_orig != norm_rew

    # Extract resolved entities / additions
    resolved_entities = []
    if is_modified:
        orig_words = set(norm_orig.split())
        rew_words = [w for w in norm_rew.split() if w not in orig_words]
        if rew_words:
            # Group into meaningful snippet
            resolved_entities = ["Added context: " + " ".join(rew_words[:8])]

    # Stage 3: Minimal Rolling Summary Update
    sum_prompt = f"""Update the short rolling summary of the user's investigation topic (maximum 2 sentences).
Previous Summary: {summary_context}
New Turn: {rewritten_query}
Return ONLY the updated summary."""

    try:
        sum_resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": sum_prompt}],
            temperature=0.1,
            max_completion_tokens=150,
        )
        updated_summary = clean_text(sum_resp.choices[0].message.content)
    except Exception:
        updated_summary = summary_context

    return RewriteResponse(
        original_query=req.query,
        rewritten_query=rewritten_query,
        extracted_context=extracted_context,
        is_modified=is_modified,
        resolved_entities=resolved_entities,
        updated_summary=updated_summary
    )
