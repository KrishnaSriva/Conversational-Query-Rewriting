import os
import json
import asyncio
from openai import AsyncOpenAI
import random
from dotenv import load_dotenv

load_dotenv()
OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_KEY:
    raise ValueError("Set OPENAI_API_KEY in environment before running.")

llm = AsyncOpenAI(api_key=OPENAI_KEY)

# ----------------------------
# Business events and question angles
# ----------------------------
BUSINESS_EVENTS = [
    "Escalation",
    "Refund",
    "Churn",
    "Product Improvement",
    "Complaint",
    "Cancellation",
    "Retention",
    "Upsell/Cross-sell",
    "Service Failure",
    "Customer Satisfaction Issue"
]

QUESTION_ANGLES = ["Implication Analysis", "Conflict Detection", "Hypothetical Reasoning"]

# ----------------------------
# Load transcripts
# ----------------------------
def load_transcripts(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    transcripts = []
    for entry in data:
        # Try both field names for compatibility
        convo = entry.get("conversation_normalised", entry.get("conversation", []))
        tid = entry.get("transcript_id", "NA")
        domain = entry.get("domain", "NA")
        intent = entry.get("intent", "NA")
        summary = entry.get("summary", "")
        messages = [
            f'[{tid}] "{t.get("speaker","UNK")}": "{t.get("normalized","")}"'
            for t in convo
        ]
        transcripts.append({
            "transcript_id": tid,
            "domain": domain,
            "intent": intent,
            "content": "\n".join(messages),
            "messages": messages,
            "summary": summary
        })
    return transcripts

# ----------------------------
# Select conversations from a group
# ----------------------------
def select_conversations(transcript_group, num_conversations):
    """
    Select a specific number of conversations from the transcript group.
    Returns the full conversations (not sampled messages).
    """
    if len(transcript_group) > num_conversations:
        selected = random.sample(transcript_group, num_conversations)
    else:
        selected = transcript_group
    return selected

# ----------------------------
# Detect business events in conversations
# ----------------------------
async def detect_business_events(selected_conversations, domain, intent):
    """
    Detect which business events are present in the selected conversations.
    Returns a list of detected business events.
    """
    # Format conversation summaries for the prompt
    summaries_text = "\n\n".join([
        f"Conversation {i+1} (ID: {t['transcript_id']}):\n{t.get('summary', 'No summary available')}"
        for i, t in enumerate(selected_conversations)
    ])
    
    business_events_str = ", ".join(BUSINESS_EVENTS)
    
    prompt = f"""
Analyze the following conversation summaries from domain '{domain}' and intent '{intent}'.
Identify which of the following key business events are present or relevant in these conversations:
{business_events_str}

Conversation Summaries:
{summaries_text}

Return ONLY a comma-separated list of business events that are present. If none are clearly present, return "None".
Example: "Escalation, Complaint" or "None"
"""
    resp = await llm.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100
    )
    text = resp.choices[0].message.content.strip()
    
    # Parse the response
    if text.lower() == "none" or not text:
        return []
    
    detected_events = []
    for event in BUSINESS_EVENTS:
        if event.lower() in text.lower():
            detected_events.append(event)
    
    return detected_events

# ----------------------------
# Generate question directly from conversations
# ----------------------------
async def generate_question(selected_conversations, angle, domain, intent, business_events=None):
    """
    Generate a general question from selected conversations using summaries.
    Focuses on business events if they are present.
    """
    # Format conversation summaries for the prompt
    summaries_text = "\n\n".join([
        f"Conversation {i+1} (ID: {t['transcript_id']}):\n{t.get('summary', 'No summary available')}"
        for i, t in enumerate(selected_conversations)
    ])
    
    # Build prompt based on whether business events are present
    if business_events and len(business_events) > 0:
        events_str = ", ".join(business_events)
        prompt = f"""
Based on the following conversation summaries from domain '{domain}' and intent '{intent}',
generate a general {angle} question that focuses on the key business events present: {events_str}.

The question should:
- Be relevant to the business events ({events_str}) present in these conversations
- Be general enough to apply to similar conversations in this domain/intent
- Not be overly specific to exact details in these conversations
- Be suitable for critical thinking and analysis about these business events

Conversation Summaries:
{summaries_text}

Generate one general question focusing on the business events:
"""
    else:
        prompt = f"""
Based on the following conversation summaries from domain '{domain}' and intent '{intent}',
generate a general {angle} question that captures the main themes or patterns
across these conversations. The question should be:
- General enough to apply to similar conversations in this domain/intent
- Not overly specific to exact details in these conversations
- Suitable for critical thinking and analysis

Conversation Summaries:
{summaries_text}

Generate one general question:
"""
    
    resp = await llm.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200
    )
    return resp.choices[0].message.content.strip()

# ----------------------------
# Generate answer + evidence
# ----------------------------
async def generate_answer(question, selected_conversations):
    """
    Generate answer and evidence from the selected conversations.
    Uses summaries for context but can reference full conversations for evidence.
    """
    # Format conversation summaries for the prompt (much smaller than full conversations)
    summaries_text = "\n\n".join([
        f"Conversation {i+1} (ID: {t['transcript_id']}):\n{t.get('summary', 'No summary available')}"
        for i, t in enumerate(selected_conversations)
    ])
    
    prompt = f"""
Answer the following question using ONLY the provided conversation summaries.
Provide a detailed answer based on the summaries. If you need to reference specific evidence,
mention which conversation (by ID) and what aspect supports your answer.

Conversation Summaries:
{summaries_text}

Question:
{question}
"""
    resp = await llm.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400
    )
    text = resp.choices[0].message.content.strip()

    evidence_list = []
    if "Evidence:" in text:
        answer, evidence_text = text.split("Evidence:", 1)
        for line in evidence_text.split("\n"):
            if line.strip():
                evidence_list.append(line.strip())
    else:
        # Extract answer and find relevant evidence from full conversations
        answer = text
        question_lower = question.lower()
        keywords = [word for word in question_lower.split() if len(word) > 3]  # Extract meaningful keywords

        # Search through all conversations for relevant evidence
        for conv in selected_conversations:
            for msg in conv["messages"]:
                msg_lower = msg.lower()
                if any(keyword in msg_lower for keyword in keywords):
                    evidence_list.append(msg.strip())

        # If no relevant evidence found, add a few key messages instead of all
        if not evidence_list:
            for conv in selected_conversations[:2]:  # Just from first 2 conversations
                evidence_list.extend(conv["messages"][:2])  # First 2 messages from each

    return answer.strip(), evidence_list

# ----------------------------
# Generate QA for a transcript group
# ----------------------------
async def generate_qa_for_group(transcript_group, num_questions=5, num_conversations=10):
    """
    Generate QA pairs for a transcript group.
    - Selects a specific number of conversations
    - Detects business events present in those conversations
    - Generates questions focused on business events (if present)
    - Creates answers with evidence
    """
    results = []
    domain = transcript_group[0]["domain"]
    intent = transcript_group[0]["intent"]

    # Select conversations from the group
    selected_conversations = select_conversations(transcript_group, num_conversations)
    
    # Detect business events present in the selected conversations
    business_events = await detect_business_events(selected_conversations, domain, intent)
    
    # Generate questions using different angles
    generated_count = 0
    for angle in QUESTION_ANGLES:
        if generated_count >= num_questions:
            break
        
        # Generate one question per angle, focusing on business events if present
        question = await generate_question(selected_conversations, angle, domain, intent, business_events)
        answer, evidence = await generate_answer(question, selected_conversations)
        
        results.append({
            "question": question,
            "answer": answer,
            "evidence": evidence,
            "domain": domain,
            "intent": intent,
            "business_events": business_events if business_events else [],
            "num_conversations_used": len(selected_conversations)
        })
        generated_count += 1
    
    # If we need more questions, generate additional ones with random angles
    while generated_count < num_questions:
        angle = random.choice(QUESTION_ANGLES)
        question = await generate_question(selected_conversations, angle, domain, intent, business_events)
        answer, evidence = await generate_answer(question, selected_conversations)
        
        results.append({
            "question": question,
            "answer": answer,
            "evidence": evidence,
            "domain": domain,
            "intent": intent,
            "business_events": business_events if business_events else [],
            "num_conversations_used": len(selected_conversations)
        })
        generated_count += 1
    
    return results

# ----------------------------
# Group transcripts by domain + intent
# ----------------------------
def group_transcripts(transcripts):
    grouped = {}
    for t in transcripts:
        key = (t["domain"], t["intent"])
        grouped.setdefault(key, []).append(t)
    return list(grouped.values())

# ----------------------------
# Main async testset generation
# ----------------------------
async def generate_testset(input_file="file.json", output_file="cross_transcript_testset.json", 
                           num_questions_per_group=5, num_conversations_per_group=10):
    """
    Generate testset from transcripts.
    
    Args:
        input_file: Path to input JSON file with transcripts
        output_file: Path to output JSON file for QA pairs
        num_questions_per_group: Number of questions to generate per domain+intent group
        num_conversations_per_group: Number of conversations to select per domain+intent group
    """
    transcripts = load_transcripts(input_file)
    grouped_transcripts = group_transcripts(transcripts)
    
    print(f"Found {len(grouped_transcripts)} domain+intent groups")
    print(f"Generating {num_questions_per_group} questions per group using {num_conversations_per_group} conversations each")
    
    tasks = [
        generate_qa_for_group(group, num_questions=num_questions_per_group, num_conversations=num_conversations_per_group)
        for group in grouped_transcripts
    ]
    all_results_nested = await asyncio.gather(*tasks)
    all_results = [qa for sublist in all_results_nested for qa in sublist]
    
    # Save
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(all_results)} QA pairs to {output_file}")

# ----------------------------
# Run as script
# ----------------------------
if __name__ == "__main__":
    print("Starting test set generation...")
    input_json = "file.json"
    output_json = "cross_transcript_testset.json"
    num_questions_per_group = 3
    num_conversations_per_group = 10  # Number of conversations to pick per domain+intent
    print(f"Input: {input_json}, Output: {output_json}")
    print(f"Questions per group: {num_questions_per_group}, Conversations per group: {num_conversations_per_group}")
    asyncio.run(generate_testset(input_json, output_json, num_questions_per_group, num_conversations_per_group))
    print("Test set generation completed!")