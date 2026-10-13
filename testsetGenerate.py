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
# Question angles / templates
# ----------------------------
QUESTION_ANGLES = ["Implication Analysis", "Conflict Detection", "Hypothetical Reasoning"]
QUESTION_TEMPLATE = (
    "Based on the following key point extracted from multiple conversations "
    "related to domain '{domain}' and intent '{intent}': '{key_point}', "
    "generate a {angle} critical thinking question comparing or synthesizing "
    "the transcripts."
)

# ----------------------------
# Load transcripts
# ----------------------------
def load_transcripts(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    transcripts = []
    for entry in data:
        convo = entry.get("conversation_normalised", [])
        tid = entry.get("transcript_id", "NA")
        domain = entry.get("domain", "NA")
        intent = entry.get("intent", "NA")
        messages = [
            f'[{tid}] "{t.get("speaker","UNK")}": "{t.get("normalized","")}"'
            for t in convo
        ]
        transcripts.append({
            "transcript_id": tid,
            "domain": domain,
            "intent": intent,
            "content": "\n".join(messages),
            "messages": messages
        })
    return transcripts

# ----------------------------
# Sample messages preserving transcript context
# ----------------------------
def sample_messages(transcript_group, max_transcripts=50, max_messages_per_transcript=20):
    """
    Sample messages from up to max_transcripts transcripts.
    Preserve context by keeping each transcript's messages together.
    Maintains chronological order within each transcript.
    """
    if len(transcript_group) > max_transcripts:
        transcript_group = random.sample(transcript_group, max_transcripts)

    sampled_msgs = []
    for t in transcript_group:
        msgs = t["messages"]
        if len(msgs) > max_messages_per_transcript:
            indices = random.sample(range(len(msgs)), max_messages_per_transcript)
            indices.sort()  # preserve chronological order
            msgs = [msgs[i] for i in indices]
        sampled_msgs.extend(msgs)

    return "\n".join(sampled_msgs)

# ----------------------------
# Generate key points
# ----------------------------
async def generate_key_points(domain, intent, sampled_text, max_points=3):
    prompt = f"""
Analyze the following conversations (all same domain: {domain}, intent: {intent})
and extract up to {max_points} key points, controversial statements,
or complex ideas that can be used for critical thinking questions.
Return each point on a separate line.

Conversations:
{sampled_text}
"""
    resp = await llm.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400
    )
    text = resp.choices[0].message.content.strip()
    points = [p.strip() for p in text.split("\n") if p.strip()]
    return points[:max_points]

# ----------------------------
# Generate cross-transcript question
# ----------------------------
async def generate_question(key_point, angle, domain, intent):
    prompt = QUESTION_TEMPLATE.format(key_point=key_point, angle=angle, domain=domain, intent=intent)
    resp = await llm.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150
    )
    return resp.choices[0].message.content.strip()

# ----------------------------
# Generate answer + evidence
# ----------------------------
async def generate_answer(question, sampled_text):
    prompt = f"""
Answer the following question using ONLY the provided conversations.
Provide a detailed answer AND quote the exact parts of the conversations that support your answer.

Conversations:
{sampled_text}

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
        # Improved fallback: extract relevant sentences that might contain keywords from the question
        answer = text
        question_lower = question.lower()
        keywords = [word for word in question_lower.split() if len(word) > 3]  # Extract meaningful keywords

        for msg in sampled_text.split("\n"):
            msg_lower = msg.lower()
            if any(keyword in msg_lower for keyword in keywords):
                evidence_list.append(msg.strip())

        # If no relevant evidence found, add a few key messages instead of all
        if not evidence_list:
            messages = [msg for msg in sampled_text.split("\n") if msg.strip()]
            evidence_list = messages[:3]  # Just add first 3 messages as fallback

    return answer.strip(), evidence_list

# ----------------------------
# Generate QA for a transcript group
# ----------------------------
async def generate_qa_for_group(transcript_group, num_questions=5):
    results = []
    domain = transcript_group[0]["domain"]
    intent = transcript_group[0]["intent"]

    # Sample messages once for the entire group to ensure consistent context
    sampled_text = sample_messages(transcript_group, max_transcripts=50, max_messages_per_transcript=20)

    key_points = await generate_key_points(domain, intent, sampled_text, max_points=num_questions)
    generated_count = 0
    for point in key_points:
        for angle in QUESTION_ANGLES:
            if generated_count >= num_questions:
                break
            question = await generate_question(point, angle, domain, intent)
            answer, evidence = await generate_answer(question, sampled_text)
            results.append({
                "question": question,
                "answer": answer,
                "evidence": evidence
            })
            generated_count += 1
        if generated_count >= num_questions:
            break
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
async def generate_testset(input_file="file.json", output_file="cross_transcript_testset.json", num_questions_per_group=5):
    transcripts = load_transcripts(input_file)
    grouped_transcripts = group_transcripts(transcripts)
    tasks = [
        generate_qa_for_group(group, num_questions=num_questions_per_group)
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
    print(f"Input: {input_json}, Output: {output_json}, Questions per group: {num_questions_per_group}")
    asyncio.run(generate_testset(input_json, output_json, num_questions_per_group))
    print("Test set generation completed!")