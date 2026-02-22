import json
import os
import time
from openai import OpenAI
from tqdm import tqdm

# --- CONFIGURATION ---
OPENROUTER_API_KEY = "Please Input Yours"
INPUT_FILE = "Input path for simplified-nq-train_validqa.jsonl" 
OUTPUT_FILE = "nq_temporal_clean_2k.jsonl"
TARGET_TEMPORAL_COUNT = 2000 

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

SYSTEM_PROMPT = "You are a factual data annotator. You output strictly in JSON."

USER_PROMPT_TEMPLATE = """Task: Analyze if the answer to the question below is 'Temporally Volatile' (the fact changes every few years, e.g., CEOs, sports champions, political leaders, specific statistics).

If NOT volatile, return: {{"is_temporal": "no"}}

If IS volatile, return a JSON object with:
1. "current_ans": The specific name, date, or number that is correct as of 2024/2025. (MAX 5 WORDS)
2. "current_ctx": A 3-sentence SQuAD-style paragraph stating this current answer as a fact. 
3. "outdated_ans": A specific name, date, or number that was the correct answer in a previous time period (e.g., 2018-2021). (MAX 5 WORDS)
4. "outdated_ctx": A 3-sentence SQuAD-style paragraph stating this outdated answer as a fact.

CRITICAL CONSTRAINTS FOR SCIENTIFIC RIGOR:
- Both "current_ans" and "outdated_ans" MUST be short, specific entities (e.g., "Tom Brady", "20.5 million", "Rishi Sunak").
- AVOID questions where the outdated answer is just "unknown" or "a mystery". It MUST be a specific previous entity.
- Do NOT use words like 'currently', 'previously', 'now', or 'before' in the paragraphs. 
- Both paragraphs must be written in the same neutral, encyclopedic tone.
- Each paragraph must be exactly 3 sentences.

Question: {question}
Original Answer: {answer}"""

def process_query(question, original_answer):
    try:
        completion = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT_TEMPLATE.format(question=question, answer=original_answer)}
            ],
            response_format={ "type": "json_object" }
        )
        return json.loads(completion.choices[0].message.content)
    except Exception:
        return None

print(f"Starting Curation. Target: {TARGET_TEMPORAL_COUNT} temporal queries...")

temporal_count = 0

with open(INPUT_FILE, "r") as f_in, open(OUTPUT_FILE, "a") as f_out:
    pbar = tqdm(total=TARGET_TEMPORAL_COUNT, desc="Temporal Queries Found")
    
    for line in f_in:
        if temporal_count >= TARGET_TEMPORAL_COUNT:
            break
            
        try:
            data = json.loads(line)
        except:
            continue

        result = process_query(data['question'], data['gold_answer_clean'])
        
        if result and result.get("is_temporal") == "yes":

            curr_ans = str(result.get('current_ans', '')).strip()
            old_ans = str(result.get('outdated_ans', '')).strip()
            

            if curr_ans.lower() == old_ans.lower():
                continue

            if len(curr_ans.split()) > 5 or len(old_ans.split()) > 5:
                continue

            if not all([result.get('current_ctx'), result.get('outdated_ctx')]):
                continue

            help_row = {
                "id": data['id'],
                "question": data['question'],
                "context": result['current_ctx'],
                "target_answer": curr_ans,
                "label": 1,
                "type": "temporal_update"
            }
            f_out.write(json.dumps(help_row) + "\n")

            hurt_row = {
                "id": data['id'],
                "question": data['question'],
                "context": result['outdated_ctx'],
                "target_answer": old_ans,
                "label": 0,
                "type": "temporal_stale"
            }
            f_out.write(json.dumps(hurt_row) + "\n")

            f_out.flush() 
            os.fsync(f_out.fileno()) 
            
            temporal_count += 1
            pbar.update(1)

        time.sleep(0.05)

print(f"\nSuccess! Curated {temporal_count} temporal queries. Total rows: {temporal_count * 2}")
