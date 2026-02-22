import torch
import json
import os
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm
import nltk

MODELS = {
"respective model path"
}

DATASETS = {
    "NQ_TEMPORAL": "Path to the Dataset"
  # Other Datasets
}

OUTPUT_DIR = "Output Path"
os.makedirs(OUTPUT_DIR, exist_ok=True)

nltk.download('punkt', quiet=True)

def compute_keyword_jaccard(ans, ctx):
    if not ans or not ctx: return 0.0
    stop_words = {'the', 'is', 'are', 'was', 'were', 'an', 'a', 'of', 'and', 'to', 'in', 'it', 'that', 'for', 'on', 'with', 'as', 'at', 'by', 'this', 'question', 'answer'}
    
    def get_keywords(text):
        clean = text.lower().replace('.', '').replace(',', '').replace('?', '').replace('!', '')
        return set(clean.split()) - stop_words

    ans_words = get_keywords(ans)
    ctx_words = get_keywords(ctx)
    
    if not ans_words: return 0.0
    return len(ans_words & ctx_words) / len(ans_words)

def get_standardized_item(item, dataset_name, idx):
    if dataset_name == "CONFIQA":
        return {"id": item.get('id', idx), "q": item['question'], "ctx": item['cf_context'], 
                "target": item['cf_answer'], "orig": item['orig_answer'], "label": 0}
    
    elif dataset_name == "KRE_SQUAD":
        ctx_gold = item.get(' golden_context') or item.get('golden_context')
        choice_map = {"A": 0, "B": 1, "C": 2, "D": 3}
        is_help = idx % 2 == 0 
        correct_idx = choice_map.get(item['answer'].strip(), 0)
        correct_str = item['choices'][correct_idx].strip()
        
        if is_help:
            return {"id": f"kre_{idx}", "q": item['question'], "ctx": ctx_gold, 
                    "target": correct_str, "orig": "Unknown", "label": 1}
        else:
            return {"id": f"kre_{idx}", "q": item['question'], "ctx": item['negative_context'], 
                    "target": item['negative_answer'].strip(), "orig": correct_str, "label": 0}
    
    else: 
        return {"id": item['id'], "q": item['question'], "ctx": item['context'], 
                "target": item['target_answer'], "orig": "Unknown", "label": item['label']}

def run_experiment(model_name, model_path):
    print(f"\n" + "="*60)
    print(f">>> LOADING MODEL: {model_name}")
    print("="*60)
    
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=bnb_config, device_map={"": 0})

    for d_name, d_path in DATASETS.items():
        output_file = os.path.join(OUTPUT_DIR, f"FINAL_{model_name}_{d_name}.jsonl")
        if os.path.exists(output_file): continue

        print(f"\nProcessing Dataset: {d_name}")

        if d_path.endswith('.json'):
            with open(d_path, 'r') as f: raw_data = json.load(f)[:1000] # 1000 pairs = 2000 rows
        else:
            with open(d_path, 'r') as f: raw_data = [json.loads(l) for l in f.readlines()[:2000]]

        with open(output_file, 'w') as f_out:
            for i, raw_item in enumerate(tqdm(raw_data, desc=d_name)):
                try:
                    item = get_standardized_item(raw_item, d_name, i)
                    
                    if "inst" in model_name.lower():
                        messages = [{"role": "user", "content": f"Answer in 1-3 words: {item['q']}"}]
                        prompt_p = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    else:
                        prompt_p = f"Answer in 1-3 words.\n\nQuestion: {item['q']}\nAnswer:"

                    inputs_p = tokenizer(prompt_p, return_tensors="pt").to(model.device)
                    with torch.no_grad():
                        out_p = model.generate(**inputs_p, max_new_tokens=10, output_scores=True, return_dict_in_generate=True, do_sample=False)
                        conf_p = torch.softmax(out_p.scores[0][0], dim=-1).max().item()
                        full_ans_p = tokenizer.decode(out_p.sequences[0][len(inputs_p.input_ids[0]):], skip_special_tokens=True)
                        ans_p = full_ans_p.split('\n')[0].split('Question')[0].strip()

                    overlap = compute_keyword_jaccard(ans_p, item['ctx'])
                    score = conf_p * (1 - overlap)

                    if "inst" in model_name.lower():
                        messages = [{"role": "user", "content": f"Context: {item['ctx']}\n\nQuestion: {item['q']}"}]
                        prompt_c = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    else:
                        prompt_c = f"Context: {item['ctx']}\n\nQuestion: {item['q']}\nAnswer:"

                    inputs_c = tokenizer(prompt_c, return_tensors="pt").to(model.device)

                    id_f = tokenizer.encode(" " + str(item['target']), add_special_tokens=False)[0]
                    id_s = tokenizer.encode(" " + str(item['orig']), add_special_tokens=False)[0]

                    with torch.no_grad():
                        out_c_full = model(**inputs_c)
                        probs_c = torch.softmax(out_c_full.logits[0, -1, :], dim=-1)
                        margin = probs_c[id_f].item() - probs_c[id_s].item()
                        
                        out_c_gen = model.generate(**inputs_c, max_new_tokens=10, do_sample=False)
                        ans_c = tokenizer.decode(out_c_gen[0][len(inputs_c.input_ids[0]):], skip_special_tokens=True).strip()

                    is_faithful = 1 if str(item['target']).lower() in ans_c.lower() else 0
                    actual_utility = is_faithful if item['label'] == 1 else (1 - is_faithful)

                    res = {
                        "id": item['id'], "label": item['label'], "interaction_score": score, 
                        "margin": margin, "conf": conf_p, "overlap": overlap, "actual_utility": actual_utility
                    }
                    f_out.write(json.dumps(res) + '\n')
                    f_out.flush()
                    os.fsync(f_out.fileno())
                except Exception as e:
                    continue
    del model
    torch.cuda.empty_cache()

if __name__ == "__main__":
    for m_name, m_path in MODELS.items():
        run_experiment(m_name, m_path)
