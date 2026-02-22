

import pandas as pd
import numpy as np
import json
import os
from scipy.stats import spearmanr
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings('ignore')
RESULTS_DIR = "./results/final_matrix/"
INTERVENTION_DIR = "./results/intervention/"

def load_jsonl(path):
    data = []
    if not os.path.exists(path):
        return pd.DataFrame()
    with open(path, 'r') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except:
                continue
    return pd.DataFrame(data)

def get_best_f1(y_true, scores):
    """Sweeps 100 thresholds to find the maximum possible F1 score."""
    if len(np.unique(y_true)) < 2:
        return 0.0
    
    best_f1 = 0
    thresholds = np.linspace(scores.min(), scores.max(), 100)
    for t in thresholds:
        y_pred = (scores < t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
    return best_f1

def generate_table_1():
    print("\n" + "="*90)
    print("TABLE 1")
    print("="*90)
    print(f"{'Model':<20} | {'Dataset':<12} | {'Corr (ρ)':<10} | {'P-Value':<10} | {'Utility F1'}")
    print("-" * 90)

    models = ["mistral_7b_base", "llama_8b_base", "llama_8b_inst", "llama_70b_base", "llama_70b_inst", "qwen_72b_base"]
    datasets = ["CONFIQA", "KRE_SQUAD", "NQ_TEMPORAL"]

    for model in models:
        for dataset in datasets:
            filename = f"FINAL_{model}_{dataset}.jsonl"
            path = os.path.join(RESULTS_DIR, filename)
            df = load_jsonl(path)
            
            if df.empty:
                continue

            rho, p = spearmanr(df['interaction_score'], df['margin'])

            f1 = get_best_f1(df['actual_utility'], df['interaction_score'])

            print(f"{model:<20} | {dataset:<12} | {rho:>10.4f} | {p:>10.2e} | {f1:>10.3f}")

def generate_table_2():
    print("\n" + "="*60)
    print("TABLE 2")
    print("="*60)
    
    path = os.path.join(RESULTS_DIR, "FINAL_llama_70b_inst_NQ_TEMPORAL.jsonl")
    df = load_jsonl(path)
    
    if df.empty:
        print("Data for Table 2 not found.")
        return

    f1_ours = get_best_f1(df['actual_utility'], df['interaction_score'])

    f1_conf = get_best_f1(df['actual_utility'], 1 - df['conf'])
    
    f1_overlap = get_best_f1(df['actual_utility'], 1 - df['overlap'])

    print(f"{'Metric Configuration':<25} | {'Utility F1'}")
    print("-" * 40)
    print(f"{'Confidence Only (Prior)':<25} | {f1_conf:>10.3f}")
    print(f"{'Overlap Only (Evidence)':<25} | {f1_overlap:>10.3f}")
    print(f"{'Interaction Score (Ours)':<25} | {f1_ours:>10.3f}")

def generate_table_3():
    print("\n" + "="*70)
    print("TABLE 3")
    print("="*70)
    print(f"{'Model Architecture':<25} | {'Self-Report Acc':<15} | {'Auditor F1'}")
    print("-" * 70)

    models = {
        "Llama-70B-Base": "",
        "Llama-70B-Instruct": "",
        "Qwen-72B-Base": ""
    }

    for name, filename in models.items():
        path = os.path.join(INTERVENTION_DIR, filename)
        df = load_jsonl(path)
        
        if df.empty:
            continue

        self_acc = (df['flagged_conflict'] == (1 - df['label'])).mean()
        auditor_f1 = get_best_f1(1 - df['label'], df['interaction_score'])

        print(f"{name:<25} | {self_acc:>15.1%} | {auditor_f1:>10.3f}")

if __name__ == "__main__":
    generate_table_1()
    generate_table_2()
    generate_table_3()
    print("\n" + "="*90)
    print("Evaluation Complete")
