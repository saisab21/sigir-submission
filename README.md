# When RAG Disagrees: Detecting Latent Epistemic Conflict via Logit Interactions 

This repository contains the research artifacts, diagnostic benchmarks, and mechanistic analysis suite for our SIGIR 2026 submission. 

Our work identifies a training-free mechanistic law—the **Interaction Score**—that predicts internal epistemic conflict in Large Language Models (LLMs) during Retrieval-Augmented Generation (RAG). By analyzing high-resolution logit distributions across 70B-parameter models in full BF16 precision, we characterize the "Neural Tug-of-War" between a model's parametric memory and contradictory contextual evidence.

## 🔬 Core Scientific Contributions

1.  **The Interaction Law:** We formalize the relationship between parametric priors and contextual alignment as a closed-form predictor of internal logit tension ($p < 10^{-42}$).
2.  **The Alignment Paradox:** We provide empirical evidence that instruction tuning (RLHF) regularizes internal probability shifts, increasing mechanistic predictability while simultaneously decoupling it from external textual behavior.
3.  **Mechanistic Auditing:** We propose a 25ms, zero-training gatekeeper that identifies latent sycophancy and internal conflict, significantly outperforming model self-reporting accuracy.

---

## 📂 Research Artifacts

The provided scripts are designed to facilitate the extraction of internal model states and the statistical validation of the paper's primary findings:

### 1. Mechanistic Extraction & Probing
*   `mechanistic_signal_extractor.py`: This core engine performs dual-pass inference across Llama, Qwen, and Mistral families. It is designed to capture high-fidelity logit distributions at the primary epistemic bottleneck (the first token of divergence) to calculate Interaction Scores and Logit Margins.
*   `honesty_intervention_extractor.py`: This script implements the "Internal Conflict" probing protocol. It evaluates the discrepancy between internal tension and textual output by explicitly instructing the model to flag contradictions, thereby exposing the "Sycophancy Mask" induced by alignment.

### 2. Evaluation & Statistical Analysis
*   `master_evaluation_and_tables.py`: The primary analytical suite. This script aggregates raw logit-space data to reproduce the paper's statistical tables, including Spearman rank correlations ($\rho$), P-values across varying task complexities, and Utility F1 scores.

### 3. Diagnostic Benchmark: NQ-Temporal-2K
*   `dataset/`: Contains the **NQ-Temporal-2K** benchmark, curated from Natural Questions (NQ) using a model-aware procedural pipeline.
*   `unique_ids_temporal_dataset.txt`: Provides the specific NQ example IDs used to construct the benchmark, ensuring experimental reproducibility and preventing data leakage.
*   The benchmark utilizes high-density, 3-sentence SQuAD-style paragraphs to isolate the signatures of knowledge conflict from contextual noise.

---

## ⚙️ Technical Requirements and Precision

### Hardware and Signal Integrity
To preserve the mathematical integrity of the logit distributions and avoid quantization-induced noise, extraction should be performed on hardware capable of supporting 70B+ models in full **BF16 precision** (e.g., NVIDIA H200 or A100 80GB clusters).

### Reproducibility Statement
The provided scripts enable the reproduction of the master results matrix and the "Alignment Paradox" findings. The full, production-optimized codebase, including extended visualization suites and multi-modal extension modules, will be released upon the camera-ready version of the paper.

---

## 🔗 Anonymized Links
- **Repository:** [https://anonymous.4open.science/r/sigir-submission-AD25](https://anonymous.4open.science/r/sigir-submission-AD25)
