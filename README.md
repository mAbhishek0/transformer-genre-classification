# Multi-Model Genre Classification & Book-to-Movie Mismatch Detection

![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)

## Brief Description
End-to-end NLP pipeline for 12-genre multi-label text classification comparing 4 transformer architectures, plus a novel book-to-movie narrative mismatch detection system.

## Project Overview
This project is structured into two main phases:
- **Phase 1: Genre Classification**: Compares DistilBERT, DeBERTa-v3, RoBERTa-base, and Longformer, along with an experiment in topic fusion.
- **Phase 2: Mismatch Detection**: Includes Wikipedia scraping and a 5-head scoring system for detecting narrative mismatch between books and their movie adaptations.

**Best result:** RoBERTa-base achieved a Test Macro-F1 score of **0.6870**.

## Repository Structure
```text
├── phase1_genre_classification/
│   ├── notebooks/
│   │   ├── 01_data_preparation.ipynb
│   │   ├── 02_distilbert_baseline.ipynb
│   │   ├── 03_deberta_training.ipynb
│   │   ├── 04_roberta_optimal.ipynb
│   │   ├── 05_topic_modeling_fusion.ipynb
│   │   └── full_pipeline_reference.ipynb
│   └── scripts/
│       ├── longformer_training.py
│       ├── generate_topics.py
│       ├── validate_topics.py
│       └── inference.py
├── phase2_mismatch_detection/
│   ├── notebooks/
│   │   └── mismatch_detection.ipynb
│   └── scripts/
│       ├── wikipedia_scraper.py
│       └── build_wiki_pairs.py
├── data/
│   ├── raw/          (source datasets)
│   ├── processed/    (cleaned data, splits, embeddings)
│   ├── wiki/         (Wikipedia scraped pairs)
│   └── topics/       (topic probability matrices)
├── models/           (gitignored - model weights)
├── results/          (outputs, logs, visualizations)
├── figures/          (architecture diagrams)
├── docs/             (proposal, requirements, walkthrough)
├── archive/          (superseded earlier versions)
├── report.tex        (LaTeX academic report)
└── report.pdf        (compiled report)
```

## Datasets
- **CMU Book Summaries**: [http://www.cs.cmu.edu/~dbamman/booksummaries.html](http://www.cs.cmu.edu/~dbamman/booksummaries.html) — 16,559 book summaries with Freebase genre annotations
- **Kaggle Movies Dataset**: [https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset) — 45,466 movie records with TMDB genre tags
- **Combined**: 51,618 samples, 12 canonical genres, 70/15/15 stratified split

## Model Performance

| Model | Val Macro-F1 | Test Macro-F1 | Test Micro-F1 | Hamming Loss | Training Time |
|-------|-------------|---------------|---------------|-------------|---------------|
| DistilBERT (baseline) | 0.6817 | 0.6831 | 0.6875 | 0.0984 | 1h 16m |
| DeBERTa-v3-base | 0.6510 | 0.6496 | 0.6689 | 0.1077 | 6h 33m |
| **RoBERTa-base** ⭐ | **0.6951** | **0.6870** | **0.6951** | **0.0975** | 3h 18m |
| Longformer | 0.6882 | — | — | — | 6h 46m |
| RoBERTa + Topic Fusion | 0.6883 | 0.6828 | 0.6897 | 0.1003 | 2h 47m |

## Quick Start
1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd NLP_Project_GitHub
   ```
2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the datasets from the links above and place them in `data/raw/`.
4. Run notebooks in order: `01` → `02` → `03` → `04` → `05`.
5. For inference:
   ```bash
   python phase1_genre_classification/scripts/inference.py
   ```

## Phase 1: Genre Classification
This phase involves a 4-model comparison approach to classify texts into 12 canonical genres. The pipeline features per-genre threshold tuning to maximize Macro-F1. We also experimented with fusing topic modeling probabilities into the transformer architecture, though it yielded a negative result.

## Phase 2: Mismatch Detection
This phase focuses on identifying divergences between books and their movie adaptations. It utilizes a Wikipedia scraping pipeline to gather plot summaries and evaluates narrative fidelity using a custom 5-head scoring system with specific weights.

## Key Findings
- **RoBERTa-base** outperforms DeBERTa-v3 despite DeBERTa being architecturally more sophisticated (an FP32 constraint limits its performance).
- **Topic fusion** does NOT improve over vanilla RoBERTa (a negative result).
- **Per-genre threshold tuning** improves the Macro-F1 score by 3-5 points.
- **Mismatch detection** effectively validates on known faithful/divergent pairs, but has a known limitation with thematic inversions (e.g., *The Shining*).

## Requirements
- Python 3.12
- PyTorch 2.x
- transformers>=4.48
- sentence-transformers
- bertopic
- umap-learn
- hdbscan
- iterative-stratification
- rapidfuzz
- vaderSentiment
- spacy
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn

## Training Environment
- **Platform**: Kaggle / Google Colab
- **Hardware**: NVIDIA Tesla T4 GPU (15.8 GB VRAM)

## License
[MIT](LICENSE)
