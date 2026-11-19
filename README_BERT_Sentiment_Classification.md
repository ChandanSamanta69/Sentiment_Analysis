# BERT-Based 3-Class Sentiment Classification

**Course:** Natural Language Processing Laboratory (PE-CSEL701AIML) — Assignment 10  
**Task:** Classify customer reviews as Negative, Neutral, or Positive.

## Project Overview

This project implements a three-class sentiment analysis system using a fine-tuned pre-trained **BERT (`bert-base-uncased`)** model. It demonstrates dataset preparation, text preprocessing, WordPiece tokenization, transformer fine-tuning, model evaluation, and comparison with a traditional machine-learning baseline.

The dataset is an IMDb 3-label sentiment corpus. E-commerce customer reviews are used as illustrative examples for the sentiment-analysis task.

## Objectives

- Prepare and explore a labelled sentiment dataset.
- Apply BERT WordPiece tokenization and construct attention masks.
- Fine-tune BERT for three sentiment classes.
- Evaluate the model using accuracy, precision, recall, F1-score, and a confusion matrix.
- Compare BERT with a TF-IDF and Logistic Regression baseline.

## Dataset

- **Source:** [IMDb Balanced Sentiment Dataset — 3 Labels](https://www.kaggle.com/datasets/hasanyiitakbulut/imdb-balanced-sentiment-dataset-3-labels)
- **KaggleHub ID:** `hasanyiitakbulut/imdb-balanced-sentiment-dataset-3-labels`
- **Total samples:** 4,374 reviews
- **Labels:** Negative (0), Neutral (1), Positive (2)
- **Class distribution:** Negative — 1,744; Neutral — 1,209; Positive — 1,421
- **Split:** 80% training (3,499) and 20% test (875), stratified with random seed 42.

Preprocessing removes HTML tags and normalizes extra whitespace while preserving punctuation and wording that may carry sentiment.

## Model and Methodology

The project uses Hugging Face Transformers and `BertForSequenceClassification` with the `bert-base-uncased` backbone (12 transformer layers, 768 hidden dimensions, 12 attention heads).

| Configuration | Value |
|---|---|
| Maximum sequence length | 128 tokens |
| Batch size | 16 |
| Training epochs | 4 |
| Optimizer | AdamW |
| Learning rate | 2 × 10⁻⁵ |
| Weight decay | 0.01 |
| Learning-rate schedule | Linear warmup (10% of steps) |
| Random seed | 42 |
| Classes | 3 |

BERT input sequences use `[CLS]` and `[SEP]` special tokens, WordPiece tokenization, padding, and attention masks. A classification head predicts one of the three sentiment labels.

## Results

### BERT — held-out test set (875 reviews)

| Metric | Score |
|---|---:|
| Accuracy | **93.61%** |
| Macro Precision | 93.41% |
| Macro Recall | 93.46% |
| Macro F1-score | **0.9343** |
| Weighted F1-score | 0.9361 |

### Per-class F1-score

| Sentiment | F1-score |
|---|---:|
| Negative | 94.40% |
| Neutral | 91.36% |
| Positive | 94.53% |

### Baseline comparison

| Model | Accuracy |
|---|---:|
| TF-IDF + Logistic Regression | 79.22% |
| Fine-tuned BERT | **93.61%** |

BERT achieved a 14.39 percentage-point accuracy improvement over the reported baseline. Neutral reviews were comparatively more difficult to classify, reflecting the ambiguity of mixed or balanced opinions.

## Project Structure

```text
Assignment_10_BERT_Sentiment_Classification/
├── Assignment_10_BERT_Sentiment_Classification.ipynb
├── Assignment_10_Project_Report_old.pdf
├── Assignment_10_UML_old.pdf
└── README.md
```

## Installation and Execution

Install the main dependencies:

```bash
pip install torch transformers scikit-learn pandas numpy matplotlib seaborn kagglehub
```

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open `Assignment_10_BERT_Sentiment_Classification.ipynb` and run the cells in order. The notebook downloads the dataset using KaggleHub; configure Kaggle access if prompted. A CUDA-compatible GPU is recommended for faster training. The notebook uses CUDA when available and otherwise falls back to CPU.

## Limitations

- The experiment uses an IMDb sentiment dataset; e-commerce reviews are illustrative application examples.
- Performance may vary with the dataset version, environment, and training conditions.
- The reported results apply to the held-out dataset split and do not guarantee performance on other domains or languages.

## References

- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers)
- [PyTorch Documentation](https://pytorch.org/docs/stable/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)

---
**Domain:** Natural Language Processing | Deep Learning | Sentiment Analysis
