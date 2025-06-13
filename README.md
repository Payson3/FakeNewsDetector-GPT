# 📰 Fake News Detector using RoBERTa

A Natural Language Processing (NLP) project that detects fake news using a fine-tuned RoBERTa model. This project leverages Hugging Face Transformers, PyTorch, and real-world news datasets to classify articles as **real** or **fake**.

## 🚀 Project Overview

This project demonstrates how to:
- Preprocess and clean textual news data
- Fine-tune a pre-trained RoBERTa model on a binary classification task
- Evaluate model performance on real vs. fake news headlines
- Deploy the model locally or integrate with automation tools like `n8n`

---

## 📁 Dataset

Data sourced from [Kaggle Fake News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset), containing:
- `kaggle_fake.csv`: Fake news articles
- `kaggle_real.csv`: Real news articles

Each file includes columns like:
- `title`: Headline of the article
- `text`: Full article content
- `subject`: Topic or category
- `date`: Publication date

---

## 🧠 Model

- **Model Type**: `roberta-base` (fine-tuned)
- **Task**: Binary Classification (Fake = 1, Real = 0)
- **Frameworks Used**:
  - Hugging Face Transformers
  - PyTorch
  - scikit-learn
  - pandas

---

## 🛠️ How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/fake-news-detector.git
cd fake-news-detector
