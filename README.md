# 🚀 NLP-Solutions-with-Transformers-LLMs-
> End-to-end applications of Transformers and LLMs for emotion detection, social sentiment analysis, and retrieval-augmented generation. Designed for real-world scalability and production pipelines.

---

## 🧠 Project Highlights

### 🧾 1. Emotion Classification with Custom Vanilla Transformer 
- Developed a **multi-label classifier** to predict nuanced emotions across 28 categories.
- Achieved **F1-score: 0.23** and **AUC: 0.85** using a **custom Transformer with sigmoid activation**.
- Implemented **per-label thresholding** and optimized **multi-label loss functions**.

---

### 🛫 2. Sentiment Analysis on Twitter (BERT Fine-Tuning)
- Fine-tuned **BERT-base** on the **Twitter US Airline dataset** for sentiment classification.
- Achieved **81.0% F1-score** across positive, negative, and neutral labels.
- Features **attention visualization**, **token-level classification**, and **training diagnostics**.

---

### 📄 3. GenRAG: Retrieval-Augmented Generation for Document QA
- Built a modular RAG pipeline to convert documents into a queryable knowledge base, enabling accurate, context-aware QA using local LLMs.

-Integrated semantic + keyword retrieval via FAISS and BM25

-Used Gemma-7B-IT for instruction-tuned generation with real context

-Achieved BERTScore F1: 0.8029

-Includes full pipeline: query rewriting, chunking, embedding, prompt augmentation, and LLM response generation

-Runs locally on GPU (e.g., RTX 4090) without LangChain or black-box tools.

---

## 📊 Evaluation Metrics

| Task                        | Metric        | Score   |
|-----------------------------|---------------|---------|
| Emotion Classification      | F1 / AUC      | 0.23 / 0.85 |
| Twitter Sentiment Analysis  | F1-score      | 81.0%   |
| RAG Document QA             | BERTScore F1    | 0.8029      |

---

## 🛠️ Tech Stack

- **Transformers**: BERT, Custom Transformer Architectures  
- **Frameworks**: TensorFlow, PyTorch, HuggingFace Transformers  
- **RAG Stack**: FAISS, BM25, SentenceTransformers, Gemma-7B-IT (local LLM)
- **Utilities**: NLTK, Scikit-learn, Matplotlib, Weights & Biases  

---

## 📁 Repository Structure

```
📦 Applied-NLP-Systems
├── NLP_Go_emotions_text_classification_custom_transformer.ipynb
├── BERT-model-for-Twitter-sentiment-analysis.ipynb
└── LLM-Powered Retrieval-Augmented Generation (RAG) for Contextual Domain Q&A.ipynb
```

---

## 🔍 Applications

- Emotion-aware systems in mental health, customer support, and UX
- Real-time social media sentiment tracking and monitoring
- Generative document Q&A systems for knowledge-based organizations

---

## 🧪 Future Add-ons

- ✅ Streamlit/Gradio Demo Interface (In Progress)
- ✅ Dockerized Inference API for Deployment
- ✅ HuggingFace Model Card for Public Release

---

## 👨‍💻 Author

**Kheer Sagar Patel**  
*M.Tech in CSE (AI & ML), PDPM IIITDM Jabalpur*  
*PhD Offer From IIT Bhilai in AI & ML*

---

## ⭐️ If this repo inspired you...

Don’t forget to ⭐ the repo and follow for more applied AI projects!

---
