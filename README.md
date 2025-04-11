# 🚀 NLP-Solutions-with-Transformers-LLMs-
> End-to-end applications of Transformers and LLMs for emotion detection, social sentiment analysis, and retrieval-augmented generation. Designed for real-world scalability and production pipelines.

---

## 🧠 Project Highlights

### 🧾 1. Emotion Classification with Custom Transformer ([GoEmotions](https://github.com/google-research/goemotions))
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
- Built a modular **RAG pipeline** to transform documents into **queryable knowledge bases**.
- Fine-tuned LLMs for QA generation with **BLEU Score: 29**.
- Pipeline includes:
  - Dynamic **query rewriting**, **chunking**, and **semantic search**
  - Integration with **FAISS**, **LangChain**, and **LLM prompt chaining**

---

## 📊 Evaluation Metrics

| Task                        | Metric        | Score   |
|-----------------------------|---------------|---------|
| Emotion Classification      | F1 / AUC      | 0.23 / 0.85 |
| Twitter Sentiment Analysis  | F1-score      | 81.0%   |
| RAG Document QA             | BLEU Score    | 29      |

---

## 🛠️ Tech Stack

- **Transformers**: BERT, Custom Transformer Architectures  
- **Frameworks**: TensorFlow, PyTorch, HuggingFace Transformers  
- **RAG Stack**: LangChain, FAISS, OpenAI API  
- **Utilities**: NLTK, Scikit-learn, Matplotlib, Weights & Biases  

---

## 📁 Repository Structure

```
📦 Applied-NLP-Systems
├── NLP_Go_emotions_text_classification_custom_transformer.ipynb
├── BERT-model-for-Twitter-sentiment-analysis.ipynb
└── GenRAG.ipynb
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

---

## ⭐️ If this repo inspired you...

Don’t forget to ⭐ the repo and follow for more applied AI projects!

---
