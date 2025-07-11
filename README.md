# 🧠 LLM-Based Product Recommendation Engine

This project is an end-to-end recommendation system powered by LLMs and vector embeddings, built using Amazon product reviews. It applies prompt-based review summarization, semantic embeddings, and vector search to generate high-quality, content-aware Top-N product recommendations.

---

## 📌 Project Overview

**Goal:**  
Recommend relevant products based on product descriptions, brand, rating, and user reviews — without using user IDs or past behavior.

**Pipeline Summary:**  
1. Input: Product metadata + user reviews  
2. Prompt-based summarization (Flan-T5)  
3. Semantic embedding (MiniLM)  
4. Faiss vector search for similarity  
5. Top-N recommendation output

---

## 🛠️ Technologies Used

- 📚 Flan-T5 for text summarization  
- 🔎 MiniLM embeddings (Hugging Face)  
- 🧠 Faiss for approximate nearest neighbor (ANN) vector search  
- 🐍 Python (Scikit-learn, Pandas, NumPy)  
- 🧪 Streamlit prototype interface (optional)  
- 📝 Prompt engineering for attribute-aware generation

---

## 💡 Example Use Case

Input:  
> Product: Facial Cleanser  
> Brand: ABC Skincare  
> Description: "Gentle daily cleanser with Vitamin C..."  
> Reviews: ~50 human-written reviews

Output:  
> Top 5 similar products with structured summaries and similarity scores.

---

## 📂 Structure

```
llm-recommender/
├── data/                 # (example or mock data)
├── prompts/              # prompt templates for summarization
├── embedding/            # MiniLM vector creation
├── search/               # Faiss search and ranking
├── app/                  # (optional Streamlit interface)
├── README.md
```

---

## 🚀 Status

- ✅ Completed pipeline with working example  
- ⚙️ Future: Integrate into RAG systems or customer-facing recommender APIs

---

## 📫 Contact

For questions or collaboration:  
[github.com/zench2302](https://github.com/zench2302)  
[linkedin.com/in/jia-jia-7a73359a](https://linkedin.com/in/jia-jia-7a73359a)
