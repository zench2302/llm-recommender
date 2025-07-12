# 📌 LLM-Based Product Recommender

### Description: 

Developed an end-to-end recommendation system powered by LLMs for Amazon product reviews. The system embeds product metadata and review content using Flan-T5 and MiniLM, then computes similarity via Faiss for Top-5 recommendations.

### Key Technologies:  
- Nous-Hermes-2 (Mistral) for generating user profiles from review history
- Flan-T5 for generating ad-style recommendation reasons  
- BGE (BAAI) and MiniLM embeddings for product and user vectorization  
- FAISS for approximate nearest neighbor (ANN) vector search and candidate retrieval  
- Prompt engineering for review summarization and recommendation reasoning

---

## 🔁 System and Recommendation Pipeline 

<img src="assets/LLM_pipeline.png" width="720">

---
### Prompts Engineering
**(1) User Profiling Prompt (Mistral)**
```text
You are a professional shopping assistant.

Analyze the following user reviews and summarize their preferences.

Reviews:
""" <user reviews> """

Return JSON with:
- "preferred_products"
- "liked_features"
- "dislikes"
- "potential_interests"
```
<details>
<summary>🔧 Full Python Implementation (click to expand)</summary>
  
```python
def generate_user_profile(user_reviews):
    prompt = f"""
You are a professional shopping assistant.

Analyze the following user reviews and summarize their preferences.

Reviews:
\"\"\"{user_reviews[:Config.MAX_REVIEW_LENGTH]}\"\"\"

Return JSON with:
- "preferred_products"
- "liked_features"
- "dislikes"
- "potential_interests"
"""
    inputs = profile_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(profile_model.device)
    outputs = profile_model.generate(
        **inputs,
        max_new_tokens=Config.MAX_NEW_TOKENS,
        temperature=Config.TEMPERATURE,
        top_p=Config.TOP_P,
        repetition_penalty=Config.REPETITION_PENALTY,
        pad_token_id=profile_tokenizer.eos_token_id
    )
    raw_output = profile_tokenizer.decode(outputs[0], skip_special_tokens=True)
    json_str = raw_output[raw_output.find("{"):raw_output.rfind("}")+1]
    return json.loads(json_str)
```
</details>

**(2) Ad-Slogan Prompt summary (Flan-T5)**
```text
You are an expert e-commerce copywriter creating unique, playful ad slogans.

Product:
- Title: ...
- Description: ...
- Rating: ...

User:
- Likes: ...
- Dislikes: ...

Your task:
- Write ONE catchy slogan (≤12 words)
- Avoid repeating product name or brand
- Use playful, emotional, or surprising tone
```

<details>
<summary>🔧 Full Python Implementation (click to expand)</summary>
  
```python
def build_ad_prompt(product_info, user_profile):
    title = product_info.get('title', 'Unknown Product')
    description = " ".join(product_info.get('description', [])) if isinstance(product_info.get('description'), list) else product_info.get('description', '')
    details = product_info.get('details', '')
    avg_rating = product_info.get('average_rating', 0)
    preferred_products = ", ".join(user_profile.get('preferred_products', []))
    liked_features = ", ".join(user_profile.get('liked_features', []))
    dislikes = ", ".join(user_profile.get('dislikes', []))
    potential_interests = ", ".join(user_profile.get('potential_interests', []))

    return f"""
You are an expert e-commerce copywriter creating unique, playful ad slogans.

Product:
- Title: {title}
- Description: {description}
- Details: {details}
- Average rating: {avg_rating}

User:
- Preferred products: {preferred_products}
- Likes: {liked_features}
- Dislikes: {dislikes}
- Interests: {potential_interests}

Your task:
- Write ONE catchy slogan (≤12 words) that excites this user.
- Match the product type (nails, hair, skincare, lashes, tools, etc.).
- Highlight the user’s likes, avoid their dislikes.
- Use playful, emotional, or surprising language.
- Do NOT copy product name, specs, or brand.
- If irrelevant, return only: SKIP.

Output:
"""
```
</details>

💡 Full prompt templates available in [PROMPTS.md](PROMPTS.md)

---

### 🧾 Selected Outputs
<img src="assets/LLM_recitem3.png" width="720">

---

### 📊 Evaluation Snapshot

| Metric                  | Score (non-cold start) |
|-------------------------|------------------------|
| Semantic Match (CosSim) | 71.8%                  |
| Ad Diversity            | 82.6%                  |
| Avg. Product Rating     | 4.31                   |

---

📁 Full report: [ST446_Project.pdf](./4%20Report/ST446_Project.pdf)
