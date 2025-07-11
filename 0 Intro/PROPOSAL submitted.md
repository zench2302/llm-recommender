## Project proposal form

Please provide the information requested in the following form. Try to provide concise and informative answers.

## Candidate numbers: 40781 , 41195, 46476, 50450

**1. What is your project title?**

A Comparative Study of Traditional and LLM-based Recommendation Systems on Amazon Review Data

**2. What is the problem that you want to solve?**

This project aims to explore and compare the effectiveness of two distinct paradigms in recommendation systems:

-	A traditional collaborative filtering method using Alternating Least Squares (ALS)
-	A modern approach leveraging Large Language Models (LLMs)

By applying both methods to the same dataset and use case, we aim to highlight the strengths and limitations of each, especially in the context of personalized product recommendations.

**3. What big data methodologies and tools do you plan to use?**

**Data Storage & Infrastructure:**
- Google Cloud Storage (GCS) for handling large-scale datasets
- Google Cloud Dataproc + HDFS for distributed data storage and compute

**Data Preprocessing:**
- PySpark DataFrames for filtering, joining, and feature engineering
- Extracting features from structured metadata (brand, category, price, etc.) and unstructured review text

**Modeling Approaches:**
1. Traditional: Alternating Least Squares (ALS) collaborative filtering (using PySpark MLlib)

2. Modern: LLM-based recommendation using two possible scenarios:
- LLM-as-Encoder: LLM generates semantic embeddings of product descriptions, which are then used with cosine similarity for Top-N item matching
- (Optional) Direct Recommendation Prompting: LLM receives user history and candidate item descriptions and outputs top-N recommendations. Due to token limits and scalability constraints, this method is evaluated primarily as a proof-of-concept.

**Evaluation Methods**

Top-N Recommendation Accuracy

- Hit Rate@K: Measures how often the actual purchased item is in the top K recommendations

- Precision@K: Ratio of recommended items in Top K that were actually interacted with

- Recall@K: Ratio of relevant items successfully recommended in Top K


Qualitative Analysis (for LLM):

- Interpretability of Recommendations: Assess if LLM-generated suggestions are coherent and explainable

- Diversity & Novelty: Are LLM recommendations more diverse or serendipitous than ALS?

**4. What dataset will you use? You may assess the suitability of the dataset for distributed computing for big data. Provide information about the dataset (size, features and labels, if applicable) and a URL for the dataset if available. If you intend to generate synthetic data, provide a description of what features you will use, whether your dataset is labelled or not (supervised versus unsupervised learning problem), how will it be generated (any functions, seed values, data ranges etc) and the expected size. When choosing or generating data, please remember to consider any aspects related to data representativeness, quality, bias and other aspects relevant to your project**

We will use the Amazon Reviews 2023 dataset published by the McAuley Lab, available on Hugging Face:
🔗 https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023

Scope: Focused on one category (e.g., Electronics)

Structure:
•	Reviews: user_id, item_id, rating, review_text, timestamp
•	Metadata: item_id, title, brand, price, description, category

Suitability:
•	Large-scale, distributed-friendly structure
•	Contains both user behavior logs and rich item content — ideal for both ALS and LLM approaches

**5. List key references (e.g. research papers) that your project will be based on. Please cite them properly and provide URLs.**

McAuley, J., Pandey, R., & Leskovec, J. (2015).
Inferring Networks of Substitutable and Complementary Products.
KDD '15: Proceedings of the 21th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining.
https://cseweb.ucsd.edu/~jmcauley/pdfs/kdd15b.pdf

He, X., Liao, L., Zhang, H., Nie, L., Hu, X., & Chua, T. S. (2017).
Neural Collaborative Filtering.
WWW '17: Proceedings of the 26th International Conference on World Wide Web.
https://dl.acm.org/doi/10.1145/3038912.3052569

Meng, X., Bradley, J., Yavuz, B., Sparks, E., Venkataraman, S., Liu, D., ... & Zaharia, M. (2016).
MLlib: Machine Learning in Apache Spark.
Journal of Machine Learning Research, 17(34), 1–7.
https://jmlr.org/papers/volume17/15-237/15-237.pdf


**Please indicate whether your project proposal is ready for review (Yes/No):**

Yes

## Feedback & approval (to be provided by the lecturer by Week 11)

[MB - 04/04/2025] The topic is relevant and adherent with a big data scenario. The dataset is well-known, with previous works addressing the same problem (recommendation) but using different methods. This was an example addressed during the course and part of the assginment, so it cannot be taken as a group project. What about considering language models to perform recommendations? You can play with open models and deploy on GCP/Spark. I've attached a list of papers/websites for consideration.

* https://www.amazon.science/publications/recmind-large-language-model-powered-agent-for-recommendation
* https://github.com/WLiK/LLM4Rec-Awesome-Papers
* https://promptengineering.org/using-large-language-models-for-recommendation-systems/
* https://www.amazon.science/publications/language-models-as-recommender-systems-evaluations-and-limitations
* https://dl.acm.org/doi/10.1145/3726871

You may consider all relevant metrics to test your models, including big data-related metrics; for instance, training on single vs multiple machines. Remember to emphasise any other aspects of your big data pipeline where you've put more work. 

**Resubmit by 12 April, 12 pm**

## Feedback & approval (to be provided by the lecturer by Week 11)

[MB - 08/04/2025] The amended proposal reads very well. The idea of comparing traditional and LLM-based recommendations is pertinet, and the dataset is compatible with a big data scenario. You may consider i) looking at more than one category (apart from Electronics) to check whether your methods are able to generalise across categories, and ii) implement the direct recommendation prompt scenario. If you use pre-trained models downloaded into your machines, then no restrictions related to tokens may occur. You may consider all relevant metrics to test your models, including big data-related metrics; for instance, training on single vs multiple machines. Remember to emphasise any other aspects of your big data pipeline where you've put more work. 

**Approved**
 
