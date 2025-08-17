## Project: A Comparative Study of Traditional and LLM-based Recommendation Systems on Amazon Review Data

### Candidate numbers: 50450, 46476, 41195, 40781

Single line summary: e-commerce product recommendation based on collaborative filtering and LLM-based techniques over Amazon Reviews 2023 data.

### 1. Introduction: Problem Formulation and Summary of Results (10%) 5

Short section covering big data characteristics of user interaction and product review data, along with limitations of collaborative filtering techniques in extracting more individualised user behaviour data. Brief discussion of the proposed solution, extending ALS with LLM-based recommendations, with potential for more diverse and interpretable outputs. Some statement related to tjhe expected performance, although no concrete results are shown/discussed.

As for improvement, you need to structure the introduction to clearly present the context and problem being addressed, a summary of your solution, along with key results and contributions.

### 2. Solution Methodology (20%) 12

Good summary of existing approaches for the chosen problem. with emphasis on matrix factorisation, hybrid and LLM-based approaches. Instead of an itemise list, you may consider expanding more each of the reference work and pointing to pros/cons of each proposal. There's a good attempt in positioning your work against the literature, in terms of "extending ideas by comparing models". Here, you should clearly identify any gaps in the literature and show how the proposed work fill these gaps.

Solution is based on ALS enriched by LLM-based recommendations. Section 4.2 can be removed as it presents theoretical definition of ALS. You must concentrate on your technical/project decisions instead, such as any specific configurations for ALS regarding the chosen dataset and execution environment. Good point related to repartitioning the data. Here, as improvement, you could provide the reader with a few more details on whether 200 is an optimal value or not. Have you tested other configurations? Still as improvement, you can remove all mentions to kernel crash and emphasise your technical choices and contributions. Kernel crashes can be mentioned as limitations in the Conclusion section. Overall, the entire section on ALS must show pros/cons of this approach as a baseline solution, so you can better justify the use of LLM.

Regarding LLM-based recommendation: perhaps the most important issue is the use of Colab vs GCP and whether the proposed solution can be effectively scaled to distributed environments. There's commendable effort in getting this pipeline fully implemented, even though GCP is not the target architecture. This could be clearly discussed at the introduction, so you set expectations about data size, scalability scenarios/metrics, and related limitations of your solution.

There's a need to better link sections 4.3 and 4.4 and the corresponding code files. Perhaps a guiding question/scenario to be discussed in each section. For instance, providing 200 recommendations can be exaggerated for existing users, compared to 5 for cold-star users. Is that correct? Good set and discussion of evaluation metrics though. It would be nice to clearly state any dependencies of your LLM models from the ALS output.

Section 4.4 brings a specific scenario for recommendation based on the proposed LLM pipeline. Good set of models, with clear discussion of limitations and observed results. Figure 6 is not legible though. However, you may observe this scenario is not necessarily linked to the rest of your work; i.e., it won't necessarily benefit from any big data or distributed approach. it is a good contribution towards recent approaches for recommender systems though.

Overall, the methodology could be structure to respond specific question and using specific methods. Would the overall idea be to prove that ALS is indeed limited and LLMs can bring substantial improvement? A more consistent experimentation flow would improve your contribution and ease understanding.

Also, any scalability metrics that could be assessed here?

### 3. Implementation (30%) 22

Code is provided in several files, both for ALS and LLM modelling and analysis. Make sure to add consistent documentation throughout the code, section headers/numbers, so you can cross-reference in the report.

Spark and GCP usage is overall fine and coherent with what was discussed during the course. Good use of Spark libraries and standard pipeline.

Good effort related to LLM modelling and the design of hybrid and specific pipeline. To ease reproducibility, it would be nice to discuss any technical issues related to the use of such models in GCP, perhaps as limitations of your work. There's good mastery of models and APIs though.

### 4. Choice and Description of Data (10%) 4

No specific section on data description. This information is split between data preprocessing and EDA sections.

Data on Amazon Reviews captured from HF and structured as two main files. Overall, good amount and mixture of variables, comprising categorical and continuous values. Good amount of observations, adherent to a big data scenario. It would be nice to show some samples and discuss any data preprocessing needs.

Good point about class imbalance: you could have elaborated more on any methods used here to circumvent this problem. Good point about missing data too; again, some discussion of any approach to dealing with missing data would improve the text.

You might consider adding frequent word analysis as another subsection and discussing the observed results a bit more. Perhaps removing some words, such as synonyms, and focusing more on adjectives would help your models to be better recommended here.

### 5. Numerical Evaluation (20%) 13

The idea of section 4.5 is overall fine. but you could have provided a bit of context at the beginning. Also, it would be nice to comment on similarities and differences across recommendations and, if possible, against the user profile, so you can discuss which approach performs better and is more reliable.

Discussion of ALS results are overall fine and covers several points related to the number of latent factors and interactions. It would be nice to support each statement with your observed results, so you can reinforce the suitability (or not) of ALS for the given task, and open some room to discuss the suitability of LLMs as well.

There's excellent discussion of LLM results in sections 5.2 and 5.3. As major contributions, the proposed experimentation was able to assess a good set of useful metrics and identify whether an LLM-only approach is feasible or not. Good discussion of the hybrid LLM approach. Here, a natural question would be to expand the number of users and experiment with different ratios of cold-start users. Also, to revise which metric would be the most important one. Regarding LLM-only results, you managed to identify prominent results towards diversity and explainability - although the report could benefit of a few examples.

As improvement, you should clearly state what would be a feasible and reliable pipeline combining all methods, and link back to the introduction/related work to emphasise your contributions to the literature.

Any scalability metrics that could be assessed here?

### 6. Conclusion (5%) 3

Good summary of the proposed solution, main contributions and limitations, and ideas for future work. Here, you could explore a bit more any dependencies between ALS and LLMs that could result in a final pipeline, so as to inform the reader whether ALS is still relevant given all these new models. There's a clear discussion of limitations, especially those related to the chosen models; and a good set of future work ideas. Here, I'm not sure that integrating more data would work as the main bottlenecks are the model itself (token and runtime) and the underlying architecture. I would suggest some MCP deployment of different models over a distributed environment, exploring more diverse data.

### 7. Presentation Quality (5%) 3

Report is well-written and can be improved by repositioning some sections and removing unnecessary sections. Make sure to revise the content to clearly specify test scenaios and guiding questions.

Code needs some working, as there are several files implementing complementary parts of the proposed pipeline. You can revise the ALS implementation and stick to a final file, properly documented. The same applies to the LLM implementation. Make sure to add consistent documentation throughout the code, section headers/numbers, so you can cross-reference in the report and point the reader to key information in the code.

**Final marks: 62**
