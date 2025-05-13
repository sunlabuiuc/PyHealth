# 📄 Hallucination-Free Patient Summary Dataset (MIMIC-IV Note v2.2)

This dataset was curated as part of our effort to **reproduce and extend** the findings from the paper:  
**[A Data-Centric Approach To Generate Faithful and High Quality Patient Summaries with Large Language Models](https://arxiv.org/abs/2402.15422)**.

Also this dataset is for UIUC MCS CS598 LHO final project. Our student ids are: ishk2 (ishk2@illinois.edu) and jw150 (jw150@illinois.edu)

## 📌 Purpose

The dataset is intended for **fine-tuning large language models** (LLMs) to generate high-quality and **hallucination-free patient summaries** from clinical notes, as described in the original paper. The work focuses on improving the *faithfulness* and *understandability* of summaries for patients discharged from the hospital.

The dataset uses **MIMIC-IV-Note v2.2**, which contains de-identified clinical notes from thousands of hospitalizations. We have followed a similar data preprocessing pipeline as the original authors, with the goal of replicating and building upon their results.

## 🏥 Background (From the Paper)

> Patients often face difficulties in understanding their hospitalizations, while healthcare workers have limited resources to provide explanations. In this work, we investigate the potential of large language models to generate patient summaries based on doctors' notes and study the effect of training data on the faithfulness and quality of the generated summaries.  
>  
> To this end, we release:  
> (i) a rigorous labeling protocol for errors in medical texts, and  
> (ii) a publicly available dataset of annotated hallucinations in 100 doctor-written and 100 generated summaries.  
>  
> We show that fine-tuning on hallucination-free data effectively reduces hallucinations from 2.60 to 1.55 per summary for Llama 2, while preserving relevant information. A similar effect is observed with GPT-4 (0.70 to 0.40). We also demonstrate that GPT-4 outperforms traditional baselines in automatic hallucination detection.

## 🔧 Reproducibility & Preprocessing

The dataset was generated using preprocessing scripts available at the original repository:  
👉 [JW150/CS598-Project – preprocess folder](https://github.com/JW150/CS598-Project/tree/master/preprocess)

Please refer to that link for implementation details, dependencies, and preprocessing logic.

## 🤝 Contribution

This dataset is contributed via pull request to support reproducibility and community experimentation with medical text summarization using LLMs.

If you use this dataset or build upon it, please cite the original paper and acknowledge the source dataset (MIMIC-IV-Note v2.2).
