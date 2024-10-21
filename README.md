# 基于LoRA和Auto-CoT的双向prompt-tuning微调方法
# BlackPrompt: A Bidirectional and LoRA enhanced Auto-Chain-of-Thought-K-Means Style prompt-tuning system For multiple choice question answering

---

## Architecture Overview
The system integrates bidirectional prompt-tuning, Auto Chain-of-Thought (Auto-CoT), Named Entity Recognition (NER), K-means clustering, and LoRA to enhance multiple-choice question answering performance.

---

## System Components:
1. Input Question:  
  1. a multiple-choice question [context + question + candidate items].
2. Bidirectional Prompt-Tuning:
  - Prefix Prompt Tokens: Add K trainable tokens at the beginning of the model input.
  - Suffix Prompt Tokens: Add K trainable tokens at the end of the model input. The latest 5 tokens represent the classification of the question. (this is done by a few-shot classification model)
  - Normally, the Prefix and Suffix tokens should be initialized during pre-training, however, due to the lack of strong computing power and the intrinsic limitation of the pre-train-initialization, we prepare to initilize the prefix and suffix prompt tokens using the output reasoning steps from the Auto-CoT module.
3. Auto Chain-of-Thought (Auto-CoT):
  - Generates intermediate reasoning steps to an MCQ question.
  - The Auto-CoT generated reasoning steps will be further processed by the NER and K-means module to  retrieve the 5 trainable prompt tokens used for forward prompt-tuning.
4. Named Entity Recognition (NER):
  - Applies NER to recognize all reasoning step strings from the Auto-CoT output.
  - Extracts meaningful entities representing reasoning steps.
  - For example, we finally get 1000 entities, each is represented by a word embedding.
5. K-Means Clustering:
  - Identifies K reasoning steps from the extracted entities.
  - Clusters the word embeddings of all reasoning steps into K clusters using the K-means algorithm.
  - Performs average pooling on each cluster to obtain K pooled vectors.
6. Pooled K Vectors:
  - The pooled vectors serve as the final K trainable forward prompt tokens for the model.
7. LoRA Integration:
  - Incorporates Low-Rank Adaptation (LoRA) into the system to efficiently fine-tune the large language model without updating all parameters.
8. Language Model Processing:
  - The modified input, enriched with bidirectional prompt tokens and optimized via LoRA, is fed into the language model.
  - The model processes the input to generate the answer.
9. Output Answer:
  - The system outputs the answer to the multiple-choice question.

---

## System Sketch
![image](https://github.com/user-attachments/assets/a2b4f441-792d-4520-b918-9be83b69addf)
