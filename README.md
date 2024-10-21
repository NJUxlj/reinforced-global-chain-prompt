# 基于LoRA和Auto-CoT的双向prompt-tuning微调方法
# BlackPrompt: A Bidirectional and LoRA enhanced Auto-Chain-of-Thought-K-Means Style prompt-tuning system For multiple choice question answering

---

## Background
- 由于受到《Pretrained Prompt-Tuning》在**使用预训练来初始化prompt tokens**后大幅提高prompt-tuning性能的影响，我们希望找到一种方法，它不需要预训练prompt-tokens，而是仅仅通过使用AutoCoT生成对应问题的推理步骤，并且将这些步骤(steps)以某种方式(clustering)转化为tokens加到原始prompt的后面，来使此种方法的MCQ问答性能超越pre-trained prompt-tuning,并且所使用的显存和算力也大幅降低，更加适用于low-resource的教学场景。 


- prompt 分为hard prompt 和 soft prompt
- 我们会把Auto-Cot的steps处理成一中semi-hard prompt： 由于这些token不是训练得到的，因此他们不是连续的continuous tokens， 他们的本质是从推理步骤中直接拿出来的hard prompt


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





## current idea
1. remove the lora part, replace it with a "Attention Router Module"
2. Dual Attention
3. shareable attention
4. 随着项目的进展，我们会在Auto-CoT steps的处理上加上若干细节
  - 我们通过对AutoCoT的结果进行NER，会产生很多steps, 比如说100个。
  - 原始的办法是将这些steps聚成5类，每一类取均值向量，成为5个tokens，再加到prompt后面。
  - 能否计算100个steps和question之间的注意力分数，再使用这个分数来生成一个context vector。把这个vector作为最后的prompt token。




## 实验配置
### 模型
1. qwen2
2. llama3
3. bert-base-uncased
4. GPT4o

### 评价指标
在评估模型在多项选择题回答任务上的性能时，常用的评价指标由以下几种：

1. **准确率（Accuracy）**：这是最常用的指标，尤其在多项选择题任务中。准确率定义为正确预测的数量与总预测数量的比率。公式为：
   $$ \text{Accuracy} = \frac{\text{Number of Correct Predictions}}{\text{Total Number of Predictions}} $$

2. **精确率（Precision）** 和 **召回率（Recall）**：这两个指标通常在多标签分类问题或模型需要在正负样本之间做权衡时使用。对于单选题的任务中，这些指标很少单独使用，但可以用于分析模型对于某一类别的判断能力。

3. **F1得分（F1 Score）**：是精确率和召回率的调和平均。它常用于需要兼顾精确度和覆盖度的场景中。公式为：
   $$ \text{F1 Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} $$

4. **Top-K准确率（Top-K Accuracy）**：在多个选项中，预测答案在前K个最可能的选项中即视为正确。这对于选项数量较多的多选题，或在模型使用时允许有多种答案时，特别有用。

5. **平均精度（Mean Reciprocal Rank，MRR）**：评估系统返回一个排序列表结果的准确性，即正确答案（通常是某一类的概率最大的样本）是否位于前几个中。公式为：
   $$ \text{MRR} = \frac{1}{N} \sum_{i=1}^{N} \frac{1}{\text{rank}_i} $$
其中 $$ \text{rank}_i $$ 表示正确答案在第i次查询的排序中的排名。


参考:
- [Evaluating QA: Metrics, Predictions, and the Null Response](https://qa.fastforwardlabs.com/no%20answer/null%20threshold/bert/distilbert/exact%20match/f1/robust%20predictions/2020/06/09/Evaluating_BERT_on_SQuAD.html)
- [Evaluation metrics for multiple correct answers in QA problem system](https://stackoverflow.com/questions/64112565/evaluation-metrics-for-multiple-correct-answers-in-qa-problem-system)
- [NLP Question Answering Mastery: Evaluation Metrics and Methods](https://gpttutorpro.com/nlp-question-answering-mastery-evaluation-metrics-and-methods-for-question-answering/)


## 实验结果



