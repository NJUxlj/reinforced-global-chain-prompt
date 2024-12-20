# 基于LoRA和Auto-CoT的双向prompt-tuning微调方法
# BlackPrompt: A Bidirectional and LoRA enhanced Auto-Chain-of-Thought-K-Means Style prompt-tuning system For multiple choice question answering

---

## Background
- 由于受到《Pretrained Prompt-Tuning》在**使用预训练来初始化prompt tokens**后大幅提高prompt-tuning性能的影响，我们希望找到一种方法，它不需要预训练prompt-tokens，而是仅仅通过使用AutoCoT生成对应问题的推理步骤，并且将这些步骤(steps)以某种方式(clustering)转化为tokens加到原始prompt的后面，来使此种方法的MCQ问答性能超越pre-trained prompt-tuning,并且所使用的显存和算力也大幅降低，更加适用于low-resource的教学场景。 


- prompt 分为hard prompt 和 soft prompt
- 我们会把Auto-Cot的steps处理成一种称为 `semi-hard prompt` 的东西： 由于这些token不是训练得到的，因此他们不是连续的continuous tokens， 他们的本质是从推理步骤中直接拿出来的hard tokens, 再对这些hard tokens取平均。
  
- 但是从过往的经验看，soft-prompt的性能往往超越hard-prompt, 因此我们希望在此基础上prompt-tokens的部分进行微调，目的是在hard-prompt的基础上增加连续性。



---

## Architecture Overview
The system integrates bidirectional prompt-tuning, Auto Chain-of-Thought (Auto-CoT), Named Entity Recognition (NER), K-means clustering, and LoRA to enhance multiple-choice question answering performance.

---

## Contributions:
We propose a novel bidirectional prompt tuning method that injects a pair of jointly aligned prefix and suffix embeddings into the input layer of the model to from a prompt template of `**[Prefix;Input;Suffix]**`. Where the prefix is initialized using the topic labels from the training set and the suffix are initialized using a global reasoning chain is marged from $K$ local chains using a **Chain Aggregator** . This new method can be comparable or surpass all of the prompt-based baselines (\citealt{lester2021powerscaleparameterefficientprompt}; \citealt{li2021prefix}; \citealt{liu2024gpt}; \citealt{liu2021ptuningv2}) on the Bert model series, leading by a largest margin of 8.1 average F1 points without relying on pre-trained soft tokens \citep{gu2021ppt}.

We also introduce a new reparameterization encoder called **BAP-Encoder** that can leverage a pair of Bi-LSTMs to bidirectionally encode the prefix and suffix separately and use a cross-attention to jointly align the prefix and suffix to mitigate the deviation and inconsistency toward the task content.

We demonstrate that BaasPrompt outperforms existing single-layer prompt-based methods (\citealt{lester2021powerscaleparameterefficientprompt}; \citealt{liu2024gpt}) on all the experimented MCQ datasets, with a largest gap of 32.1-points of precision and 29-points of F1 on the SciQ, and can also be comparable to or even execeed the deep prompting method like prefix-tuning \citep{li2021prefix} and p-tuning v2 \citep{liu2021ptuningv2}.

---



## Baas-Prompt





---

## System Sketch
![image](https://github.com/user-attachments/assets/a2b4f441-792d-4520-b918-9be83b69addf)





## current idea
1. remove the lora part, replace it with a "Attention Router Module"
2. Dual Attention
3. shareable attention


以下是5种可以将这10个任务特定的双向Prompt Embedding整合的方法，以防止灾难性遗忘，并在大部分MCQ任务（如RACE、medQA、SQuAD）上超越之前的高效微调方法。

1. **注意力加权的上下文向量整合**：计算每个Embedding与当前问题之间的注意力分数，利用这些分数对Embeddings进行加权求和，生成一个上下文向量，作为最后的Prompt Token。这种方法可以让模型关注与当前问题最相关的知识，减少灾难性遗忘的影响。

2. **任务特定Embedding的选择与融合**：为每个任务（数据集）训练一组独立的双向Prompt Embedding，例如10个Embeddings。在推理时，使用注意力机制计算当前问题与每个任务Embedding的相关性，选择Top-3最重要的Embeddings进行融合。这有助于模型在不同任务之间共享信息，同时保持任务特定性。

3. **硬注意力任务路由**：采用硬注意力机制，将不同任务的Embeddings路由到对应的任务处理中，类似于“通过对任务的硬注意力克服灾难性遗忘”的方法1。通过明确的任务路由，可以减少不同任务之间的干扰。

4. **基于注意力的Prompt融合网络**：构建一个小型的网络，学习如何根据当前问题动态地组合这10个Embeddings。该网络使用注意力机制，对Embeddings进行加权组合，生成最终的Prompt Embedding。这种方法能够自适应地融合任务之间的知识。

5. **连续学习与正则化**：在训练过程中，加入正则化项，限制新任务训练时对已有Embeddings的更新幅度，防止模型过度调整，从而避免灾难性遗忘。这与连续学习领域中的方法类似，可以参考相关研究2。

6. **Orthogonal Embedding Decomposition**: Decompose each task-specific embedding into orthogonal components to minimize interference. Combine them into a unified prompt embedding that preserves unique task features while reducing overlap.

7. **Dynamic Memory Networks with Prompt Embeddings**: Implement a dynamic memory network that stores each prompt embedding in memory slots. Use a gating mechanism to read and write to these slots based on the input question.

8. **Attention over Attention Mechanism**: Introduce a second-level attention mechanism that not only attends over the prompt embeddings but also considers the attention distributions themselves, enhancing the selection process.

9. **Semantic Clustering of Prompt Embeddings**: Cluster prompt embeddings based on semantic similarity and create cluster representations. Use these representations to guide the integration of embeddings for related tasks.


## 实验配置
### 模型
1. GPT2, GPT2-Medium
3. bert-base, bert-large, roberta
4. GPT-4o (only for the Auto-CoT)


### 评价指标
在评估模型在多项选择题回答任务上的性能时，常用的评价指标由以下几种：

1. **准确率（Accuracy）**：这是最常用的指标，尤其在多项选择题任务中。准确率定义为正确预测的数量与总预测数量的比率。公式为：
   $$ \text{Accuracy} = \frac{\text{Number of Correct Predictions}}{\text{Total Number of Predictions}} $$

2. **精确率（Precision）** 和 **召回率（Recall）**：这两个指标通常在多标签分类问题或模型需要在正负样本之间做权衡时使用。对于单选题的任务中，这些指标很少单独使用，但可以用于分析模型对于某一类别的判断能力。

3. **F1得分（F1 Score）**：是精确率和召回率的调和平均。它常用于需要兼顾精确度和覆盖度的场景中。公式为：
   $$ \text{F1 Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} $$




参考:
- [Evaluating QA: Metrics, Predictions, and the Null Response](https://qa.fastforwardlabs.com/no%20answer/null%20threshold/bert/distilbert/exact%20match/f1/robust%20predictions/2020/06/09/Evaluating_BERT_on_SQuAD.html)
- [Evaluation metrics for multiple correct answers in QA problem system](https://stackoverflow.com/questions/64112565/evaluation-metrics-for-multiple-correct-answers-in-qa-problem-system)
- [NLP Question Answering Mastery: Evaluation Metrics and Methods](https://gpttutorpro.com/nlp-question-answering-mastery-evaluation-metrics-and-methods-for-question-answering/)

## 生成 `requirements.txt` 文件
```shell
pipreqs . --force
```

## 推送项目到仓库
```shell
git add .
# 先将大文件移出暂存库
git rm -rf --cached save data ... 文件夹名
# 检查暂存库的内容
git status
```

- 然后在项目目录中创建一个.gitignore 文件，内容如下：

```text
\save\
\data\
```

```shell
git add .gitignore
git commit -m 'add .gitignore'
git push
```

- 如果还不行，需要下额外的包：

```shell
pip install git-filter-repo

# 使用 git filter-repo 移除大文件  
git filter-repo --strip-blobs-bigger-than 20M  

git remote add origin <远端仓库地址>
git push origin --force  

# 如果没有同步远端，则：
git push --set-upstream origin main --force
```

## 运行项目
main.py

## 训练流程
![image](./image/Snipaste_2024-10-21_18-12-23.png)

![image](./image/Snipaste_2024-10-21_18-12-33.png)

![image](./image/Snipaste_2024-10-21_18-11-39.png)

## 实验结果


| Method       | Dataset |   Model  | Accuracy | Precision | Recall | F1 Score |  
|--------------|---------|----------|----------|-----------|--------|----------|  
| LoRA         | RACE & MedQA & SQuAD   |   bert-base-uncased   | 0.86     | 0.85      | 0.84   | 0.85   |  
|              |         |   qwen2.5-0.5B    | 0.78     | 0.77      | 0.76   | 0.77     |  
|              |         |   llama-3.2-1B       | 0.82     | 0.81      | 0.80   | 0.81     |  
|              |         |   GPT4o       | 0.82     | 0.81      | 0.80   | 0.81     |
| AdaLoRA      |   RACE & MedQA & SQuAD      |   bert-base-uncased | 0.88     | 0.87 | 0.86   | 0.87  |  
|             |    |      qwen2.5-0.5B     | 0.80     | 0.79      | 0.78   | 0.79     |  
|             |    |     llama-3.2-1B     | 0.85     | 0.84      | 0.83   | 0.84     |  
| DoRA         | RACE & MedQA & SQuAD    |          | 0.85     | 0.84      | 0.83   | 0.84     |  
|          |    |          | 0.77     | 0.76      | 0.75   | 0.76     |  
|          |    |          | 0.81     | 0.80      | 0.79   | 0.80     |  
| X-LoRA  | RACE & MedQA & SQuAD   |          | 0.85     | 0.84      | 0.83   | 0.84     |
|          |    |          | 0.77     | 0.76      | 0.75   | 0.76     |
| O-LoRA  | RACE & MedQA & SQuAD   |          | 0.85     | 0.84      | 0.83   | 0.84     |
|          |    |          | 0.77     | 0.76      | 0.75   | 0.76     |
| AM-LoRA | RACE & MedQA & SQuAD   |          | 0.85     | 0.84      | 0.83   | 0.84     |
|          |    |          | 0.77     | 0.76      | 0.75   | 0.76     |
| Prompt-Tuning | RACE & MedQA & SQuAD   |          | 0.83     | 0.82      | 0.81   | 0.82     |  
|      |      |          | 0.75     | 0.74      | 0.73   | 0.74     |  
|      |      |          | 0.79     | 0.78      | 0.77   | 0.78     |  
| Prefix-Tuning | RACE & MedQA & SQuAD   |          | 0.84     | 0.83      | 0.82   | 0.83     |  
|      |    |          | 0.76     | 0.75      | 0.74   | 0.75     |
|      |    |          | 0.81     | 0.80      | 0.79   | 0.80     |
| P-Tuning     | RACE & MedQA & SQuAD    |          | 0.84     | 0.83      | 0.82   | 0.83     |  
|      |    |          | 0.76     | 0.75      | 0.74   | 0.75     |  
|      |    |          | 0.80     | 0.79      | 0.78   | 0.79     |
| P-Tuning v2  | RACE & MedQA & SQuAD    |          | 0.85     | 0.84      | 0.83   | 0.84     |
|      |         |          | 0.77     | 0.76      | 0.75   | 0.76     |
|      |         |          | 0.81     | 0.80      | 0.79   | 0.80     |
| BlackPrompt  | RACE & MedQA & SQuAD    |          | 0.85     | 0.84      | 0.83   | 0.84     |
|            |        |          | 0.77     | 0.76      | 0.75   | 0.76     |
|            |          |          | 0.81     | 0.80      | 0.79   | 0.80     |






