# 基于LoRA和Auto-CoT的双向prompt-tuning微调方法
# BlackPrompt: A Bidirectional and LoRA enhanced Auto-Chain-of-Thought-K-Means Style prompt-tuning system For multiple choice question answering

---

## Background
- 由于受到《Pretrained Prompt-Tuning》在**使用预训练来初始化prompt tokens**后大幅提高prompt-tuning性能的影响，我们希望找到一种方法，它不需要预训练prompt-tokens，而是仅仅通过使用AutoCoT生成对应问题的推理步骤，并且将这些步骤(steps)以某种方式(clustering)转化为tokens加到原始prompt的后面，来使此种方法的MCQ问答性能超越pre-trained prompt-tuning,并且所使用的显存和算力也大幅降低，更加适用于low-resource的教学场景。 


- prompt 分为hard prompt 和 soft prompt
- 我们会把Auto-Cot的steps处理成一种称为 `semi-hard prompt` 的东西： 由于这些token不是训练得到的，因此他们不是连续的continuous tokens， 他们的本质是从推理步骤中直接拿出来的hard tokens, 再对这些hard tokens取平均。
  
- 但是从过往的经验看，soft-prompt的性能往往超越hard-prompt, 因此我们希望在此基础上prompt-tokens的部分进行微调，目的是在hard-prompt的基础上增加连续性。


---

## Contributions:
We propose a novel bidirectional prompt tuning method that injects a pair of jointly aligned prefix and suffix embeddings into the input layer of the model to from a prompt template of **`[Prefix;Input;Suffix]`**. Where the prefix is initialized using the topic labels from the training set and the suffix are initialized using a global reasoning chain is marged from $K$ local chains using a **`Chain Aggregator`** . This new method can be comparable or surpass all of the prompt-based baselines (Lester et al. 2021;
Li and Liang 2021; Liu et al. 2024; Liu et al. 2021) on the Bert model series, leading by a largest margin of 8.1 average F1 points without relying on pre-trained soft tokens (Gu et al., 2021).

We also introduce a new reparameterization encoder called **`BAP-Encoder`** that can leverage a pair of Bi-LSTMs to bidirectionally encode the prefix and suffix separately and use a cross-attention to jointly align the prefix and suffix to mitigate the deviation and inconsistency toward the task content.

We demonstrate that BaasPrompt outperforms existing single-layer prompt-based methods (Lester
et al. 2021; Liu et al. 2024) on all the experimented MCQ datasets, with a largest gap of 32.1-points of precision and 29-points of F1 on the SciQ, and can also be comparable to or even execeed the deep prompting method like prefix-tuning (Li and Liang, 2021) and p-tuning v2 (Liu et al., 2021).

---


## System Overview
![image](./image/BaasPrompt.png)


---

## 实验配置
### 模型
1. GPT2, GPT2-Medium
3. bert-base, bert-large, roberta
4. GPT-4o (only for the Auto-CoT)


### 评价指标
Accuracy, Precision, Recall, F1 Score





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






