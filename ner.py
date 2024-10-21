

from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import nltk
nltk.download('punkt') # 下载分词模型

from config import Config
from typing import List

# 初始化NER模型和分词器
model_path = Config['models']['bert-base-NER']['model_path']
tokenizer_ner = AutoTokenizer.from_pretrained(model_path)
model_ner = AutoModelForTokenClassification.from_pretrained(model_path).cuda()
label_list = model_ner.config.id2label # 提取NER标签映射

# 提取实体
def extract_entities(text):
    """
    对输入文本进行NER，提取实体
    """
    # encode: transfer input text to token ids
    # tokenize: return a List[str]
    tokens = tokenizer_ner.tokenize(tokenizer_ner.decode(tokenizer_ner.encode(text)))
    inputs:List[int] = tokenizer_ner.encode(text, return_tensors="pt").cuda()
    outputs = model_ner(inputs)[0] # shape = (1, seq_len, num_labels)
    predictions = torch.argmax(outputs, dim=2)
    entities = []
    for token, prediction in zip(tokens, predictions[0].cpu().numpy()):
        label = label_list[prediction]
        if label != 'O':
            entities.append((token, label))
    return entities

# 示例用法
if __name__ == "__main__":
    sample_text = "Paris is the capital city of France."
    entities = extract_entities(sample_text)
    print("Extracted Entities:")
    print(entities)

