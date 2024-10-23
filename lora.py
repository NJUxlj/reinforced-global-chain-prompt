import torch
import torch.nn as nn
from config import Config


from datasets import(
    load_dataset,
    Dataset,
    DatasetBuilder
)



from peft import (
    LoraConfig,
    get_peft_model,
)


from transformers import(
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments
)








def train_lora(model):
    
    config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["query", "value"],
        lora_dropout=0.1,
        bias="none",
        modules_to_save=["classifier"], # 除了LoRA层以外, 还有哪些模块需要进行训练和保存
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()





if __name__ == "__main__":
    model_path = Config["models"]["bert-base-uncased"]["model_path"]
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    train_lora(model)