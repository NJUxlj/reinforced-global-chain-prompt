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
        modules_to_save=["classifier"], # 指定了哪些模块或层的权重在微调过程中不会被LoRA修改，并且在保存模型时会被单独保存。
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()





if __name__ == "__main__":
    model =
    train_lora(model)