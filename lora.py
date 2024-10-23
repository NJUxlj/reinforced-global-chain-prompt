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
    
    
    batch_size = 8
    
    args = TrainingArguments(
        peft_model_id,
        remove_unused_columns=False,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-3,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=batch_size,
        fp16=True,
        num_train_epochs=5,
        logging_steps=10,
        load_best_model_at_end=True,
        label_names=["labels"],
    )
    
    trainer = Trainer(
        model,
        args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=image_processor,
        data_collator=collate_fn,
    )
    trainer.train()





if __name__ == "__main__":
    model_path = Config["models"]["bert-base-uncased"]["model_path"]
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    train_lora(model)