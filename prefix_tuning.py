import torch

from config import Config


from load import (
    preprocess_function_race_pt, 
    preprocess_function_race,
    load_dataset_from_huggingface,
    preprocess_race,
)


from torch.utils.data import DataLoader
from datasets import (
    Dataset, 
    load_dataset
)

from transformers import (
    set_seed,
    default_data_collator,
    AutoModelForCausalLM, 
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForMultipleChoice,
    get_linear_schedule_with_warmup
)
from peft import (
    TaskType,
    PeftType,
    PromptEncoderConfig, 
    get_peft_model, 
    get_peft_config,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    PromptTuningConfig,
    PrefixTuningConfig,
    PromptEmbedding,
    AutoPeftModelForCausalLM,
    AutoPeftModelForSequenceClassification,
    # AutoPeftModelForMultipleChoice,
)

from tqdm import tqdm







def train_prefix_tuning(model, tokenizer):
    # 初始化参数  
    model_name = 'bert-base-uncased'  
    num_labels = 4  # ['A', 'B', 'C', 'D']
    prompt_length = 30  
    batch_size = 2  
    num_epochs = 5  
    learning_rate = 5e-5  
    max_length = 512 - prompt_length
    
    
    # 加载数据集
    dataset_name = "race"
    dataset_path = Config["datasets"][dataset_name]
    ds = load_dataset_from_huggingface(dataset_path,"high")
    # coarse-grained preprocessing
    ds, classes, tokenizer = preprocess_race(ds, tokenizer)

    processed_ds = ds.map(
        lambda examples: preprocess_function_race(examples, max_length=max_length, tokenizer=tokenizer), # 从load.py导入  max_length = 492, 等下要加20个virtual tokens
        batched=True,
        num_proc=1,
        remove_columns=ds['train'].column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on dataset",    
    )   
    
    train_ds = processed_ds["train"]
    eval_ds = processed_ds["test"]


    # # 创建数据集和数据加载器  
    # train_dataset = Racedataset(dataset['train'], tokenizer, prompt_length)  
    # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  
    
    train_dataloader = DataLoader(train_ds, shuffle=True, collate_fn=default_data_collator, batch_size=batch_size)
    eval_dataloader = DataLoader(eval_ds, collate_fn=default_data_collator, batch_size=batch_size)

    # 初始化模型  
    device = Config['device'] 
    
    tokenizer_path = Config["models"]["bert-base-uncased"]["model_path"]
    
    # Prompt-tuning
    peft_config = PromptTuningConfig(
        peft_type="PROMPT_TUNING",
        task_type=TaskType.SEQ_CLS, 
        num_virtual_tokens=prompt_length, 
        token_dim=768,  
        num_transformer_submodules=1,
        # In many cases, this is set to 1, 
        # meaning that the prompt tuning will interact with a single submodule, 
        # often the self-attention submodule, to inject the prompt information into the model.
        num_attention_heads=12,
        num_layers=1,
        prompt_tuning_init = "TEXT",
        prompt_tuning_init_text = "Classify the answer of this question among  A, B, C, and D",
        tokenizer_name_or_path = tokenizer_path,
    )
    
    # Input Shape: (batch_size, total_virtual_tokens)

    # Output Shape: (batch_size, total_virtual_tokens, token_dim)
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    

    lr = 3e-2
    num_epochs = 5

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(train_dataloader) * num_epochs),
    )
    
    device = Config['device']
    model = model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for step, batch in enumerate(tqdm(train_dataloader)):
            print(f"Batch labels: {batch['labels']}") 
            batch = {k: v.to(device) for k, v in batch.items()}
            # batch = {"input_ids": tensor([[101, 7592, 2199, 2, ...], [101, 7592, 2199, ...]]), "attention_mask": tensor([[1, 1, 1,  ..., 0, 0, 0], [1, 1, 1, ...]])}
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.detach().float()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        model.eval()
        eval_loss = 0
        eval_preds = []
        for step, batch in enumerate(tqdm(eval_dataloader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            print("batch['input_ids'].shape = ", batch['input_ids'].shape)  
            with torch.no_grad():
                outputs = model(**batch)
            loss = outputs.loss
            eval_loss += loss.detach().float()
            eval_preds.extend(
                tokenizer.batch_decode(torch.argmax(outputs.logits, -1).detach().cpu().numpy(), skip_special_tokens=True)
            )

        eval_epoch_loss = eval_loss / len(eval_dataloader)
        eval_ppl = torch.exp(eval_epoch_loss)
        train_epoch_loss = total_loss / len(train_dataloader)
        train_ppl = torch.exp(train_epoch_loss)
        print(f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=}")


    # 保存权重
    torch.save(model.state_dict(), Config['save_model_dir']['bert-base-uncased']['prefix-tuning']['race']) 







if __name__ == "__main__":
    '''
    
    '''
    model_path = Config["models"]["bert-base-uncased"]["model_path"]
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=4)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print(f"Model's current num_labels: {model.config.num_labels}") 
     
    model_config = model.config
    model_name_or_path = model_config.name_or_path
    print("model_name_or_path = ", model_name_or_path)
    
    if any(k in model_name_or_path for k in ("gpt", "opt", "bloom")):
        padding_side = "left"
    else:
        padding_side = "right"
    
    print("padding_side = ", padding_side)

    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side=padding_side)

    
    train_prefix_tuning(model,tokenizer)