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
    default_data_collator,
    AutoModelForCausalLM, 
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForMultipleChoice,
    get_linear_schedule_with_warmup
)
from peft import (
    TaskType,
    PromptEncoderConfig, 
    get_peft_model, 
    PromptTuningConfig,
    PromptEmbedding,
    AutoPeftModelForCausalLM,
    AutoPeftModelForSequenceClassification,
    # AutoPeftModelForMultipleChoice,
)

from tqdm import tqdm









# 加载数据集

# dataset_path = Config["datasets"]['race']
# ds = load_dataset_from_huggingface(dataset_path,"high")

# classes = [k.replace("_", " ") for k in ds["train"].features["Label"].names]
# ds = ds.map(
#     lambda x: {"text_label": [classes[label] for label in x["Label"]]},
#     batched=True,
#     num_proc=1,
# )
# ds["train"][0]



def train(model):
    # 加载数据集
    dataset_name = "race"

    dataset_path = Config["datasets"][dataset_name]
    ds = load_dataset_from_huggingface(dataset_path,"high")
    
    # coarse-grained preprocessing
    ds, classes, tokenizer = preprocess_race(ds)
    
    Config["classes"][dataset_name] = classes
    
    # fine-grained preprocessing
    # the preprocessed dataset only contains ["input_ids", "attention_mask", "labels"]
    num_virtual_tokens=10
    max_length = 512-num_virtual_tokens
    processed_ds = ds.map(
        lambda examples: preprocess_function_race(examples, max_length=max_length), # 从load.py导入  max_length = 492, 等下要加20个virtual tokens
        batched=True,
        num_proc=1,
        remove_columns=ds['train'].column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on dataset",    
    )   
    
    train_ds = processed_ds["train"]
    eval_ds = processed_ds["test"]
    
    # print("train_ds[0] = ", train_ds[0])

    batch_size = 8

    
    
    # print("train_ds[0]", train_ds[0])
    # print("eval_ds[0]",eval_ds[0])
    
    print("dataset is preprocessed successfully ~~~")
    
    
    # train_dataloader = DataLoader(train_ds, shuffle=True, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True)
    # eval_dataloader = DataLoader(eval_ds, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True)
    train_dataloader = DataLoader(train_ds, shuffle=True, collate_fn=default_data_collator, batch_size=batch_size)
    eval_dataloader = DataLoader(eval_ds, collate_fn=default_data_collator, batch_size=batch_size)
    
    tokenizer_path = Config["models"]["bert-base-uncased"]["model_path"]
    
    # Prompt-tuning
    peft_config = PromptTuningConfig(
        peft_type="PROMPT_TUNING",
        task_type=TaskType.SEQ_CLS, 
        num_virtual_tokens=num_virtual_tokens, 
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
    torch.save(model.state_dict(), Config['save_model_dir']['bert-base-uncased']['prompt-tuning']['race'])    

def inference_on_race(save_path, ds:Dataset):
    '''
     inference on race dataset, just a simple test
     
     save_path: 训练好的模型权重, make sure it is a PeftModel
    '''
    ds.column_names
    model = AutoPeftModelForSequenceClassification.from_pretrained(save_path).to("cuda")
    
    tokenizer_path = Config["models"]["bert-base-uncased"]["model_path"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    device = Config['device']   
    i = 15
    inputs = f"Artical:{ds['article'][i]}\n\nQuestion:{ds['question'][i]}\n\n \
              Options:{ds['options'][i]}\n\nAnswer:"
              
    inputs = tokenizer(inputs, return_tensors="pt")
    # inputs: {"input_ids": tensor([[  101, 10001, 10002,  ...,     0,     0,     0]]), "attention_mask": tensor([[1, 1, 1,  ..., 0, 0, 0]])}
    
    with torch.no_grad():
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # outputs.shape = (batch_size, max_length)
        outputs = model.generate(input_ids=inputs["input_ids"], max_new_tokens=10)
        print(tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True))



if __name__ == '__main__':
    '''
    
    '''
    model = AutoModelForSequenceClassification.from_pretrained(Config["models"]["bert-base-uncased"]["model_path"])
    
    train(model)
    