
import torch
import csv
import evaluate
import numpy as np
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
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    AutoModelForSequenceClassification,
)

from peft import (
    TaskType,
    PeftType,
    PromptEncoder,
    PromptEncoderConfig, 
    get_peft_model,
    PromptTuningConfig,
    PromptEmbedding,
    AutoPeftModelForCausalLM,
    AutoPeftModelForSequenceClassification,
)


from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support



def train_p_tuning(model, tokenizer):
    
    
    # define hyperparameters
    num_virtual_tokens=20
    batch_size = 2  
    lr = 3e-2
    num_epochs = 5
    num_epochs = 5
    max_length = 512 - num_virtual_tokens
    
    
    
    # 加载数据集
    dataset_name = "race"

    dataset_path = Config["datasets"][dataset_name]
    ds = load_dataset_from_huggingface(dataset_path,"high")
    
    # coarse-grained preprocessing
    ds, classes, tokenizer = preprocess_race(ds, tokenizer)
    
    
    # fine-grained preprocessing
    # the preprocessed dataset only contains ["input_ids", "attention_mask", "labels"]
    
    
    processed_ds = ds.map(
        lambda examples: preprocess_function_race(examples, max_length=max_length, tokenizer=tokenizer), # 从load.py导入
        batched=True,
        num_proc=1,
        remove_columns=ds['train'].column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on dataset",
    )
    
    train_ds = processed_ds["train"]
    eval_ds = processed_ds["test"]
    

    print("dataset is preprocessed successfully ~~~")
    
    
    train_dataloader = DataLoader(train_ds, shuffle=True, collate_fn=default_data_collator, batch_size=batch_size)
    eval_dataloader = DataLoader(eval_ds, collate_fn=default_data_collator, batch_size=batch_size)
    
    device = Config['device'] 
    tokenizer_path = Config["models"]["bert-base-uncased"]["model_path"]
    
    # Prompt-tuning
    peft_config = PromptEncoderConfig(
        peft_type="P_TUNING",
        task_type= TaskType.SEQ_CLS, 
        num_virtual_tokens=num_virtual_tokens, 
        token_dim=768,
        num_transformer_submodules=1,
        num_attention_heads=12,
        num_layers=1,
        encoder_reparameterization_type="MLP",
        encoder_hidden_size=768,
    )
    
    # Input Shape: (batch_size, total_virtual_tokens)

    # Output Shape: (batch_size, total_virtual_tokens, token_dim)
    
    model = get_peft_model(model, peft_config)
    
    
    # make sure to frozen the base model parameters
    for param in model.base_model.parameters():  
        param.requires_grad = False  
        
    # make sure that the prefix tokens is trainable
    for name, param in model.named_parameters():  
        if 'prefix_encoder' in name:  
            param.requires_grad = True
    
    
    model.print_trainable_parameters()
    



    # optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=lr
    )

    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(train_dataloader) * num_epochs),
    )
    
    device = Config['device']
    model = model.to(device)
    global_step = 0

    # 定义一个列表来存储评估结果  
    evaluation_results = []  
    
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
            
            # evaluate for each 5 batch-steps
            if global_step % 5 == 0 or step == len(train_dataloader) - 1:  
                model.eval()  
                all_preds = []  
                all_labels = []  
                with torch.no_grad():  
                    for val_batch in eval_dataloader:  
                        val_input_ids = val_batch['input_ids'].to(device)  
                        val_attention_mask = val_batch['attention_mask'].to(device)  
                        val_labels = val_batch['labels'].to(device)  
                        val_outputs = model(input_ids=val_input_ids, attention_mask=val_attention_mask)  
                        logits = val_outputs['logits']  
                        preds = torch.argmax(logits, dim=1).cpu().numpy()  
                        labels_cpu = val_labels.cpu().numpy()  
                        all_preds.extend(preds)  
                        all_labels.extend(labels_cpu)  
                # 计算评价指标  
                accuracy = np.mean(np.array(all_preds) == np.array(all_labels))  
                precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')  
                print(f"Step {global_step}, Validation Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")  
                model.train()  

                if step == len(train_dataloader) - 1 and epoch == num_epochs - 1:  
                    evaluation_results.append({  
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1': f1,
                    })  

            global_step+=1

        avg_loss = total_loss / len(train_dataloader)   
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")  
            
    # 保存评估结果到 CSV 文件  
    csv_file_path = 'evaluation_output.csv'  

    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as csv_file:  
        fieldnames = ['peft_method', 'dataset','model_name','accuracy', 'precision','recall','f1']  
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)  

        writer.writeheader()  
        for result in evaluation_results:  
            writer.writerow(result)  

    print(f"Evaluation results saved to {csv_file_path}") 

    # 保存权重
    save_path = Config['save_model_dir']['bert-base-uncased']['p-tuning']['race']
    # torch.save(model.state_dict(), save_path) 
    model.save_pretrained(save_path)
    
    
    
def inference_on_race(save_path, ds:Dataset):
    '''
     inference on race dataset, just a simple test
     
     save_path: 训练好的模型权重, make sure it is a PeftModel
    '''
    ds.column_names
    model = AutoPeftModelForCausalLM.from_pretrained(save_path).to("cuda")
    
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

    
    train_p_tuning(model, tokenizer)
    