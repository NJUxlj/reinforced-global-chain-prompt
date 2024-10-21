
from transformers import Trainer, TrainingArguments  
from transformers import AutoModel, AutoTokenizer
from datasets import load_metric  
import numpy as np

from config import Config




# 加载评估指标  
metric = load_metric("accuracy")  

# 自定义评估函数  
def compute_metrics(eval_pred):  
    logits, labels = eval_pred  
    predictions = np.argmax(logits, axis=-1)  
    return metric.compute(predictions=predictions, references=labels) 



def train(model_name):
    
    if model_name == "qwen":
        model_name = "qwen2"
        Config["model_name"] = "Qwen/Qwen-14B-Chat"
        Config["tokenizer_name"] = "Qwen/Qwen-14B-Chat"
    elif model_name == "llama":
        model_name = "llama3"
        Config["model_name"] = "meta-llama/Llama-2-7b-chat-hf"
        Config["tokenizer_name"] = "meta-llama/Llama-2-7b-chat-hf"
    elif model_name == 'bert':
        model_name = "bert"
        Config["model_name"] = "bert-base-uncased"
        Config["tokenizer_name"] = "bert-base-uncased"
    elif model_name == "gpt4o":
        model_name = "gpt4o"
        Config["model_name"] = "gpt4o"
        
        # Use the openai's fine-tune function through API
        try:
            '''
            '''
        except Exception:
            print("OpenAI fine-tune error captured")
        return 
    else:
        raise Exception("Model name not found, please select in [qwen, llama, bert, gpt4o]")
        
    
    # build model
    model = AutoModel.from_pretrained(Config)
    
    
    # 设置训练参数  
    training_args = TrainingArguments(  
        output_dir=Config["output_dir"],  
        learning_rate=2e-5,  
        per_device_train_batch_size=8,  
        num_train_epochs=3,  
        evaluation_strategy="epoch",  
    )  

    # 初始化Trainer  
    trainer = Trainer(  
        model=model,  
        args=training_args,  
        train_dataset=Config['train_data_path'],  
        eval_dataset=Config['test_data_path'],  
        compute_metrics=compute_metrics,  
    )  
    
    return trainer


def train_many():
    pass


if __name__ == '__main__':
    trainer = train("qwen2")
    
    trainer.train()
    
    
     # 评估模型  
    eval_result = trainer.evaluate()  
    print(f"Evaluation Result: {eval_result}") 