from transformers import AutoModelForCausalLM, AutoTokenizer  
import torch
import torch.nn as nn
from config import Config

model_path = Config["models"]["qwen"]["Qwen2.5-0.5B"]['model_path']
tokenizer_cot = AutoTokenizer.from_pretrained(model_path)  
model_cot = AutoModelForCausalLM.from_pretrained(model_path).cuda()  






# Chain-of-Thought
def generate_reasoning_steps(question, max_length=100):  
    """  
    输入问题，生成推理步骤  
    """  
    prompt = f"Question: {question}\nLet's think step by step:"  
    inputs = tokenizer_cot.encode(prompt, return_tensors='pt').cuda()  
    outputs = model_cot.generate(  
        inputs,  
        max_length=max_length,  
        num_return_sequences=1,  
        no_repeat_ngram_size=2,  
        early_stopping=True  
    )  
    reasoning_steps = tokenizer_cot.decode(outputs[0], skip_special_tokens=True)  
    return reasoning_steps.replace(prompt, "").strip()  

# 示例用法  
if __name__ == "__main__":  
    sample_question = "What is the capital of France?"  
    reasoning = generate_reasoning_steps(sample_question)  
    print("Generated Reasoning Steps:")  
    print(reasoning)  