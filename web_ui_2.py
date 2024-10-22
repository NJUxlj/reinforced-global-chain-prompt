import gradio as gr  
import torch  
import time  
import threading  
import numpy as np  
import matplotlib.pyplot as plt  
from typing import List

# 可用的模型、微调方法和数据集  
models = ["bert-base-uncased", "roberta-base", "gpt-2"]  
finetune_methods = [  
    "prompt-tuning",  
    "p-tuning",  
    "lora",  
    "prefix-tuning",  
    "bidirectional-prompt-tuning",  
    "AdaLoRA",  
    "DoRA",  
    "VeRA",  
    "AM-LoRA",  
    "O-LoRA"  
]  
datasets = ["Race", "SciQ", "MultiRC", "ARC", "MedQA", "LogiQA"]  

# 微调方法对应的参数  
method_params = {  
    "prompt-tuning": {  
        "learning_rate": 1e-3,  
        "num_prompt_tokens": 10  
    },  
    "p-tuning": {  
        "learning_rate": 1e-3,  
        "num_virtual_tokens": 20  
    },  
    "lora": {  
        "task_type": "classification",  
        "r": 8,  
        "lora_alpha": 32,  
        "target_modules": "all",  
        "lora_dropout": 0.1,  
        "bias": "none"  
    },  
    "prefix-tuning": {  
        "learning_rate": 1e-4,  
        "num_prefix_tokens": 30  
    },  
    "bidirectional-prompt-tuning": {  
        "learning_rate": 1e-4,  
        "num_prefix_tokens": 10,  
        "num_suffix_tokens": 10  
    },  
    "AdaLoRA": {  
        "target_r": 4,  
        "init_r": 12,  
        "beta1": 0.85,  
        "beta2": 0.85,  
        "tinit": 200,  
        "tfinal": 1000,  
        "deltaT": 10  
    },  
    "DoRA": {  
        "dora_rank": 4,  
        "learning_rate": 1e-4  
    },  
    "VeRA": {  
        "vera_rank": 4,  
        "initializer_range": 0.02  
    },  
    "AM-LoRA": {  
        "r": 8,  
        "lora_alpha": 16,  
        "attention_module": "MHA"  
    },  
    "O-LoRA": {  
        "orthogonal_loss_weight": 0.1,  
        "learning_rate": 1e-4  
    }  
}  

# 用于保存训练Loss的全局变量  
training_loss = []  
stop_training = False  # 用于控制训练线程  

def get_method_params(method):  
    '''
    get all hyperparams of the given fine-tuning method
    '''
    params = method_params.get(method, {})  
    return params  

def update_params_ui(method):  
    '''
     return param_rows: List[gr.Row], 用于动态更新参数输入框
    '''
    params = get_method_params(method)  
    ui_elements = []  
    for param_name, default_value in params.items():  
        ui_elements.append(gr.Number(label=param_name, value=default_value))  
    # 将参数输入框按每行三个进行布局  
    num_params = len(ui_elements)  
    param_rows = []  
    for i in range(0, num_params, 3):  
        row = ui_elements[i:i+3]  
        param_rows.append(gr.Row(row))  
    return param_rows  

def train_model(model_name, dataset_name, method, params):  
    global training_loss, stop_training  
    training_loss = []  
    stop_training = False  
    num_epochs = 10  
    # 模拟训练过程  
    for epoch in range(num_epochs):  
        if stop_training:  
            break  
        loss = np.exp(-epoch / 5) + np.random.rand() * 0.02  
        training_loss.append(loss)  
        time.sleep(0.5)  # 模拟训练时间  
    # 模拟评估结果  
    metrics = {  
        "Accuracy": round(np.random.uniform(0.7, 1.0), 4),  
        "Precision": round(np.random.uniform(0.7, 1.0), 4),  
        "Recall": round(np.random.uniform(0.7, 1.0), 4),  
        "F1": round(np.random.uniform(0.7, 1.0), 4)  
    }  
    return metrics  

def plot_training_loss(losses):  
    plt.figure(figsize=(8, 4))  
    plt.plot(losses, label="Training Loss")  
    plt.xlabel("Epoch")  
    plt.ylabel("Loss")  
    plt.legend()  
    plt.tight_layout()  
    plt.savefig("image/training_loss.png")  
    plt.close()  
    return "image/training_loss.png"  

def start_training(model_name, dataset_name, method, *param_values):  
    # 将参数值转换为字典  
    params_keys = list(get_method_params(method).keys())  
    params = dict(zip(params_keys, param_values))  
    metrics = train_model(model_name, dataset_name, method, params)  
    return metrics  

def stop_training_fn():  
    global stop_training  
    stop_training = True  
    return "训练已停止。"  

def save_model_fn():  
    # 模拟保存模型  
    time.sleep(1)  
    return "模型已保存到本地。"  

def main():  
    with gr.Blocks() as demo:  
        gr.Markdown("# 微调模型可视化Demo")  
        with gr.Row():  
            model_choice = gr.Dropdown(choices=models, label="选择模型", value=models[0])  
            method_choice = gr.Dropdown(choices=finetune_methods, label="选择微调方法", value=finetune_methods[0])  
            dataset_choice = gr.Dropdown(choices=datasets, label="选择数据集", value=datasets[0])  
        
        # we uses it to store all the parameter boxes
        param_ui_container = gr.Column()  

        # 动态更新参数输入框  
        def update_ui(method):  
            param_rows = update_params_ui(method)  
            return gr.Column(param_rows)  
        method_choice.change(fn=update_ui, inputs=method_choice, outputs=param_ui_container)  

        # 初始化参数输入框  
        param_rows = update_params_ui(finetune_methods[0])  
        param_ui_container.children = param_rows  

        with gr.Row():  
            train_button = gr.Button("开始训练")  
            stop_button = gr.Button("停止训练")  
            save_button = gr.Button("保存模型")  

        with gr.Row():  
            loss_plot = gr.Image(label="训练Loss曲线")  
            metrics_output = gr.Json(label="评估指标")  

        # 训练按钮点击事件  
        def on_train_click(model_name, dataset_name, method, *params):  
            global training_loss  
            metrics = {}  

            # 在后台线程中运行训练，以避免阻塞UI  
            def training_thread():  
                metrics.update(start_training(model_name, dataset_name, method, *params))  
            thread = threading.Thread(target=training_thread)  
            thread.start()  

            while thread.is_alive():  
                time.sleep(1)  
                if training_loss:  
                    loss_plot.update(plot_training_loss(training_loss))  

            loss_plot.update(plot_training_loss(training_loss))  
            metrics_output.update(metrics)  

        input_params = [model_choice, dataset_choice, method_choice]  

        # 由于参数输入框数量不定，需要动态处理  
        def get_training_inputs():  
            params_keys = list(get_method_params(method_choice.value).keys())  
            param_inputs = []  
            for param in param_ui_container.children:  
                for element in param.children:  
                    param_inputs.append(element)  
            return input_params + param_inputs  

        train_button.click(  
            fn=on_train_click,  
            inputs=get_training_inputs(),  
            outputs=[loss_plot, metrics_output]  
        )  

        # 停止训练按钮  
        stop_button.click(fn=stop_training_fn, outputs=None)  

        # 保存模型按钮  
        save_button.click(fn=save_model_fn, outputs=None)  

    demo.launch(share=True)  

if __name__ == "__main__":  
    main()