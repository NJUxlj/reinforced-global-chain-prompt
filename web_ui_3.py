import gradio as gr  
import torch  
import time  
import threading  
import numpy as np  
import matplotlib.pyplot as plt  

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
        "num_prompt_tokens_front": 10,  
        "num_prompt_tokens_back": 10  
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
    plt.savefig("training_loss.png")  
    plt.close()  
    return "training_loss.png"  

def start_training(model_name, dataset_name, method, params):  
    # 将参数值转换为字典  
    params_keys = list(method_params[method].keys())  
    params = dict(zip(params_keys, params))  
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

        # 创建所有可能的参数输入组件，并存储在列表中  
        param_components = []  # 所有参数输入组件  
        param_component_methods = []  # 每个组件对应的微调方法  

        current_method = method_choice.value  

        with gr.Column() as param_ui_container:  
            # param_components
            for method in method_params.keys():  
                params = method_params[method]  
                inputs = []  
                for param_name, default_value in params.items():  
                    # 设置组件的初始可见性  
                    visible = True if method == current_method else False  
                    # 选择合适的输入组件，这里使用Textbox，您可以根据需要选择合适的组件类型  
                    input_component = gr.Textbox(label=param_name, value=str(default_value), visible=visible)  
                    inputs.append(input_component)  
                    
                    param_components.append(input_component)  
                    param_component_methods.append(method)  
                #param_inputs[method] = inputs  # 如需保留每个方法的参数组件列表，可以取消注释  

            
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

            # 提取当前方法对应的参数值  
            method_param_names = list(method_params[method].keys())  
            param_values = []  
            for value, comp_method in zip(params, param_component_methods):  
                if comp_method == method:  
                    param_values.append(value)  

            # 在后台线程中运行训练，以避免阻塞UI  
            def training_thread():  
                metrics.update(start_training(model_name, dataset_name, method, param_values))  
            thread = threading.Thread(target=training_thread)  
            thread.start()  

            while thread.is_alive():  
                time.sleep(1)  
                if training_loss:  
                    loss_plot.update(plot_training_loss(training_loss))  

            loss_plot.update(plot_training_loss(training_loss))  
            metrics_output.update(metrics)  

        # 定义当微调方法改变时的更新函数  
        def update_params(method):  
            updates = []  
            for comp, comp_method in zip(param_components, param_component_methods):  
                if comp_method == method:  
                    updates.append(gr.update(visible=True))  
                else:  
                    updates.append(gr.update(visible=False))  
            return updates  

        # 当微调方法改变时，更新参数输入组件的可见性  
        method_choice.change(  
            fn=update_params,  
            inputs=method_choice,  
            outputs=param_components  
        )  

        # 定义训练按钮的输入  
        train_inputs = [model_choice, dataset_choice, method_choice] + param_components  

        train_button.click(  
            fn=on_train_click,  
            inputs=train_inputs,  
            outputs=[]  
        )  

        # 停止训练按钮  
        stop_button.click(fn=stop_training_fn, outputs=[])  

        # 保存模型按钮  
        save_button.click(fn=save_model_fn, outputs=[])  

    demo.launch(share=True)  

if __name__ == "__main__":  
    main()