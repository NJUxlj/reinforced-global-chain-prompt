import gradio as gr  
import torch  
import time  
import threading  
import numpy as np  
import matplotlib.pyplot as plt  

# 假设有一些模型和微调方法可供选择  
models = ["bert-base-uncased", "roberta-base", "gpt-2"]  
finetune_methods = ["prompt-tuning", "p-tuning", "lora"]  

# 微调方法对应的参数  
method_params = {  
    "prompt-tuning": {"learning_rate": 1e-3, "num_prompt_tokens": 10},  
    "p-tuning": {"learning_rate": 1e-3, "num_virtual_tokens": 20},  
    "lora": {"alpha": 32, "rank": 8, "learning_rate": 1e-4}  
}  

# 用于保存训练Loss的全局变量  
training_loss = []  
stop_training = False  # 用于控制训练线程  

def get_method_params(method):  
    params = method_params.get(method, {})  
    return params  

def update_params_ui(method):  
    params = get_method_params(method)  
    ui_elements = []  
    for param_name, default_value in params.items():  
        ui_elements.append(gr.Number(label=param_name, value=default_value))  
    return ui_elements  

def train_model(model_name, method, params):  
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
    metrics = {"Accuracy": np.random.rand(), "Precision": np.random.rand(), "Recall": np.random.rand(), "F1": np.random.rand()}  
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

def start_training(model_name, method, *param_values):  
    # 将参数值转换为字典  
    params_keys = list(get_method_params(method).keys())  
    params = dict(zip(params_keys, param_values))  
    metrics = train_model(model_name, method, params)  
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

        param_ui = gr.Column()  

        # 动态更新参数输入框  
        def update_ui(method):  
            return gr.Row(update_params_ui(method))  
        method_choice.change(fn=update_ui, inputs=method_choice, outputs=param_ui)  

        # 初始化参数输入框  
        param_inputs = update_params_ui(finetune_methods[0])  

        with gr.Row():  
            train_button = gr.Button("开始训练")  
            stop_button = gr.Button("停止训练")  
            save_button = gr.Button("保存模型")  

        with gr.Row():  
            loss_plot = gr.Image(label="训练Loss曲线")  
            metrics_output = gr.Json(label="评估指标")  

        # 训练按钮点击事件  
        def on_train_click(model_name, method, *params):  
            global training_loss  
            metrics = {}  

            # 在后台线程中运行训练，以避免阻塞UI  
            def training_thread():  
                metrics.update(start_training(model_name, method, *params))  
            thread = threading.Thread(target=training_thread)  
            thread.start()  

            while thread.is_alive():  
                time.sleep(1)  
                if training_loss:  
                    loss_plot.update(plot_training_loss(training_loss))  

            loss_plot.update(plot_training_loss(training_loss))  
            metrics_output.update(metrics)  

        input_params = [model_choice, method_choice] + param_inputs  
        train_button.click(fn=on_train_click, inputs=input_params, outputs=[loss_plot, metrics_output])  

        # 停止训练按钮  
        stop_button.click(fn=stop_training_fn, outputs=None)  

        # 保存模型按钮  
        save_button.click(fn=save_model_fn, outputs=None)  

    demo.launch(share=True)  

if __name__ == "__main__":  
    main()



