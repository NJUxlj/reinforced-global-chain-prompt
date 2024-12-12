from dataclasses import dataclass  
from typing import Dict, Optional, List, Union, Any  
import torch  
import numpy as np  
from accelerate import Accelerator  
from transformers import PreTrainedModel  
from torch.utils.data import DataLoader  
from sklearn.metrics import (  
    accuracy_score, precision_score, recall_score,   
    f1_score, confusion_matrix, classification_report  
)  
import evaluate  # huggingface evaluate library  

from load import DatasetConfig



class ModelEvaluator:  
    """  
    模型评估器类  
    支持多分类评估，特别适用于问答/选择题类型的任务  
    集成了分布式评估能力和丰富的评估指标  
    """  
    def __init__(  
        self,  
        accelerator: Accelerator,  
        config: DatasetConfig,  
    ):  
        self.accelerator = accelerator  
        self.config = config  
        self.custom_metrics=False
        self.verbose=True
        self.metric_average='weighted'
        
        # 初始化huggingface evaluate指标  
        self.metric = evaluate.combine([  
            evaluate.load("accuracy"),  
            evaluate.load("precision"),  
            evaluate.load("recall"),  
            evaluate.load("f1")  
        ])  
        
    def _process_batch_outputs(  
        self,  
        outputs: Any,  
        batch: Dict[str, torch.Tensor]  
    ) -> Dict[str, torch.Tensor]:  
        """  
        处理模型输出的批次结果  
        Args:  
            outputs: 模型输出  
            batch: 输入批次数据  
        Returns:  
            包含处理后预测结果的字典  
        """  
        # 获取logits并计算概率分布  
        logits = outputs.logits  # [batch_size x num_options, 2]  
        probs = torch.softmax(logits, dim=1)[:, 1]  # 获取正类概率  
        
        # 重塑概率分布和标签  
        probs = probs.view((-1, self.config.num_options))  # [batch_size, num_options]  
        labels = batch['labels'].view((-1, self.config.num_options))  
        
        # 获取预测答案和真实答案  
        pred_answers = probs.argmax(dim=1)  # [batch_size]  
        true_answers = labels.argmax(dim=1)  # [batch_size]  
        
        return {  
            'probs': probs,  
            'pred_answers': pred_answers,  
            'true_answers': true_answers,  
            'batch_size': pred_answers.size(0)  
        }  

    def _compute_metrics(  
        self,  
        all_preds: np.ndarray,  
        all_labels: np.ndarray,  
        all_probs: np.ndarray,  
        total_questions: int,  
        correct_questions: int  
    ) -> Dict[str, float]:  
        """  
        计算评估指标  
        """  
        metrics = {  
            'accuracy': accuracy_score(
                all_labels, all_preds, 
            ),  
            'precision': precision_score(  
                all_labels, all_preds,   
                average=self.metric_average  
            ),  
            'recall': recall_score(  
                all_labels, all_preds,   
                average=self.metric_average  
            ),  
            'f1': f1_score(  
                all_labels, all_preds,   
                average=self.metric_average  
            ),  
            'macro_accuracy': accuracy_score(
                all_labels, all_preds, 
            ), 
            'macro_precision': precision_score(  
                all_labels, all_preds,   
                average="macro"  
            ),  
            'macro_recall': recall_score(  
                all_labels, all_preds,   
                average="macro"  
            ),  
            'macro_f1': f1_score(  
                all_labels, all_preds,   
                average="macro"  
            ),  
            'question_accuracy': correct_questions / total_questions,  
            'mean_confidence': np.mean(all_probs.max(axis=1))  
        }  
        
        # 添加自定义指标  
        if self.custom_metrics:  
            # 这里可以添加自定义指标的计算逻辑  
            pass  
            
        return metrics  

    def _print_evaluation_results(  
        self,  
        metrics: Dict[str, float],  
        all_preds: np.ndarray,  
        all_labels: np.ndarray,  
        total_questions: int,  
        correct_questions: int  
    ) -> None:  
        """  
        打印评估结果  
        """  
        if not self.verbose:  # 不需要打印详细信息
            return  
            
        print("\n****************** Evaluation Results **************************")  
        print(f"Total questions evaluated: {total_questions}")  
        print(f"Correct questions: {correct_questions}")  
        print(f"Question-level accuracy: {metrics['question_accuracy']:.4f}")  
        print(f"Mean confidence: {metrics['mean_confidence']:.4f}")  
        
        # 打印预测分布  
        print("\nPrediction Distribution:")  
        for i in range(self.config.num_options):  
            count = (all_preds == i).sum()  
            print(f"Option {i}: {count} ({count/len(all_preds):.2%})")  
        
        # 打印混淆矩阵和分类报告  
        print("\nConfusion Matrix:")  
        print(confusion_matrix(all_labels, all_preds))  
        print("\n******************** Classification Report *********************")  
        print(classification_report(all_labels, all_preds))  
        print("*************************************************************\n")  

    def evaluate(  
        self,  
        model: PreTrainedModel,  
        eval_dataloader: DataLoader,  
    ) -> Dict[str, float]:  
        """  
        执行模型评估  
        Args:  
            model: 预训练模型  
            eval_dataloader: 评估数据加载器  
        Returns:  
            包含各项评估指标的字典  
        """  
        # 保存原始训练状态  
        training_state = model.training  
        model.eval()  
        
        # 初始化评估统计  
        all_preds = []  
        all_labels = []  
        all_probs = []  
        total_questions = 0  
        correct_questions = 0  
        
        # 执行评估  
        for batch in eval_dataloader:  
            with torch.no_grad():  
                outputs = model(**batch)  
                batch_results = self._process_batch_outputs(outputs, batch)  
                
                # 收集预测结果  
                preds, labels = self.accelerator.gather_for_metrics(  
                    (batch_results['pred_answers'], batch_results['true_answers'])  
                )  
                
                # 更新统计  
                total_questions += batch_results['batch_size']  
                correct_questions += (  
                    batch_results['pred_answers'] == batch_results['true_answers']  
                ).sum().item()  
                
                all_preds.append(preds.cpu())  
                all_labels.append(labels.cpu())  
                all_probs.append(batch_results['probs'].cpu())  
        
        # 合并所有结果  
        all_preds = torch.cat(all_preds).numpy()  
        all_labels = torch.cat(all_labels).numpy()  
        all_probs = torch.cat(all_probs).numpy()  
        
        # 计算指标  
        metrics = self._compute_metrics(  
            all_preds, all_labels, all_probs,  
            total_questions, correct_questions  
        )  
        
        # 在主进程上打印结果  
        if self.accelerator.is_main_process:  
            self._print_evaluation_results(  
                metrics, all_preds, all_labels,  
                total_questions, correct_questions  
            )  
        
        # 等待所有进程完成  
        self.accelerator.wait_for_everyone()  
        
        # 恢复模型原始状态  
        model.train(training_state)  
        
        return metrics