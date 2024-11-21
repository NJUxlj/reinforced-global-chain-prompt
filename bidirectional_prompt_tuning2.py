from transformers import (
    default_data_collator,
    get_linear_schedule_with_warmup,
    BertForSequenceClassification,
    AutoModel,
    AutoTokenizer,  
    BertTokenizerFast,
    AutoModelForSequenceClassification,  
    Trainer,  
    TrainingArguments  
)

from transformers.modeling_outputs import (
    SequenceClassifierOutput, # 是一种官方的输出格式
    BaseModelOutputWithPoolingAndCrossAttentions,
    MultipleChoiceModelOutput
)
# from transformers import PromptForSequenceClassification  
from peft import (
    PromptTuningInit, 
    PromptTuningConfig,
    get_peft_model,
    TaskType
)

from accelerate import(
    Accelerator,
)
from accelerate.logging import get_logger


from load import *

from utils import *

from causal_modeling import *


from autocot.make_embeddings import (
    get_cot_context,
    rollback_one_step_extend,
    ChainEncodingArguments
)

from datasets import (
    load_dataset,
)
from torch.utils.data import (
    DataLoader,
    Dataset
)
from torch.utils.data.distributed import DistributedSampler

from config import *

import torch
import torch.nn as nn
import torch.nn.functional as F  
from torch.optim import Adam
import numpy as np
import csv
import evaluate
import gensim
from gensim import corpora, models
import nltk
import logging
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics.pairwise import cosine_similarity  

from sklearn.cluster import KMeans
from typing import List, Dict
from collections import defaultdict

from sentence_transformers import SentenceTransformer, models
from dataclasses import dataclass


import nltk  
# nltk.download('punkt') 
# nltk.download('punkt_tab') 
# nltk.download('stopwords')  
from nltk.corpus import stopwords  
from nltk.tokenize import word_tokenize
stop_words = set(stopwords.words('english')) 




device = Config['device']




@dataclass
class BaasPromptConfig:
    model_name: str = "bert-base-uncased"
    model_path: str = "bert-base-uncased"  # 预训练模型名称
    peft_method: str = "baas-tuning"
    auto_model_class:type = AutoModelForSequenceClassification # 对于类类型的字段，使用 type 作为类型注解
    dataset_name:str = "race" 
    prefix_length: int = 10                        # prefix-tuning的默认前缀长度  
    suffix_length: int = 10
    num_labels: int = 2                           # MCQA的选项数量 (A,B,C,D)  
    batch_size:int = 32
    num_epochs:int = 2
    dropout: float = 0.1                          # dropout率  
    max_seq_length: int = 512                         # 最大序列长度  
    learning_rate: float = 0.3                   # 前缀参数的学习率  
    model_learning_rate: float = 1e-5             # 模型参数的学习率（如果需要微调）  
    
    prefix_projection: bool = True               # 是否使用MLP投影前缀  
    prefix_hidden_size: int = 768               # MLP中的P_theta'  即，MLP输入的隐单元维度  huggingface 默认它==encoder_hidden_size
    encoder_hidden_size:int = 512   # 重参数化器MLP的隐藏层大小# prefix token 的维度 (P_theta')
    
    warmup_steps: int = 500  # 添加预热步骤  
    weight_decay: float = 1e-5  # 添加权重衰减 
    beta1_decay:float = 0.9   #beta1: 一阶矩估计的指数衰减率（默认0.9）用于Adam优化器
    beta2_decay:float = 0.8   # 用于AdaFactor optimizer
    total_training_steps = 30000  # 总的训练步数
    early_stop_steps = 10
    optimizer_class:type = Adam 
    
    all_layers:bool = False  # 是否在所有层插入prefix, suffix
    

    
class BaasPromptEncoder(nn.Module):  
    def __init__(  
        self,   
        config: BassPromptConfig,
        num_layers: int = 2,  
        dropout: float = 0.1
    ):  
        super().__init__()  
        self.hidden_size = config.prefix_hidden_size  
        
        # self.prefix_embeddings = prefix_embeddings # shape = (prefix_length, hidden_size)
        # self.suffix_embeddings = suffix_embeddings  # shape = (suffix_length, hidden_size)
        
        # 2. 双向LSTM编码器  
        self.forward_lstm = nn.LSTM(  
            self.hidden_size,  
            self.hidden_size // 2,  
            num_layers=num_layers,  
            bidirectional=True,  
            batch_first=True  
        )  
        self.backward_lstm = nn.LSTM(  
            self.hidden_size,  
            self.hidden_size // 2,  
            num_layers=num_layers,  
            bidirectional=True,  
            batch_first=True  
        )  
        
        self.prefix_to_suffix_attention = MultiHeadAttention(  
            self.hidden_size, num_heads=8  
        )  
        self.suffix_to_prefix_attention = MultiHeadAttention(  
            self.hidden_size, num_heads=8  
        )  
        
        # 4. 自适应门控融合  
        self.prefix_gate = AdaptiveGate(self.hidden_size)  
        self.suffix_gate = AdaptiveGate(self.hidden_size)  
        
        # 5. 位置编码  
        self.prefix_pos_embedding = SinusoidalPositionalEmbedding(self.hidden_size)  
        self.suffix_pos_embedding = SinusoidalPositionalEmbedding(self.hidden_size)  
        
        # 6. 输出转换层  
        self.prefix_output_layer = OutputTransformation(self.hidden_size)  
        self.suffix_output_layer = OutputTransformation(self.hidden_size)  
        
        self.dropout = nn.Dropout(dropout)  
        self.layer_norm = nn.LayerNorm(self.hidden_size)  

    def forward(
            self,
            prefix_embeddings: torch.Tensor,  
            suffix_embeddings: torch.Tensor,
            batch_size:int =1 ,
        ):  
        '''
        Args:
            prefix_embeddings, suffix_embeddings: shape = (seq_length, hidden_size)
            
        return  
            prefix_output, suffix_output   shape = (batch_size, seq_length, hidden_size)
        
        '''
        # 1. 扩展batch维度  
        prefix_embeds = prefix_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        suffix_embeds = suffix_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        
        # 2. 添加位置编码  
        prefix_embeds = self.prefix_pos_embedding(prefix_embeds)  
        suffix_embeds = self.suffix_pos_embedding(suffix_embeds)  
        
        # 3. 双向LSTM编码  
        # 前向处理  
        prefix_forward, _ = self.forward_lstm(prefix_embeds)  
        suffix_forward, _ = self.forward_lstm(suffix_embeds)  
        
        # 反向处理  
        prefix_backward, _ = self.backward_lstm(torch.flip(prefix_embeds, [1]))  
        suffix_backward, _ = self.backward_lstm(torch.flip(suffix_embeds, [1]))  
        prefix_backward = torch.flip(prefix_backward, [1])  
        suffix_backward = torch.flip(suffix_backward, [1])  
        
        # 融合双向表示  
        prefix_bidirectional = self.prefix_gate(prefix_forward, prefix_backward)  
        suffix_bidirectional = self.suffix_gate(suffix_forward, suffix_backward)  
        
        # 4. 交互注意力  
        prefix_attended = self.prefix_to_suffix_attention(  
            prefix_bidirectional, suffix_bidirectional, suffix_bidirectional  
        )  
        suffix_attended = self.suffix_to_prefix_attention(  
            suffix_bidirectional, prefix_bidirectional, prefix_bidirectional  
        )  
        
        # 5. 残差连接和归一化  
        prefix_output = self.layer_norm(prefix_bidirectional + prefix_attended)  
        suffix_output = self.layer_norm(suffix_bidirectional + suffix_attended)  
        
        # 6. 最终转换  
        prefix_output = self.prefix_output_layer(prefix_output)  
        suffix_output = self.suffix_output_layer(suffix_output)  
        
        return prefix_output, suffix_output 
    


class BassPromptModel(torch.nn.Module):  
    def __init__(self, model, config:BaasPromptConfig):  
        super(BassPromptModel, self).__init__()  
        self.model = model
        
        self.rollback_decoder = RollbackDecoderWithHead(
            d_model=config.prefix_hidden_size,
            d_ff=config.prefix_hidden_size*4,
            num_heads=8,
        )
        
        self.chain_encode_args = ChainEncodingArguments(
            dataset=config.dataset_name,
            hidden_size=config.encoder_hidden_size, # 这个hidden_size最终会传给encode cot chain用到的 sentence transformer
            output_dir="experiment/race",
            embedding_dir = "./embeddings/race",
            context_dir = "./context/race"
        )
        
        self.prefix_embeddings = self.initialize_prefix_prompts(
            
        )
        
        self.suffix_embeddings = self.initialize_suffix_prompts() #shape =  (seq_length, hidden_size)
        
        self.prompt_encoder = BaasPromptEncoder(config)
          
        self.prefix_embeddings, self.suffix_embeddings = self.prompt_encoder.forward(
            prefix_embeddings = self.prefix_embeddings,
            suffix_embeddings = self.suffix_embeddings,
            )  # shape = (batch_size, seq_length, hidden_size)
        
        self.num_prefix_tokens = config.prefix_length
        self.num_suffix_tokens = config.suffix_length
        self.embedding_layer = self.model.get_input_embeddings()  # 获取词嵌入层
        

    
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):  
        # 原始输入嵌入
        # print(f"input_ids.shape = {input_ids.shape}")
        input_ids = input_ids.squeeze(1) 
        inputs_embeds = self.embedding_layer(input_ids)  
        
        batch_size = inputs_embeds.size(0)  
        
        # 将前缀和后缀Prompt Embeddings扩展到batch维度  
        prefix_embeds = self.prefix_embeddings.expand(batch_size, -1, -1)  
        suffix_embeds = self.suffix_embeddings.expand(batch_size, -1, -1)  
        
        # print(f"prefix.shape = {prefix_embeds.shape}")
        # print(f"suffix.shape = {suffix_embeds.shape}")
        # print(f"inputs_embeds.shape = {inputs_embeds.shape}")

        # 拼接前缀、原始输入和后缀嵌入  
        inputs_embeds = torch.cat([prefix_embeds, inputs_embeds, suffix_embeds], dim=1)  # (4, 512, 768)
        
        # 调整attention_mask  
        if attention_mask is not None:  
            prefix_mask = torch.ones(batch_size, self.num_prefix_tokens, device=device)  
            suffix_mask = torch.ones(batch_size, self.num_suffix_tokens, device=device)  

            # print(f"attention_mask.shape = {attention_mask.shape}")
            # print(f"prefix_mask.shape = {prefix_mask.shape}")
            # print(f"suffix_mask.shape = {suffix_mask.shape}")
            
            attention_mask = attention_mask.squeeze(1)
            attention_mask = torch.cat([prefix_mask, attention_mask, suffix_mask], dim=1)  # (4, 522)
            
            # print(f"attention_mask.shape after concat = {attention_mask.shape}") 
        
        if token_type_ids is not None:  
            prefix_type_ids = torch.zeros(batch_size, self.num_prefix_tokens, device=device)
            suffix_type_ids = torch.zeros(batch_size, self.num_suffix_tokens, device=device)
            
            # print(f"token_type_ids.shape = {token_type_ids.shape}")
            # print(f"prefix_type_ids.shape = {prefix_type_ids.shape}")
            # print(f"suffix_type_ids.shape = {suffix_type_ids.shape}")
            
            token_type_ids = token_type_ids.squeeze(1)
            token_type_ids = torch.cat([prefix_type_ids, token_type_ids, suffix_type_ids], dim=1)
            
            token_type_ids = token_type_ids.long() # (4, 522)
            
            # print(f"token_type_ids.shape after concat = {token_type_ids.shape}")

        
        # 调用原始模型的forward方法  
        outputs = self.model(  
            inputs_embeds=inputs_embeds,  
            attention_mask=attention_mask,  
            token_type_ids=token_type_ids,
            labels=labels  
        )  
        
        return outputs  

    def print_trainable_parameters(self):  
        """print trainable parameters' number and ratio"""  
        trainable_params = 0  
        all_params = 0  
        for name, param in self.named_parameters():  
            num_params = param.numel()  
            all_params += num_params  
            if param.requires_grad:  
                trainable_params += num_params  
        print(f"trainable param number: {trainable_params}")  
        print(f"total param number: {all_params}")  
        print(f"trainable param ratio: {100 * trainable_params / all_params:.2f}%")  


    def initialize_suffix_prompts(self)->torch.Tensor:  
        """  
        use Auto-CoT reasoning steps as the suffix prompts
        
        return 
        
            shape = (num_suffix_tokens, embedding_size)
        """  
        # 假设我们已经有AutoCoT生成的推理步骤steps，并将其保存为文本列表
        context = rollback_one_step_extend(
            self.num_suffix_tokens,
            args = self.chain_encode_args,
            model = self.rollback_decoder
            )
        
        suffix_embeddings = context
        
        
        return suffix_embeddings
    

    def initialize_prefix_prompts(self, dataset_path, model, tokenizer, num_prefix_tokens, embedding_size, config:BaasPromptConfig ,classes_initiate_method = "cluster", K=5): 
        """ 
        use article classification tokens' weighted sum as prefix prompts
        """
        class_embeddings = None
        if classes_initiate_method == "normal":
            class_embeddings = get_classes_for_dataset(dataset_path, model, tokenizer, embedding_size = embedding_size, num_topics=num_prefix_tokens, K=5, max_length=512)
        elif classes_initiate_method == "cluster":
            class_embeddings = get_classes_by_clustering(dataset_path, model, tokenizer, embedding_size = embedding_size, num_topics=num_prefix_tokens, K=5, max_length=512)
        elif classes_initiate_method == "lda":
            class_embeddings = get_classes_by_lda(dataset_path, model, tokenizer, embedding_size = embedding_size, num_topics=num_prefix_tokens, K=5, max_length=512)
        else:
            raise ValueError("Invalid classes_initiate_method, Please choose from ['normal', 'cluster', 'lda']")

        
        prefix_embeddings = torch.zeros(num_prefix_tokens, embedding_size, device=device)
        
        for i in range(num_prefix_tokens):  
            prefix_embeddings[i] = class_embeddings[i]
        
        prefix_prompt_embeddings = torch.nn.Parameter(prefix_embeddings, requires_grad=True)   # (num_prefix_tokens, embedding_size)

        return prefix_prompt_embeddings





def reformat_input(dataset_path, tokenizer, max_length=512, reformat_type = "normal"):
    '''
       This function will be used when generating class labels for each sample in the dataset.
        
       根据数据集的格式将问题格式化为相应的字符串
        e.g. if dataset = race: "Article: ... Question: ... Options: ... Answer:"
        
        
        reformat_type: "normal" or "lda" or "cluster":
        
            "normal": use the get_classes_for_dataset() to get class labels
            "lda": use the get_classes_lda() to get class labels
            "cluster": use the get_classes_cluster() to get class labels
    '''
    
    reformat_dict = {
        "normal": "get_classes_for_dataset()",
        "lda": "get_classes_by_lda()",
        "cluster": "get_classes_by_cluster()"
    }
    
    print(f"reformat input using reformat_type = {reformat_dict[reformat_type]}")
    if dataset_path == Config["datasets"]["race"]:
        dataset_name = "race"
        
        if reformat_type == "normal":
            # # corse-grained preprocessing
            # ds, classes, tokenizer = preprocess_race(ds, tokenizer)
            
            # # fine-grained preprocessing
            # processed_ds = ds.map(
            #     lambda examples: preprocess_function_race(examples, max_length=max_length, tokenizer=tokenizer), 
            #     batched=True,
            #     num_proc=NUM_CPU_PROCESSES,
            #     remove_columns=ds['train'].column_names,
            #     load_from_cache_file=False,
            #     desc="Running tokenizer on dataset",
            # )     
            
            processed_ds = preprocess_dataset_peft(dataset_name, max_length = max_length)
            train_ds = processed_ds["train"]
            
        elif reformat_type == "lda":
            ds = load_dataset_from_huggingface(dataset_path, "all")
            
            processed_ds = ds.map(
                lambda examples: {
                    "combined_input": [f"Artical:{examples['article'][index]}\n\nQuestion:{examples['question'][index]}\n\n \
                                            Options:{examples['options'][index]}\n\nAnswer:" for index, x in enumerate(examples['article'])]  
                },
                batched=True,
                num_proc=NUM_CPU_PROCESSES,
                remove_columns=ds['train'].column_names,
                load_from_cache_file=False,
                desc="Running tokenizer on dataset",
            )
            
            # transfer the dataset into List[str]
            processed_ds = [text for text in processed_ds['train']['combined_input']]  
            
            
            train_ds: List[List[str]] = []
            for text in processed_ds:
                text = text.lower()
                tokens = word_tokenize(text)
                # remove stopwords and non-alphabetic characters
                tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
                train_ds.append(tokens)
            
        elif reformat_type == "cluster":
            processed_ds = preprocess_dataset_peft(dataset_name, max_length = max_length)
            train_ds = processed_ds["train"]
        else:
            raise ValueError("Invalid reformat_type, please choose from 'normal', 'lda', 'cluster")
        
                    
        
    elif dataset_path == Config["datasets"]["sciq"]:
        dataset_name = "sciq"
        
        if reformat_type == "normal":
            processed_ds = preprocess_dataset_peft(dataset_name, max_length = max_length)
            train_ds = processed_ds["train"]
            
        elif reformat_type == "lda":
            ds = load_dataset_from_huggingface(dataset_path, "all")
            
            processed_ds = ds.map(
                lambda examples: {
                    "combined_input": [f"Artical:{examples['article'][index]}\n\nQuestion:{examples['question'][index]}\n\n \
                                            Options:{examples['options'][index]}\n\nAnswer:" for index, x in enumerate(examples['article'])]  
                },
                batched=True,
                num_proc=NUM_CPU_PROCESSES,
                remove_columns=ds['train'].column_names,
                load_from_cache_file=False,
                desc="Running tokenizer on dataset",
            )
            
            # transfer the dataset into List[str]
            processed_ds = [text for text in processed_ds['train']['combined_input']]  
            
            
            train_ds: List[List[str]] = []
            for text in processed_ds:
                text = text.lower()
                tokens = word_tokenize(text)
                # remove stopwords and non-alphabetic characters
                tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
                train_ds.append(tokens)
            
        elif reformat_type == "cluster":
            processed_ds = preprocess_dataset_peft(dataset_name, max_length = max_length)
            train_ds = processed_ds["train"]
        else:
            raise ValueError("Invalid reformat_type, please choose from 'normal', 'lda', 'cluster")
        
    
    elif dataset_path == Config["datasets"]["commonsense_qa"]:
        dataset_name = "commonsense_qa"
        if reformat_type == "normal":
            processed_ds = preprocess_dataset_peft(dataset_name, max_length = max_length)
            train_ds = processed_ds["train"]
            
        elif reformat_type == "lda":
            ds = load_dataset_from_huggingface(dataset_path, "all")
            
            processed_ds = ds.map(
                lambda examples: {
                    "combined_input": [f"Artical:{examples['article'][index]}\n\nQuestion:{examples['question'][index]}\n\n \
                                            Options:{examples['options'][index]}\n\nAnswer:" for index, x in enumerate(examples['article'])]  
                },
                batched=True,
                num_proc=NUM_CPU_PROCESSES,
                remove_columns=ds['train'].column_names,
                load_from_cache_file=False,
                desc="Running tokenizer on dataset",
            )
            
            # transfer the dataset into List[str]
            processed_ds = [text for text in processed_ds['train']['combined_input']]  
            
            
            train_ds: List[List[str]] = []
            for text in processed_ds:
                text = text.lower()
                tokens = word_tokenize(text)
                # remove stopwords and non-alphabetic characters
                tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
                train_ds.append(tokens)
            
        elif reformat_type == "cluster":
            processed_ds = preprocess_dataset_peft(dataset_name, max_length = max_length)
            train_ds = processed_ds["train"]
        else:
            raise ValueError("Invalid reformat_type, please choose from 'normal', 'lda', 'cluster")
    
    
    elif dataset_path == Config["datasets"]["dream"]['all']:
        dataset_name = "dream"
        if reformat_type == "normal":
            processed_ds = preprocess_dataset_peft(dataset_name, max_length = max_length)
            train_ds = processed_ds["train"]
            
        elif reformat_type == "lda":
            ds = load_dataset_from_huggingface(dataset_path, "all")
            
            processed_ds = ds.map(
                lambda examples: {
                    "combined_input": [f"Artical:{examples['article'][index]}\n\nQuestion:{examples['question'][index]}\n\n \
                                            Options:{examples['options'][index]}\n\nAnswer:" for index, x in enumerate(examples['article'])]  
                },
                batched=True,
                num_proc=NUM_CPU_PROCESSES,
                remove_columns=ds['train'].column_names,
                load_from_cache_file=False,
                desc="Running tokenizer on dataset",
            )
            
            # transfer the dataset into List[str]
            processed_ds = [text for text in processed_ds['train']['combined_input']]  
            
            
            train_ds: List[List[str]] = []
            for text in processed_ds:
                text = text.lower()
                tokens = word_tokenize(text)
                # remove stopwords and non-alphabetic characters
                tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
                train_ds.append(tokens)
            
        elif reformat_type == "cluster":
            processed_ds = preprocess_dataset_peft(dataset_name, max_length = max_length)
            train_ds = processed_ds["train"]
        else:
            raise ValueError("Invalid reformat_type, please choose from 'normal', 'lda', 'cluster")
        
        
    else:
        raise ValueError("dataset_path not supported, we can not reformat dataset using a wrong name, please change another in [race, sciq, commonsense_qa, dream]")
    
    return train_ds




def get_classes_for_dataset(dataset_path, model, tokenizer, embedding_size, num_topics = 5, K=5, max_length=512)->List[torch.Tensor]:
    '''
    get the label collection of some specific dataset
        
    Args:
        dataset_path: the path of dataset, should be in [race, race-m, race-h, multirc, arc]
        K: number of classes in the dataset
        model: we need to get the embedding weight to calc cosine similarity
        
        num_topics: number of classes to be generated == num_suffix_tokens
        
        K : number of sub-classes to be generated
    
    Procedure:
        1. load the dataset
        2. load the first 1000 examples from the training set.
        3. get the sequence embedding of each example.
        4. get the avearage pooling of the 1000 sequence embeddings, for every 1000 examples, so there are in total len(train_data)//1000 pooled embeddings.
        5. calculate the cosine similarity between the pooled embedding and every token in the vocab, average the similarity on len(train_data)//1000 pooled embeddings.
        6. for each token in the vocab, we multiply the similarity by the frequency of the token in the training set. (TF-AS)
        6. select the top-K tokens in the vocab as the class labels, with the highest TF-AS score.
    
    Return:
        classes: a list of str, which contains the class labels
    '''
    
    print("get class labels by TF-IQF-AS (normal method) ~~~")
    
    classes = []
    model = model.eval()
    device = Config['device']
    model.to(device)
    
    vocab = tokenizer.get_vocab()
    inverse_vocab = {v:k for k,v in vocab.items()} # index-token mapping
    
    all_pooled_embeddings = [] # store the pooled embeddings of every 1000 examples
    
    vocab_size = tokenizer.vocab_size
    # count the word frequency of each word in vocab on the training set
    word_freq = {token_id:0 for token_id in range(vocab_size)}
    
    
    train_ds = reformat_input(dataset_path, tokenizer, max_length=max_length)
    
    train_data_loader = DataLoader(train_ds, batch_size=1000, 
                                   collate_fn=default_data_collator, 
                                   num_workers=0, # use as your need
                                   shuffle=False)
    
    # record whether this token occurs in this example or not, 1 for yes, 0 for no
    word_occurence_per_example:Dict[int, Dict[int, int]] = {
                                                                token_id:{
                                                                    example_id:0 
                                                                        for example_id in range(len(train_ds))
                                                                    } 
                                                                for token_id in range(vocab_size)
                                                            }

    
    with torch.no_grad():
        global_step = 0
        for index, batch in enumerate(tqdm(train_data_loader)):
            # get the average embedding of each batch
            batch = {k:v.to(device) for k, v in batch.items()}
            
            input_ids = batch['input_ids']
            input_ids_np = input_ids.cpu().numpy()
            for ids in input_ids_np:  
                for id in ids:
                    id = int(id)  
                    word_freq[id] += 1
                    word_occurence_per_example[id][global_step] = 1
                
                global_step += 1
            
            # outputs = model(**batch)
            embeddings = model.embeddings(input_ids) # shape = (batch_size, seq_len, hidden_size)
            # firstly, pooling on the seq_len dimension
            pooled_embedding = embeddings.mean(dim=1) # shape = (batch_size, hidden_size)
            # then, pooling on the batch_size dimension
            # all_pooled_embeddings.append(pooled_embedding.mean(dim=0).cpu()) # shape = (hidden_size)
            all_pooled_embeddings.append(pooled_embedding.mean(dim=0).unsqueeze(0)) # shape = (1, hidden_size)
            
    
    # print("all_pooled_embeddings = ", all_pooled_embeddings)
    print("=================================")
    print("all_pooled_embeddings[0].shape = ", all_pooled_embeddings[0].shape)
    print("all_pooled_embeddings[10].shape = ", all_pooled_embeddings[10].shape)
    print("all_pooled_embeddings[20].shape = ", all_pooled_embeddings[20].shape)
    
    
    print("all_pooled_embeddings.length = ", len(all_pooled_embeddings))
    print("len(train_data) = ", len(train_data_loader)) # 88
    print("len(train_data)//1000 = ", len(train_data_loader)//1000) # 0
    print("==================================")
    # vertically stack all the pooled embeddings into a matrix
    # "dim" points to the dimension of each pooled embedding
    pooled_embeddings_matrix = torch.cat(all_pooled_embeddings, dim=0) # shape = (len(train)//1000, hidden_size)

    num_pooled = len(train_ds)//1000
    if num_pooled == 0:
        num_pooled = 1
        
    vocab_size =  tokenizer.vocab_size
    token_embeddings = model.embeddings.word_embeddings.weight # [vocab_size, hidden_size]
    
    
    print("pooled embeddings matrix shape: ", pooled_embeddings_matrix.shape)
    print("token embeddings matrix shape: ", token_embeddings.shape)
    
    # 计算池化嵌入与词汇表中每个词嵌入的余弦相似度，并取平均值
    all_similarities_token_to_corpus = []
    for token_embedding in tqdm(token_embeddings):
        pooled_embedding = pooled_embedding.squeeze(0) # shape = (hidden_size)
        cosine_similarities = F.cosine_similarity(
            x1=token_embedding,
            x2 =pooled_embeddings_matrix,
            dim=1 # 代表我们在行的维度上进行计算
        ) # shape = (num_pooled)
        
        all_similarities_token_to_corpus.append(cosine_similarities.mean(dim=0).item())

    all_similarities_token_to_corpus = torch.tensor(all_similarities_token_to_corpus)
    print("all_similarities_token_to_corpus.shape = ", all_similarities_token_to_corpus.shape)
    
    # avg_similarities = cosine_similarities.mean(dim=0) # shape = (vocab_size)
    
    
    # transfer the word frequency to tensor
    freq_tensor = torch.zeros(vocab_size) 
    for token_id, freq in word_freq.items():  
        if token_id < vocab_size:  
            freq_tensor[token_id] = freq  
    
    # store the total occurence number of each token on all examples
    word_occurence_total = [0]*len(train_ds)
    
    # calculate the total occurence in all examples of each token
    for token_id in range(vocab_size):
        for example_id in range(len(train_ds)):
            word_occurence_total[token_id] += word_occurence_per_example[token_id][example_id]
    
    # calculate TF-IQF-VS score
    tf_iqf_vs_scores:torch.Tensor = all_similarities_token_to_corpus * freq_tensor * torch.tensor(word_occurence_total/len(train_ds))
    
    # filter special tokens
    filtered_scores = []
    for token_id, score in enumerate(tf_iqf_vs_scores):
        token = inverse_vocab[token_id]
        if token.strip() and not token.startswith("[") and token.isalpha() and token not in stop_words:
            filtered_scores.append([token_id, score])
    filtered_scores = torch.tensor(filtered_scores)
    
    # choose Top-K as class labels
    topk_scores, topk_indices = filtered_scores.topk(num_topics, dim=1)
    for idx in topk_indices:    
        classes.append((idx, inverse_vocab[idx]))  
    
    print("class labels are: ")
    for idx, label in classes:
        print(f"class {idx}: {label}")
    
    class_embeddings = []
    for index, _ in classes:
        class_embeddings.append(token_embeddings[index])
    return class_embeddings

def get_classes_by_clustering(
    dataset_path, 
    model, 
    tokenizer, 
    embedding_size, 
    num_topics=5, 
    K=5, 
    max_length=512, 
    use_trained_embeddings=False,
    cache_dir='cached_embeddings'  # 存储embeddings的目录  
    )->List[torch.Tensor]:
    '''
    Args:
        num_topics: number of classes to be generated == num_suffix_tokens
        
        K : number of sub-classes to be generated
        
        cache_dir: 存储embeddings的目录  
        use_trained_embeddings: 是否使用已缓存的embeddings  
    
    '''
    print(f"get class labels by Clustering ~~~~")
    os.makedirs(cache_dir, exist_ok=True)
    
    
    # 生成唯一的缓存文件名（基于模型名称和数据集路径）  
    model_name = get_model_name_using_model(model)
    dataset_name = os.path.basename(dataset_path) 
    cache_filename = f"embeddings_{model_name}_{dataset_name}.pt"  
    cache_path = os.path.join(cache_dir, cache_filename) 
  
    # 保存数据集信息，用于验证缓存是否匹配  
    metadata = {  
        'dataset_path': dataset_path,  
        'model_name': model_name,  
        'max_length': max_length,  
        'embedding_size': embedding_size  
    }  
    metadata_path = os.path.join(cache_dir, f"{cache_filename}_metadata.json")  

    # 如果启用缓存且缓存文件存在，尝试加载缓存的embeddings  
    if use_trained_embeddings and os.path.exists(cache_path) and os.path.exists(metadata_path):  
        try:
            # 首先验证metadata是否匹配  
            with open(metadata_path, 'r') as f:  
                cached_metadata = json.load(f)  
                
            if all(cached_metadata[k] == metadata[k] for k in metadata.keys()):  
                print(f"Loading cached embeddings from {cache_path}")  
                # 使用torch.load加载缓存的embeddings  
                embeddings = torch.load(cache_path)  
                print(f"Loaded embeddings shape: {embeddings.shape}")
                # return process_embeddings(embeddings, num_topics, K)  # 假设有这个后处理函数[聚类逻辑]
                
            else:  
                print("Cache metadata mismatch, recomputing embeddings for clustering...") 
        except (json.JSONDecodeError, FileNotFoundError, RuntimeError) as e:
            print(f"Error, when loading cache: {e}. Recomputing embeddings...")
    
    else:
        classes = []
        device = Config['device']
        # model.to(device)
        model = model.eval()
        
        
        accelerator = Accelerator()
        # model = accelerator.prepare(model)
        
        
        all_pooled_embeddings = [] # store the pooled embeddings
        
        vocab_size = tokenizer.vocab_size

        
        train_ds: Dict[str,torch.Tensor[List]] = reformat_input(dataset_path, tokenizer, max_length=max_length, reformat_type = 'normal')
        
        print(f"The training data is reformated, now we get each example's embedding using the model~~~")
        train_data_loader = DataLoader(train_ds, 
                                    batch_size=Config['batch_size'], 
                                    collate_fn=default_data_collator, 
                                    num_workers=NUM_CPU_PROCESSES, # use as your need
                                    pin_memory=True,
                                    shuffle=False)
        
        model, train_data_loader = accelerator.prepare(model, train_data_loader)
        
        all_batch_embeddings = []
        with torch.no_grad():
            for index, batch in enumerate(tqdm(train_data_loader)):
                # batch = {k: v.to(device) for k, v in batch.items()}
                
                # if 'labels' exists in batch，delete it，because inference does not need labels  
                if 'labels' in batch:
                    del batch['labels']
                
                outputs = model(**batch)
                if hasattr(outputs, "last_hidden_state"): # AutoModel
                    last_hidden_state = outputs.last_hidden_state # shape = (batch_size, seq_len, hidden_size)
                elif hasattr(outputs, "hidden_states"): # AutoModelForSequenceClassification
                    last_hidden_state = outputs.hidden_states[-1] # shape = (batch_size, seq_len, hidden_size)
                else:
                    raise ValueError("Can not extract the \"last_hidden_state\" from the model output")
                
                
                # average pooling 
                batch_embeddings = last_hidden_state.mean(1) # shape = (batch_size,  hidden_size)
                
                # print("batch_embeddings.shape = ", batch_embeddings.shape)
                # for embedding in batch_embeddings:
                all_batch_embeddings.append(batch_embeddings) # shape = (1, hidden_size) # 避免循环 
                
                # 定期清理缓存  
                if torch.cuda.is_available():  
                    torch.cuda.empty_cache()  
                
                
        embeddings = torch.cat(all_batch_embeddings, dim=0)
        print("all_embeddings.shape = ", embeddings.shape)
        
        # 在分布式环境中收集所有进程的嵌入向量  
        embeddings = accelerator.gather(embeddings)  
        
        
        # 保存embeddings到缓存  
        print(f"Saving new embeddings to {cache_path}")  
        torch.save(embeddings, cache_path)  
        
        # 保存metadata  
        with open(metadata_path, 'w') as f:  
            json.dump(metadata, f) 
    
    # no matter how you get embeddings (cache/model infer), we need to convert it to numpy array for clustering
    embeddings = embeddings.cpu().numpy() 
             
    # clustering
    print(f"doing K-Means clustering, cluster number = {num_topics}, each topic contains {K} words")
    
    kmeans = KMeans(n_clusters=num_topics, random_state=42)
    kmeans.fit(embeddings)
    
    centroids = kmeans.cluster_centers_
    
    print("prepare candidate vocabulary...")  
    vocab = tokenizer.get_vocab()  
    inv_vocab = {v: k for k, v in vocab.items()}  # inverse to a id-to-token mapping

    # filter the vocabulary，we only keep the words that are not stop words, all alphabetic words, and not in special tokens [CLS]...
    candidate_tokens = []
    for token_id in inv_vocab:  
        token = inv_vocab[token_id]  
        # skip special tokens, like: [CLS], [SEP], [PAD], [MASK], and sub-words (begining with ##)  
        if token.startswith('[') or token.startswith('##'):  
            continue  
        if token.isalpha() and token not in stop_words:  
            candidate_tokens.append(token)  

    print(f"candidate token numbers: {len(candidate_tokens)}")

    print("calculate the candidate token embedding...")  
    candidate_token_embeddings = []  

    candidate_token_ids = [index for index, _ in enumerate(candidate_tokens)]  
    
    candidate_token_ids = torch.tensor(candidate_token_ids, dtype=torch.long).to(device).unsqueeze(1) # shape = (num_tokens, 1) == (batch_size, seq_len)
    
    # extract the base model from the accelerator, so that we can get embeddings
    model_unwrapped = accelerator.unwrap_model(model)
    model_unwrapped.eval() # save resources
    
    
    # 加入判断逻辑
    # candidate_token_embeddings = model_unwrapped.embeddings(candidate_token_ids) # shape = (batch_size, 1, hidden_size) == (num_tokens, 1, hidden_size)
    candidate_token_embeddings = get_vocab_embeddings_from_model(model_unwrapped, candidate_token_ids)
    candidate_token_embeddings =  candidate_token_embeddings.squeeze(1)
    
    print("candidate token embeddings shape: ", candidate_token_embeddings.shape)

    print("find each centroid a most similar token...")  

    cluster_labels = []  
    cluster_label_embeddings = []
    

    for idx, centroid in enumerate(centroids):  
        max_similarity = -1  
        best_token_id = -1  
        #  Calculate the similarity with all candidate tokens
        for token_id, embedding in enumerate(candidate_token_embeddings):  
            # return ndarray(n_samples_X, n_samples_Y)
            similarity:np.ndarray = cosine_similarity([centroid], [embedding.detach().cpu().numpy()])[0][0]  
            if similarity > max_similarity:  
                max_similarity = similarity  
                best_token_id = token_id  
        cluster_labels.append((idx, best_token_id, max_similarity))  
        cluster_label_embeddings.append(candidate_token_embeddings[best_token_id])

 
    for idx, best_token_id, similarity in cluster_labels:  
        print(f"Cluster {idx + 1}'s best suit token：{candidate_tokens[best_token_id]}, similarity: {similarity:.4f}") 
        classes.append(candidate_tokens[best_token_id])
        
        
    return cluster_label_embeddings
        
def get_classes_by_lda(dataset_path, model, tokenizer, embedding_size, num_topics = 5, K=5, max_length=512)->List[torch.Tensor]:
    '''
    use the LDA model to extract the class labels
    
    num_topics: number of classes to be generated == num_suffix_tokens
        
    K : number of sub-classes to be generated
    '''
    print("get class labels by latent Drichtlet Allocation (LDA) Model")
    classes = []
    model = model.eval()
    device = Config['device']
    model.to(device)
    
    all_pooled_embeddings = [] # store the pooled embeddings of every 1000 examples
    
    vocab_size = tokenizer.vocab_size
    # count the word frequency of each word in vocab on the training set
    word_freq = {token_id:0 for token_id in range(vocab_size)}
    
    
    train_ds = reformat_input(dataset_path, tokenizer, max_length=max_length)
    
    train_data_loader = DataLoader(train_ds, batch_size=1000, 
                                   collate_fn=default_data_collator, 
                                   num_workers=NUM_CPU_PROCESSES, # use as your need
                                   shuffle=False)
    
    
    processed_questions: List[List[str]] = reformat_input(dataset_path, tokenizer, max_length=max_length, reformat_type = 'lda')

    print("create dictionary and corpus...")  
    # 创建词典：词语到id的映射  
    dictionary = corpora.Dictionary(processed_questions)  
    # 过滤极端词汇（可选）  
    dictionary.filter_extremes(no_below=2, no_above=0.5)  
    # 将文档转换为词袋(Bag-of-Words)表示  
    corpus: List[List[tuple]] = [dictionary.doc2bow(text) for text in processed_questions]  
        
    
    # train LDA model  
    print(f"Training LDA model, class label number = {K}...")  
    lda_model = models.ldamodel.LdaModel(  
        corpus=corpus,  
        id2word=dictionary,  
        num_topics=num_topics,  
        random_state=42,  
        passes=10,  # training epochs
        iterations=50  
    )  
  
    print("Extract the class labels...")  
    for idx, topic in lda_model.show_topics(formatted=False, num_words=K, num_topics=num_topics):  
        topic_words = [word for word, prob in topic]  
        # print(", ".join(topic_words))  
        classes.append(topic_words[0])
    
    print("class labels = ", classes)
    
    
    print("transfer class labels to label embeddings...")  
    topic_word_embeddings = []

    with torch.no_grad():  
        for id, word in enumerate(classes):  
        
            encoded_input = tokenizer(
                word, 
                padding = "max_length",
                truncation=True,  
                max_length=5,  # we set that each label can be divided into 5 tokens
                add_special_tokens=True,
                return_tensors='pt').to(device)  
            
            model_output = model(**encoded_input)  
            # 使用 [CLS] 标记的向量作为词嵌入  
            word_embedding = model_output.last_hidden_state[:, 0, :].squeeze(0)  # [hidden_size]  
            topic_word_embeddings.append(word_embedding)  


        
    return topic_word_embeddings




def get_label_collection_for_class(dataset_path, classes:List[str]):
    '''
    将原问题分类标签列表中的每个标签扩展成一个标签集合
    1. 训练一个线性层，将经过MLM的mask token分类到原有的标签
    
    return dict["class1" : set(label1, label2, ...)]
    '''

def train_bidirectional_prompt_tuning(config:BaasPromptConfig):

    
    model_name = config.model_name
    dataset_name = config.dataset_name
    model_path = config.model_path
    num_labels = config.num_labels
    batch_size = config.batch_size
    
    model,tokenizer = prepare_model_tokenizer(model_path, AutoModelForSequenceClassification, num_labels=num_labels)
    K=5
    # 定义双向Prompt Tuning的参数       
    num_prefix_tokens = config.prefix_length   # 前缀Prompt Tokens的数量  
    num_suffix_tokens = config.suffix_length   # 后缀Prompt Tokens的数量  

    lr = config.learning_rate
    num_epochs = config.num_epochs
    
    max_length = get_max_length_from_model(model)
    print(f"before inserting prompt tokens, {model_name}'s max length = {max_length}")
    max_length = max_length - num_prefix_tokens - num_suffix_tokens
    print(f"After inserting prompt tokens, {model_name}'s max length = {max_length}")

    
    
    bidirectional_prompt_model = BassPromptModel(  
        model=model,
        config=config  
    )


    
    # the preprocessed dataset only contains ["input_ids", "attention_mask", "labels"]
    
    processed_ds = preprocess_dataset_peft(dataset_name, max_length = max_length)
    
    
    train_ds = processed_ds["train"]
    eval_ds = processed_ds["test"]
    

    print("dataset is preprocessed successfully ~~~")
    # 使用DistributedSampler进行数据分布  
    train_sampler = DistributedSampler(  
        train_ds,  
        shuffle=True,  
        seed=42  
    ) if torch.distributed.is_initialized() else None 
    
    eval_sampler = DistributedSampler(  
        eval_ds,  
        shuffle=False,  
        seed=42  
    ) if torch.distributed.is_initialized() else None 
    
    
    
    train_dataloader = DataLoader(
            train_ds, 
            # shuffle=True, # shuffle is not necessary when using DistributedSampler
            collate_fn=default_data_collator, 
            batch_size=Config['batch_size'],
            pin_memory=True,
            sampler=train_sampler
        )
    
    eval_dataloader = DataLoader(
            eval_ds, 
            collate_fn=default_data_collator, 
            batch_size=Config['batch_size'],
            pin_memory=True,
            sampler=eval_sampler
        )


    
    # make sure to frozen the base model parameters
    for param in bidirectional_prompt_model.model.parameters():  
        param.requires_grad = False  
        
    # make sure that the prefix and suffix tokens is trainable
    bidirectional_prompt_model.prefix_embeddings.requires_grad = True  
    bidirectional_prompt_model.suffix_embeddings.requires_grad = True 
    
    
    bidirectional_prompt_model.print_trainable_parameters()
    
    # for name, param in bidirectional_prompt_model.named_parameters():  
    #   print(f"{name}: requires_grad = {param.requires_grad}") 
    # print("============================")
    # for param in bidirectional_prompt_model.parameters():  
    #   print(f"param = {param}: requires_grad = {param.requires_grad}")

    # make sure that the fine-tuning will only update virual tokens
    # optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, bidirectional_prompt_model.parameters()), 
        lr=lr
    )
    
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(train_dataloader) * num_epochs),
    )
    
    model = bidirectional_prompt_model 
    
    
    accelerator = Accelerator(
        # gradient_accumulation_steps=1,  
        # mixed_precision='fp16', 
    )
    
    model, optimizer, lr_scheduler, train_dataloader, eval_dataloader= accelerator.prepare(
        model, optimizer, lr_scheduler, train_dataloader, eval_dataloader)
    
    if accelerator.is_main_process:
        logging_dir = Config['logging_dir'][model_name]["bidirectional-prompt-tuning"][dataset_name]
        if not os.path.exists(logging_dir):
            os.makedirs(logging_dir)  
            print(f"已创建新的log存储路径: {logging_dir}") 
        logger = get_logger(name=__name__, logging_dir=logging_dir, log_level="INFO")
    

        
        
    global_step = 0
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for step, batch in enumerate(tqdm(train_dataloader)):
            # print(f"Batch labels: {batch['labels']}") 
            # batch = {k: v.to(device) for k, v in batch.items()}
            # batch = {"input_ids": tensor([[101, 7592, 2199, 2, ...], [101, 7592, 2199, ...]]), "attention_mask": tensor([[1, 1, 1,  ..., 0, 0, 0], [1, 1, 1, ...]])}
            
            # labels=None
            # if isinstance(batch, dict):  
            #     print(f"batch 的类型：{type(batch)}")  
            #     print(f"batch 的 keys：{batch.keys()}")  

            #     # 提取 labels，不修改原始 batch  
            #     labels = batch["labels"]  

            #     # 创建新的输入字典，不包含 labels  
            #     # 原因：在多进程环境中，最好避免对共享对象batch进行原地修改。
            #     inputs = {k: v for k, v in batch.items() if k != "labels"}  
            #     print(f"inputs 的 keys：{inputs.keys()}")  

            # else:  
            #     print("batch 不是字典类型")  
            #     continue  # 跳过非字典类型的 batch  
            
            labels = batch["labels"]  
            
            outputs = model(**batch)
            
            criterion = nn.CrossEntropyLoss()
            
            logits = outputs.logits
            
            loss = criterion(logits, labels.long())
            total_loss += loss.detach().float()

            # loss.backward()
            accelerator.backward(loss)
            
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            if step == len(train_dataloader)-1:  
                model.eval()  
                all_preds = []  
                all_labels = []  
                with torch.no_grad():  
                    for val_batch in eval_dataloader:  
                        # val_input_ids = val_batch['input_ids'].to(device)  
                        # val_attention_mask = val_batch['attention_mask'].to(device)  
                        # val_labels = val_batch['labels'].to(device)  
                        val_input_ids = val_batch['input_ids']
                        val_attention_mask = val_batch['attention_mask'] 
                        val_labels = val_batch['labels']
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
                if accelerator.is_main_process:
                    logger.info({'epoch': epoch, 'loss': loss.item(), 'accuracy':accuracy, "precision": precision, "recall": recall, "f1": f1 })  
                    # print()

                model.train()  
            global_step+=1

        avg_loss = total_loss / len(train_dataloader)   
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")  
            

    
    # 判断模型名称
    
    # model_name = get_model_name_using_model(model)
    print("model name = ", model_name)

    # 保存权重
    save_path = Config['save_model_dir'][model_name]['bidirectional-prompt-tuning'][dataset_name]
    # torch.save(model.state_dict(), save_path) 

    # wait every GPU processes to reach here
    torch.distributed.barrier()  

    # only the master process can save model
    # if torch.distributed.get_rank() == 0:  
    #     model.module.save_pretrained(save_path) 
    if not os.path.exists(save_path):
        os.makedirs(save_path)  
        print(f"已创建新的权重存储路径: {save_path}") 
        
    # accelerator.save(model.state_dict(), save_path)   

        # tokenizer.save_pretrained('path_to_save_tokenizer')   




def evaluate_bidirectional_prompt_tuning(trained_model_path, model, dataset_name = "race", eval_dataloader:DataLoader=None):
    # 初始化accelerator  
    accelerator = Accelerator()  
    
    # 获取logger  
    logger = accelerator.logging.get_logger(__name__)  
    
    # 1. 加载保存的模型权重  
    checkpoint = torch.load(trained_model_path)  
    
    # 2. 获取保存的prompt embeddings  
    prefix_prompt_embeddings = checkpoint['prefix_embeddings']  
    suffix_prompt_embeddings = checkpoint['suffix_embeddings']  
    
    # 3. 确保embeddings是Parameter类型  
    prefix_prompt_embeddings = torch.nn.Parameter(prefix_prompt_embeddings)  
    suffix_prompt_embeddings = torch.nn.Parameter(suffix_prompt_embeddings)  
    
    # 4. 创建带有双向Prompt的模型实例  
    bidirectional_prompt_model = BassPromptModel(  
        model=model,  # 假设model是从外部传入的基础模型  
        prefix_embeddings=prefix_prompt_embeddings,  
        suffix_embeddings=suffix_prompt_embeddings,  
        num_prefix_tokens=prefix_prompt_embeddings.shape[0],   
        num_suffix_tokens=suffix_prompt_embeddings.shape[0],  
    )  
    
    # 5. 使用accelerator准备模型和数据加载器  
    bidirectional_prompt_model, eval_dataloader = accelerator.prepare(  
        bidirectional_prompt_model, eval_dataloader  
    )  
    
    # 6. 设置为评估模式  
    bidirectional_prompt_model.eval()  
    
    all_preds = []  
    all_labels = []  
    
    # 7. 评估循环  
    with torch.no_grad():  
        for batch in eval_dataloader:  
            # 获取输入数据  
            input_ids = batch['input_ids']  
            attention_mask = batch['attention_mask']  
            labels = batch['labels']  
            
            # 获取token_type_ids（如果存在）  
            token_type_ids = batch.get('token_type_ids', None)  
            
            # 前向传播  
            outputs = bidirectional_prompt_model(  
                input_ids=input_ids,  
                attention_mask=attention_mask,  
                token_type_ids=token_type_ids,  
                labels=labels  
            )  
            
            # 获取预测结果  
            logits = outputs.logits  
            
            # 使用accelerator.gather收集所有进程的结果  
            gathered_logits = accelerator.gather(logits)  
            gathered_labels = accelerator.gather(labels)  
            
            # 计算预测结果  
            preds = torch.argmax(gathered_logits, dim=1).cpu().numpy()  
            labels_cpu = gathered_labels.cpu().numpy()  
            
            all_preds.extend(preds)  
            all_labels.extend(labels_cpu)  
    
    # 8. 确保只在主进程上计算和打印指标  
    if accelerator.is_main_process:  
        # 计算评价指标  
        accuracy = np.mean(np.array(all_preds) == np.array(all_labels))  
        precision, recall, f1, _ = precision_recall_fscore_support(  
            all_labels, all_preds, average='weighted'  
        )  
        
        # 记录日志  
        logger.info(  
            f"Validation Metrics:\n"  
            f"Accuracy: {accuracy:.4f}\n"  
            f"Precision: {precision:.4f}\n"  
            f"Recall: {recall:.4f}\n"  
            f"F1: {f1:.4f}"  
        )  
        
        # 使用accelerator记录指标  
        accelerator.log({  
            'eval/accuracy': accuracy,  
            "eval/precision": precision,  
            "eval/recall": recall,  
            "eval/f1": f1  
        })  
        
        return {  
            'accuracy': accuracy,  
            'precision': precision,  
            'recall': recall,  
            'f1': f1  
        }  
    
    # 非主进程返回None  
    return None



if __name__ == "__main__":
    model_name = "bert-large-uncased"
    model_path = Config["models"]["bert-large-uncased"]["model_path"]

    
    # 加载数据集
    dataset_name = "race"

    dataset_path = Config["datasets"][dataset_name]
    model, tokenizer = prepare_model_tokenizer(model_path, AutoModelForSequenceClassification)
    
    max_seq_length = get_max_length_from_model(model)

    config = BaasPromptConfig(
        model_name = model_name,
        model_path = model_path,
        dataset_name="race",
        max_seq_length=max_seq_length,
        num_epochs=5,
        num_labels=2,
    )
    train_bidirectional_prompt_tuning(config)
    
    
    
    
   
    # ds = load_dataset_from_huggingface(dataset_path,"high")
    
    
    # classes= get_classes_for_dataset(dataset_path)
    # initialize_prefix_prompts(ds, tokenizer,20, 768, classes)
    
    # classes = get_classes_for_dataset(dataset_path,model, tokenizer)
    # classes = get_classes_by_lda(dataset_path, model, tokenizer)
    # classes = get_classes_by_clustering(dataset_path, model, tokenizer, num_topics=5, K=5, max_length=512)
    
    # print("classes = ", classes)