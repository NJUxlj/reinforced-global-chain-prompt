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

from load import (
    preprocess_function_race_pt, 
    preprocess_function_race,
    load_dataset_from_huggingface,
    preprocess_race,
)

from training_hub import (
    prepare_model_tokenizer, 
)

from datasets import (
    # Dataset,
    load_dataset,
)
from torch.utils.data import (
    DataLoader,
    Dataset
)

from config import Config

import torch
import torch.nn as nn
import torch.nn.functional as F  
import numpy as np
import csv
import evaluate
import gensim
from gensim import corpora, models
import nltk
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics.pairwise import cosine_similarity  

from sklearn.cluster import KMeans
from typing import List, Dict
from collections import defaultdict


import nltk  
nltk.download('punkt') 
nltk.download('punkt_tab') 
nltk.download('stopwords')  
from nltk.corpus import stopwords  
from nltk.tokenize import word_tokenize
stop_words = set(stopwords.words('english')) 




device = Config['device']

# # 初始化模型  
# model_path = Config["models"]["bert-base-uncased"]["model_path"]
# model = AutoModelForSequenceClassification.from_pretrained(model_path).cuda()  
# tokenizer = AutoTokenizer.from_pretrained(model_path)





# 3. 修改模型的输入嵌入函数，添加前缀和后缀Prompt Tokens  
class BidirectionalPromptModel(torch.nn.Module):  
    def __init__(self, model, prefix_embeddings, suffix_embeddings, num_prefix_tokens, num_suffix_tokens):  
        super(BidirectionalPromptModel, self).__init__()  
        self.model = model  
        self.prefix_embeddings = prefix_embeddings  
        self.suffix_embeddings = suffix_embeddings 
        self.num_prefix_tokens = num_prefix_tokens
        self.num_suffix_tokens = num_suffix_tokens
        self.embedding_layer = self.model.get_input_embeddings()  
    
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):  
        # 原始输入嵌入
        # print(f"input_ids.shape = {input_ids.shape}")
        input_ids = input_ids.squeeze(1) 
        inputs_embeds = self.embedding_layer(input_ids)  
        
        batch_size = inputs_embeds.size(0)  
        
        # 将前缀和后缀Prompt Embeddings扩展到batch维度  
        prefix_embeds = self.prefix_embeddings.unsqueeze(0).expand(batch_size, -1, -1)  
        suffix_embeds = self.suffix_embeddings.unsqueeze(0).expand(batch_size, -1, -1)  
        
        # print(f"prefix.shape = {prefix_embeds.shape}")
        # print(f"suffix.shape = {suffix_embeds.shape}")
        # print(f"inputs_embeds.shape = {inputs_embeds.shape}")

        # 拼接前缀、原始输入和后缀嵌入  
        inputs_embeds = torch.cat([prefix_embeds, inputs_embeds, suffix_embeds], dim=1)  # (4, 522, 768)
        
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





def initialize_suffix_prompts(ds, tokenizer, num_suffix_tokens, embedding_size):  
    """  
     use Auto-CoT reasoning steps as the suffix prompts
    """  
    # 假设我们已经有AutoCoT生成的推理步骤steps，并将其保存为文本列表  
    steps_list: List[str] = []  
    
    for sample in tqdm(ds['train']):
        # generate AutoCoT-style input  
        text = f"Article: {sample['article']}\n\nQuestion: {sample['question']}\n\nOptions: {', '.join(sample['options'])}\n\nAnswer:" 
        
        # use pretrained model to generate reasoning steps as a string
        steps = "Generated reasoning steps for the sample."  
    
    
    suffix_prompt_embeddings = torch.tensor(cluster_centers, requires_grad=True).cuda()  
    return suffix_prompt_embeddings
    



def initialize_prefix_prompts(dataset_path, model, tokenizer, num_prefix_tokens, embedding_size, classes_initiate_method = "cluster", K=5): 
    """ 
    use article classification tokens' weighted sum as prefix prompts
    """
    class_embeddings = None
    if classes_initiate_method == "normal":
        class_embeddings = get_classes_for_dataset(dataset_path,model, tokenizer, num_topics=num_prefix_tokens, K=5, max_length=512)
    elif classes_initiate_method == "cluster":
        class_embeddings = get_classes_by_clustering(dataset_path,model, tokenizer, num_topics=num_prefix_tokens, K=5, max_length=512)
    elif classes_initiate_method == "lda":
        class_embeddings = get_classes_by_lda(dataset_path, model, tokenizer, num_topics=num_prefix_tokens, K=5, max_length=512)
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
    if dataset_path == Config["datasets"]["race"] or \
            dataset_path == Config["datasets"]["race-m"] or \
                dataset_path == Config["datasets"]["race-h"]:
                    
        ds = load_dataset_from_huggingface(dataset_path, "all")
        
        if reformat_type == "normal":
            # corse-grained preprocessing
            ds, classes, tokenizer = preprocess_race(ds, tokenizer)
            
            # fine-grained preprocessing
            processed_ds = ds.map(
                lambda examples: preprocess_function_race(examples, max_length=max_length, tokenizer=tokenizer), 
                batched=True,
                num_proc=1,
                remove_columns=ds['train'].column_names,
                load_from_cache_file=False,
                desc="Running tokenizer on dataset",
            )
            train_ds = processed_ds["train"]
            
        elif reformat_type == "lda":
            processed_ds = ds.map(
                lambda examples: {
                    "combined_input": [f"Artical:{examples['article'][index]}\n\nQuestion:{examples['question'][index]}\n\n \
                                            Options:{examples['options'][index]}\n\nAnswer:" for index, x in enumerate(examples['article'])]  
                },
                batched=True,
                num_proc=1,
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
            pass
        else:
            raise ValueError("Invalid reformat_type, please choose from 'normal', 'lda', 'cluster")
        
                    
        
    elif dataset_path == Config["datasets"]["multirc"]:
        pass
    elif dataset_path == Config["datasets"]["arc"]:
        pass
    elif dataset_path == Config["datasets"]["dream"]:
        pass
    else:
        raise ValueError("dataset_path not supported, we can not reformat dataset using a wrong name, please change another in [race, race-m, race-h, multirc, arc]")
    
    return train_ds

def get_classes_for_dataset(dataset_path, model, tokenizer, num_topics = 5, K=5, max_length=512):
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
    
    # calculate TF-IQF-AS score
    tf_as_scores = all_similarities_token_to_corpus * freq_tensor * torch.tensor(word_occurence_total/len(train_ds))
    
    
    # choose Top-K as class labels
    topk_scores, topk_indices = tf_as_scores.topk(num_topics)
    for idx in topk_indices:    
        classes.append(tokenizer.decode(idx)) # here idx can be a integer or list of integers

    # remove special possible characters
    tmp_classes=[]
    for label in classes:
        if label.strip() and not label.startswith("["):
            tmp_classes.append(label.strip())
    classes = tmp_classes   
    
    # get the label collections expanded from the class labels
 
    
    return classes

def get_classes_by_clustering(dataset_path, model, tokenizer, num_topics=5, K=5, max_length=512)->List[torch.Tensor]:
    '''
        num_topics: number of classes to be generated == num_suffix_tokens
        
        K : number of sub-classes to be generated
    
    '''

    classes = []
    model = model.eval()
    device = Config['device']
    model.to(device)
    
    all_pooled_embeddings = [] # store the pooled embeddings
    
    vocab_size = tokenizer.vocab_size

    
    train_ds: Dict[str,torch.Tensor[List]] = reformat_input(dataset_path, tokenizer, max_length=max_length, reformat_type = 'normal')
    
    
    train_data_loader = DataLoader(train_ds, batch_size=32, 
                                   collate_fn=default_data_collator, 
                                   num_workers=0, # use as your need
                                   shuffle=False)
    
    all_batch_embeddings = []
    with torch.no_grad():
        for index, batch in enumerate(tqdm(train_data_loader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            del batch['labels']
            
            outputs = model(**batch)
            
            # average pooling 
            batch_embeddings = outputs.last_hidden_state.mean(1) # shape = (batch_size,  hidden_size)
            
            print("batch_embeddings.shape = ", batch_embeddings.shape)
            for embedding in batch_embeddings:
                all_batch_embeddings.append(embedding.unsqueeze(0)) # shape = (1, hidden_size)
                print("embedding.shape = ", embedding.shape)
            
            
            
    embeddings = torch.cat(all_batch_embeddings, dim=0)
    print("all_embeddings.shape = ", embeddings.shape)
    
    embeddings = embeddings.detach().cpu().numpy() 
    
    
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
    
    candidate_token_ids = torch.tensor(candidate_token_ids, dtype=torch.long).to(device).unsqueeze(1)
    candidate_token_embeddings = model.embeddings(candidate_token_ids) # shape = (batch_size, 1, hidden_size)
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
        
def get_classes_by_lda(dataset_path, model, tokenizer, num_topics = 5, K=5, max_length=512):
    '''
    use the LDA model to extract the class labels
    
    num_topics: number of classes to be generated == num_suffix_tokens
        
    K : number of sub-classes to be generated
    '''
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
                                   num_workers=0, # use as your need
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
        
    return classes




def get_label_collection_for_class(dataset_path, classes:List[str]):
    '''
    将原问题分类标签列表中的每个标签扩展成一个标签集合
    1. 训练一个线性层，将经过MLM的mask token分类到原有的标签
    
    return dict["class1" : set(label1, label2, ...)]
    '''

def train_bidirectional_prompt_tuning(model, tokenizer):
    device = Config['device']

    K=5
    # 定义双向Prompt Tuning的参数       
    num_prefix_tokens = 5   # 前缀Prompt Tokens的数量  
    num_suffix_tokens = 5   # 后缀Prompt Tokens的数量  
    embedding_size = model.get_input_embeddings().embedding_dim

    batch_size = Config['batch_size']
    lr = 3e-2
    num_epochs = Config['num_epochs']
    max_length = 512 - num_prefix_tokens - num_suffix_tokens


    # 1. initialize the trainable prefix prompt embeddings
    # prefix_prompt_embeddings = torch.nn.Parameter(
    #     torch.rand(num_prefix_tokens, embedding_size,requires_grad=True, device= device),   # (num_prefix_tokens, embedding_size)
    # )
    
    dataset_path = Config["datasets"][dataset_name]
    
    prefix_prompt_embeddings = initialize_prefix_prompts(dataset_path, model, tokenizer, 
                                                         num_prefix_tokens=num_prefix_tokens, 
                                                         embedding_size=embedding_size, classes_initiate_method="cluster", K=5)

    # 2. initialize the trainable suffix prompt embeddings
    suffix_prompt_embeddings  = torch.nn.Parameter(  
        torch.rand(num_suffix_tokens, embedding_size,requires_grad=True, device= device),   # (num_suffix_tokens, embedding_size)
    )

    # 4. 创建带有双向Prompt的模型实例  
    bidirectional_prompt_model = BidirectionalPromptModel(  
        model=model,  
        prefix_embeddings=prefix_prompt_embeddings,  
        suffix_embeddings=suffix_prompt_embeddings,
        num_prefix_tokens=num_prefix_tokens, 
        num_suffix_tokens=num_suffix_tokens,
    ).to(device)  

    # 加载数据集
    dataset_name = "race"

    
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
    
    
    accelerator = Accelerator()
    model, optimizer, train_dataloader, lr_scheduler= accelerator.prepare(
        model, optimizer, lr_scheduler, train_dataloader)

    global_step = 0
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for step, batch in enumerate(tqdm(train_dataloader)):
            # print(f"Batch labels: {batch['labels']}") 
            batch = {k: v.to(device) for k, v in batch.items()}
            # batch = {"input_ids": tensor([[101, 7592, 2199, 2, ...], [101, 7592, 2199, ...]]), "attention_mask": tensor([[1, 1, 1,  ..., 0, 0, 0], [1, 1, 1, ...]])}
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.detach().float()

            # loss.backward()
            accelerator.backward(loss)
            
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            # evaluate for each 5 batch-steps
            if step == len(train_dataloader)-1:  
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
            global_step+=1

        avg_loss = total_loss / len(train_dataloader)   
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")  
            


    # 保存权重
    save_path = Config['save_model_dir']['bert-base-uncased']['bidirectional-prompt-tuning']['race']
    # torch.save(model.state_dict(), save_path) 


    model.save_pretrained(save_path)
    # tokenizer.save_pretrained('path_to_save_tokenizer')   









if __name__ == "__main__":
    
    model_path = Config["models"]["bert-base-uncased"]["model_path"]

    
    # 加载数据集
    dataset_name = "race"

    dataset_path = Config["datasets"][dataset_name]
    model, tokenizer = prepare_model_tokenizer(model_path, AutoModel)
    tokenizer = BertTokenizerFast.from_pretrained(model_path)
    

    train_bidirectional_prompt_tuning(model, tokenizer)
    
    
    
    
   
    # ds = load_dataset_from_huggingface(dataset_path,"high")
    
    
    # classes= get_classes_for_dataset(dataset_path)
    # initialize_prefix_prompts(ds, tokenizer,20, 768, classes)
    
    # classes = get_classes_for_dataset(dataset_path,model, tokenizer)
    # classes = get_classes_by_lda(dataset_path, model, tokenizer)
    # classes = get_classes_by_clustering(dataset_path, model, tokenizer, num_topics=5, K=5, max_length=512)
    
    # print("classes = ", classes)