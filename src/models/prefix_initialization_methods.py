import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    AutoConfig,
)
from peft import get_peft_model, LoraConfig, TaskType
import os
import numpy as np
from typing import List, Optional, Union, Dict, Optional, Callable
from tqdm import tqdm
from datasets import load_dataset




import nltk  
# nltk.download('punkt') 
# nltk.download('punkt_tab') 
# nltk.download('stopwords')  
from nltk.corpus import stopwords  
from nltk.tokenize import word_tokenize
stop_words = set(stopwords.words('english')) 


from src.models.rgc_models import (
    BaasPromptConfig
)


'''

本文件的作用：
    用 clustering-based, TF-QDF-VS, LDA 等方法来初始化 prefix tokens。

    1. 用 clustering-based 方法来初始化 prefix tokens
        1.1 用 TF-QDF-VS 方法来计算每个 sample 的 embedding
        1.2 用 k-means 方法来聚类这些 embeddings
        
    2. 用 LDA 方法来初始化 prefix tokens
        2.1 用 LDA 方法来计算每个 sample 的 topic distribution
        2.2 用 k-means 方法来聚类这些 topic distributions
        
    3. 用 TF-QDF-VS 方法来初始化 prefix tokens
        3.1 用 TF-QDF-VS 方法来计算每个 sample 的 embedding
        3.2 用 k-means 方法来聚类这些 embeddings
        
        
'''



def reformat_input(config:BaasPromptConfig, tokenizer, reformat_type = "normal"):
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
    
    input_key = 'input'
    
    max_length = config.max_seq_length
    
    print(f"reformat input using reformat_type = {reformat_dict[reformat_type]}")
    wrapper = McqDatasetWrapper()
    dataset_configs = wrapper.dataset_configs
    
    if config.dataset_name == 'race' or config.dataset_name == 'sciq' \
        or config.dataset_name == 'dream' or config.dataset_name == 'commonsense_qa':
        
        
        article_key = dataset_configs[config.dataset_name].article_key
        question_key = dataset_configs[config.dataset_name].question_key
        options_key = dataset_configs[config.dataset_name].options_key
        label_key = dataset_configs[config.dataset_name].label_key
        
        
        
        if reformat_type == "normal": 
            # ds,_ = wrapper.load_mcq_dataset(config.dataset_name, train_size=5000) 
            # # processed_ds = preprocess_dataset_peft(dataset_name, max_length = max_length)
            # ds = ds["train"]
            
            ds = preprocess_dataset_peft(config.dataset_name,config.model_path,config.max_seq_length, seq_cls_type=config.seq_cls_type, train_size=500)

            train_ds = ds['train']
            
        elif reformat_type == "lda":
            ds,_ = wrapper.load_mcq_dataset(config.dataset_name, train_size=5000)
            ds = ds["train"]
            processed_ds = ds.map(
                lambda examples: {
                    input_key: [f"{article_key}:{examples[article_key][index]}\n\n{question_key}:{examples[question_key][index]}\n\n \
                                            {options_key}:{examples[options_key][index]}\n\n{label_key}:" for index, x in enumerate(examples[article_key])]  
                },
                batched=True,
                num_proc=NUM_CPU_PROCESSES,
                remove_columns=[article_key, question_key, options_key],
                load_from_cache_file=False,
                desc=f"Running reformat function's mapping on dataset {config.dataset_name}",
            )
            
            # transfer the dataset into List[str]
            processed_ds:List[str] = [text for text in processed_ds[input_key]]  
            
            print("processed_ds[:10] = \n", processed_ds[:10])
            print("*******************************************")
            
            
            train_ds: List[List[str]] = []
            for text in processed_ds:
                text = text.lower()
                tokens = word_tokenize(text)
                # remove stopwords and non-alphabetic characters
                tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
                train_ds.append(tokens)
                
            print("train_ds[:10] = \n", train_ds[:10])
            
        elif reformat_type == "cluster":
            # processed_ds = preprocess_dataset_peft(dataset_name, max_length = max_length)
            # train_ds = processed_ds["train"]
            ds,_ = wrapper.load_mcq_dataset(config.dataset_name)
            
            # print("load_mcq_dataset ds[0]", ds[0])
            print("type(ds) = ",type(ds))
            train_ds = ds["train"]
            
            train_ds = train_ds.map(
                lambda examples: {
                    input_key: [f"{article_key}:{examples[article_key][index]}\n\n{question_key}:{examples[question_key][index]}\n\n \
                                            {options_key}:{examples[options_key][index]}\n\n{label_key}:" for index, x in enumerate(examples[article_key])]  
                },
                batched=True,
                num_proc=NUM_CPU_PROCESSES,
                remove_columns=[article_key, question_key, options_key],
                load_from_cache_file=False,
                desc=f"Running reformat function's mapping on dataset {config.dataset_name}",
            )
            
            if input_key not in train_ds.column_names:  
                raise KeyError(  
                    f"Failed to create 'input' column. "  
                    f"Current columns: {train_ds.column_names}"  
                )  
            # train_ds = processed_ds["train"]
        
        else:
            raise ValueError("Invalid reformat_type, please choose from 'normal', 'lda', 'cluster")
        
    else:
        raise ValueError(f"dataset_name: {config.dataset_name} not supported, we can not reformat dataset using a wrong name, please change another in [race, sciq, commonsense_qa, dream]")
    
    return train_ds, input_key




def get_classes_for_dataset(dataset_path, model, tokenizer, config, embedding_size, num_topics = 5, K=5, max_length=512)->List[torch.Tensor]:
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
    model:AutoModelForSequenceClassification = model.eval()
    device = Config.device
    
    model.to(device)
    
    vocab = tokenizer.get_vocab()
    inverse_vocab = {v:k for k,v in vocab.items()} # index-token mapping
    
    all_pooled_embeddings = [] # store the pooled embeddings of every 1000 examples
    
    vocab_size = tokenizer.vocab_size
    # count the word frequency of each word in vocab on the training set
    word_freq = {token_id:0 for token_id in range(vocab_size)}
    
    
    train_ds,_ = reformat_input(config, tokenizer, reformat_type='normal')
    print("len(train_ds) = ", len(train_ds))
    print("vocab_size = ", vocab_size)
    
    train_data_loader = DataLoader(
                                   train_ds, 
                                   batch_size=10, 
                                   collate_fn=default_data_collator, 
                                   num_workers=NUM_PROCESSES,
                                   shuffle=False
                                   )
    
    print("dataset {} loaded ~~".format(config.dataset_name))
    
    # record whether this token occurs in this example or not, 1 for yes, 0 for no
    word_occurence_per_example:Dict[int, Dict[int, int]] = {
                                                                token_id:{
                                                                    example_id:0 
                                                                        for example_id in range(len(train_ds))
                                                                    } 
                                                                for token_id in range(vocab_size)
                                                            }
    print("calculate word frequency ~~~")
    
    with torch.no_grad():
        global_step = 0
        for index, batch in enumerate(tqdm(train_data_loader)):
            # get the average embedding of each batch
            batch = {k:v.to(device) for k, v in batch.items()}
            
            if index==0:
                print("batch.keys() = ", batch.keys())
            
            input_ids = batch['input_ids']
            input_ids_np = input_ids.cpu().numpy()
            for ids in input_ids_np:  
                for id in ids:
                    id = int(id)  
                    word_freq[id] += 1
                    word_occurence_per_example[id][global_step] = 1
                
                global_step += 1
            
            # outputs = model(**batch)
            # embeddings = model.embeddings(input_ids) # shape = (batch_size, seq_len, hidden_size)
            embeddings= model.get_input_embeddings()(input_ids)
            # firstly, pooling on the seq_len dimension
            pooled_embedding = embeddings.mean(dim=1) # shape = (batch_size, hidden_size)
            # then, pooling on the batch_size dimension
            # all_pooled_embeddings.append(pooled_embedding.mean(dim=0).cpu()) # shape = (hidden_size)
            all_pooled_embeddings.append(pooled_embedding.mean(dim=0).unsqueeze(0)) # shape = (1, hidden_size)
            
    print("all_pooled_embeddings loaded ~")
    
    # print("all_pooled_embeddings = ", all_pooled_embeddings)
    print("=================================")
    print("all_pooled_embeddings[0].shape = ", all_pooled_embeddings[0].shape)
    print("all_pooled_embeddings[10].shape = ", all_pooled_embeddings[10].shape)
    print("all_pooled_embeddings[20].shape = ", all_pooled_embeddings[20].shape)
    
    
    print("all_pooled_embeddings.length = ", len(all_pooled_embeddings))
    print("len(train_data) = ", len(train_data_loader)) # 88
    print("len(train_data)//1000 = ", len(train_data_loader)//10) # 0
    print("==================================")
    # vertically stack all the pooled embeddings into a matrix
    # "dim" points to the dimension of each pooled embedding
    pooled_embeddings_matrix = torch.cat(all_pooled_embeddings, dim=0) # shape = (len(train)//1000, hidden_size)

    num_pooled = len(train_ds)//10
    if num_pooled == 0:
        num_pooled = 1
        
    vocab_size =  tokenizer.vocab_size
    token_embeddings = model.base_model.embeddings.word_embeddings.weight # [vocab_size, hidden_size]
    
    
    print("pooled embeddings matrix shape: ", pooled_embeddings_matrix.shape)
    print("token embeddings matrix shape: ", token_embeddings.shape)
    
    # 计算池化嵌入与词汇表中每个词嵌入的余弦相似度，并取平均值
    all_similarities_token_to_corpus = []
    for token_embedding in tqdm(token_embeddings):
        # pooled_embedding = pooled_embedding.squeeze(0) # shape = (hidden_size)
        cosine_similarities = F.cosine_similarity(
            x1=token_embedding,
            x2 =pooled_embeddings_matrix,
            dim=1 # 代表我们在行的维度上进行计算
        ) # shape = (num_pooled)
        
        all_similarities_token_to_corpus.append(cosine_similarities.mean(dim=0).item()) # shape = 

    all_similarities_token_to_corpus = torch.tensor(all_similarities_token_to_corpus)
    print("all_similarities_token_to_corpus.shape = ", all_similarities_token_to_corpus.shape)
    
    # avg_similarities = cosine_similarities.mean(dim=0) # shape = (vocab_size)
    
    
    # transfer the word frequency to tensor
    freq_tensor = torch.zeros(vocab_size) 
    for token_id, freq in word_freq.items():  
        if token_id < vocab_size:  
            freq_tensor[token_id] = freq  
    
    # store the total occurence number of each token on all examples
    word_occurence_total = [0]*vocab_size
    
    # calculate the total occurence in all examples of each token
    for token_id in range(vocab_size):
        for example_id in range(len(train_ds)):
            tmp = word_occurence_per_example[token_id][example_id]
            word_occurence_total[token_id] += tmp
            
    word_occurence_total_tensor = torch.tensor(word_occurence_total, dtype=torch.float)  
    normalized_occurence = word_occurence_total_tensor / len(train_ds)  
    # calculate TF-IQF-VS score
    tf_iqf_vs_scores:torch.Tensor = all_similarities_token_to_corpus * freq_tensor * normalized_occurence # torch.tensor(word_occurence_total/len(train_ds))
    
    # filter special tokens
    filtered_scores = []
    for token_id, score in enumerate(tf_iqf_vs_scores):
        token = inverse_vocab[token_id]
        if token.strip() and not token.startswith("[") and token.isalpha() and token not in stop_words:
            filtered_scores.append((token_id, score))
    # filtered_scores = torch.tensor(filtered_scores) # shape = (num_tokens, 2)
    
    print("len(filtered_scores) = ", len(filtered_scores))
    print("filtered_scores[:5] = ", filtered_scores[:5])
    print("num_topics = ",num_topics)
    
    # choose Top-K as class labels
    # topk_scores, topk_indices = filtered_scores.topk(num_topics, dim=1)
    # print("Top-k scores:", topk_scores)
    # print("Top-k indices:", topk_indices)
    filtered_scores.sort(key=lambda x: x[1], reverse=True)  
    
    
    sorted_indices = []
    sorted_scores = []
    for idx, pair in enumerate(filtered_scores):
        sorted_indices.append(pair[0])
        sorted_scores.append(pair[1])
    
    print("sorted_indices = ",sorted_indices)
    print("sorted_scores = ",sorted_scores)
    
    topk_indices = sorted_indices[:num_topics]  
    
    for idx in topk_indices:    
        idx = int(idx)
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
    model: AutoModelForSequenceClassification, 
    tokenizer, 
    config:BaasPromptConfig,
    embedding_size,  # hidden_size
    num_topics=5,  
    max_length=512, 
    use_trained_embeddings=True,
    cache_dir='./class_label_cluster_embeddings'  # 存储embeddings的目录  
    )->List[torch.Tensor]:
    '''
    Args:
        num_topics: number of classes to be generated == num_suffix_tokens
        
        K : number of sub-classes to be generated
        
        cache_dir: 存储embeddings的目录  
        use_trained_embeddings: 是否使用已缓存的embeddings  
    
    return 
        class_embeddings: a list of tensor embeddings, each embedding corresponds to a label
    
    '''
    print(f"***************** Get Class Labels by Clustering ~~~~ *********************88")
    os.makedirs(cache_dir, exist_ok=True)
    device = Config.device
    classes = []
    model = model.to(device)
    
    
    # 生成唯一的缓存文件名（基于模型名称和数据集路径）  
    model_name = get_model_name_using_model(model)
    dataset_name = os.path.basename(dataset_path) 
    cache_filename = f"label_embeddings_{dataset_name}_{embedding_size}.pt"  
    cache_path = os.path.join(cache_dir, cache_filename) 
    
    
    
    # 直接加载最终结果，有的话直接返回
    final_label_embeddings_filename = f"final_label_embeddings_{dataset_name}_{embedding_size}.pt"
    final_label_metadata_filename = f"final_label_metadata_{dataset_name}_{embedding_size}.pt"
    final_label_embeddings_path = os.path.join(cache_dir, 'final_embeddings',final_label_embeddings_filename)
    final_label_metadata_path = os.path.join(cache_dir, 'final_embeddings', final_label_metadata_filename)
    
    
    final_metadata = {
        'dataset_path': dataset_path,  
        'model_name': model_name,  
        'embedding_size': embedding_size  
    }
    
    if os.path.exists(final_label_embeddings_path) and os.path.exists(final_label_metadata_path):

        # 验证metadata
        with open(final_label_metadata_path, 'r') as f:  
                cached_final_metadata = json.load(f)  
                
        if all(cached_final_metadata[k] == final_metadata[k] for k in final_metadata.keys() if k != 'max_length'):  
            print(f"Loading final label embeddings from {final_label_embeddings_path}")
            device,local_rank = setup_distributed()
            cluster_label_embeddings = torch.load(final_label_embeddings_path, map_location=device)  
            print(f"Loaded final label embeddings shape: {cluster_label_embeddings.shape}")
            return cluster_label_embeddings
        else:  
            print("============= cached final meta data ========================")
            print(cached_final_metadata)
            print()
            print("============== final meta data ================================")
            print(final_metadata)
            print()
            print("Sentence embedding's cache final metadata mismatch, recomputing final label embeddings for clustering...") 
            print()

    # print(f"Loading final label embeddings from {final_label_embeddings_path}")
    # cluster_label_embeddings = torch.load(final_label_embeddings_path)  
    # print(f"Loaded final label embeddings shape: {cluster_label_embeddings.shape}")
    # return cluster_label_embeddings
    
    
  
    # 保存数据集信息，用于验证sentence embedding的缓存是否匹配  
    metadata = {  
        'dataset_path': dataset_path,  
        'model_name': model_name,  
        'embedding_size': embedding_size  
    }  
    metadata_path = os.path.join(cache_dir, f"{cache_filename}_metadata_{embedding_size}.json")  
    
    embeddings = None

    # 如果启用缓存且缓存文件存在，尝试加载缓存的embeddings  
    if use_trained_embeddings and os.path.exists(cache_path) and os.path.exists(metadata_path):  
        try:
            # 首先验证metadata是否匹配  
            with open(metadata_path, 'r') as f:  
                cached_metadata = json.load(f)  
                
            if all(cached_metadata[k] == metadata[k] for k in metadata.keys() if k != 'max_length'):  
                print(f"Loading cached embeddings from {cache_path}")  
                # 使用torch.load加载缓存的embeddings  
                embeddings = torch.load(cache_path)  
                print(f"Loaded embeddings shape: {embeddings.shape}")
                # return process_embeddings(embeddings, num_topics, K)  # 假设有这个后处理函数[聚类逻辑]
                
            else:  
                print("============= cached meta data ========================")
                print(cached_metadata)
                print()
                print("============== meta data ================================")
                print(metadata)
                raise RuntimeError("Sentence embedding's cache metadata mismatch, recomputing embeddings for clustering...") 

        except (json.JSONDecodeError, FileNotFoundError, RuntimeError) as e:
            print(f"Error, when loading cache: {e}, meta data = {metadata}. Recomputing embeddings...")


        # embeddings = torch.load(cache_path)  
        
    # else:
    train_ds, input_key = reformat_input(config, tokenizer, reformat_type = 'cluster')
    train_ds: Dict[str,List[str]]
    
    # 验证列名是否存在  
    if input_key not in train_ds.column_names:  
        available_columns = train_ds.column_names  
        raise KeyError(  
            f"Column '{input_key}' not found in dataset. "  
            f"Available columns are: {available_columns}. "  
            f"Please check your dataset structure or specify the correct input_key."  
        )  
    
    sentences = train_ds[input_key]
    
    print(f"The training data is reformated to only one column {input_key}, now we get each example's embedding using the SentenceTransformer~~~")
    
    encoder = SentenceEncoder(
        hidden_size=embedding_size
    ).to(device)
    
    embeddings = encoder.encode(
        sentences=sentences
    ) # shape = (dataset_size, hidden_size)
            
    
    print("all_embeddings.shape = ", embeddings.shape)
        
    
    # 保存embeddings到缓存  
    print(f"Saving new embeddings to {cache_path}")  
    torch.save(embeddings, cache_path)  
    
    # 保存metadata  
    with open(metadata_path, 'w') as f:  
        json.dump(metadata, f) 
    
    
    if embeddings ==None:
        raise RuntimeError("Sentence embeddings for clustering is None, created failed, please check the code")

    # no matter how you get embeddings (cache/model infer), we need to convert it to numpy array for clustering
    embeddings = embeddings.cpu().numpy() 
             
    # clustering
    print(f"doing K-Means clustering, cluster number = {num_topics}")
    
    kmeans = KMeans(n_clusters=num_topics, random_state=42)
    kmeans.fit(embeddings)
    
    centroids = kmeans.cluster_centers_ # shape = ndarray(n_clusters, n_features)
    
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
    # model_unwrapped = accelerator.unwrap_model(model)
    # model_unwrapped.eval() # save resources
    
    
    # 加入判断逻辑
    # candidate_token_embeddings = model_unwrapped.embeddings(candidate_token_ids) # shape = (batch_size, 1, hidden_size) == (num_tokens, 1, hidden_size)
    candidate_token_embeddings = get_vocab_embeddings_from_model(model, candidate_token_ids).to(device)
    candidate_token_embeddings =  candidate_token_embeddings.squeeze(1) # shape = (num_tokens, hidden_size)
    
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
        
    cluster_label_embeddings = torch.concat(cluster_label_embeddings, dim=0) # shape = (num_topics, hidden_size)

    if not os.path.exists(os.path.dirname(final_label_embeddings_path)):
        os.makedirs(os.path.dirname(final_label_embeddings_path))
        
    print(f"Saving final label embeddings to {final_label_embeddings_path}")  
    torch.save(cluster_label_embeddings, final_label_embeddings_path) 
    
    # 保存元信息 
    with open(final_label_metadata_path, 'w') as f:  
            json.dump(final_metadata, f) 
        
    return cluster_label_embeddings
        
def get_classes_by_lda(dataset_path, model, tokenizer, config, embedding_size, num_topics = 5, K=1, max_length=512)->torch.Tensor:
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
    
    
    # train_ds = reformat_input(config, tokenizer, reformat_type="lda")
    
    # train_data_loader = DataLoader(train_ds, batch_size=1000, 
    #                                collate_fn=default_data_collator, 
    #                                num_workers=NUM_CPU_PROCESSES, # use as your need
    #                                shuffle=False)
    
    
    processed_questions,_ = reformat_input(config, tokenizer, reformat_type = 'lda')
    processed_questions:List[List[str]]
    print(f"processed question type = ", type(processed_questions))

    print("create dictionary and corpus...")  
    # 创建词典：词语到id的映射  
    dictionary = corpora.Dictionary(processed_questions)  
    # 过滤极端词汇（可选）  
    dictionary.filter_extremes(no_below=2, no_above=0.5)  
    # 将文档转换为词袋(Bag-of-Words)表示  
    corpus: List[List[tuple]] = [dictionary.doc2bow(text) for text in processed_questions]  
        
    
    # train LDA model  
    print(f"Training LDA model, class label number = {K}...")  
    lda_model = gensim.models.LdaModel( 
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
    
    encoder = SentenceEncoder(
        hidden_size=embedding_size
    ).to(device)
    
    for id, word in enumerate(classes): 
        encoded_topic = encoder.encode(
            sentences=word,
        )
        topic_word_embeddings.append(encoded_topic)
    
    # 转为二维张量
    topic_word_embeddings = torch.stack(topic_word_embeddings, dim=0)

    

    # with torch.no_grad():  
    #     for id, word in enumerate(classes):  
        
    #         encoded_input = tokenizer(
    #             word, 
    #             padding = "max_length",
    #             truncation=True,  
    #             max_length=5,  # we set that each label can be divided into 5 tokens
    #             add_special_tokens=True,
    #             return_tensors='pt').to(device)  
            
    #         model_output = model(**encoded_input)  
    #         # 使用 [CLS] 标记的向量作为词嵌入  
    #         word_embedding = model_output.last_hidden_state[:, 0, :].squeeze(0)  # [hidden_size]  
    #         topic_word_embeddings.append(word_embedding)  


        
    return topic_word_embeddings
