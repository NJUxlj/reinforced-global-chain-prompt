
# 获取当前文件所在目录的父目录  
import os, sys
parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
# 将父目录添加到sys.path  
sys.path.insert(0, parent_directory) 

sub_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, sub_directory) 

import argparse
from autocot_utils import *
from autocot_models import *
from pathlib import Path
from sentence_transformers import SentenceTransformer, models
import torch
import torch.nn as nn
import numpy as np

from transformers import (
    AutoModelForSequenceClassification
)



from config import Config
from config import NUM_CPU_PROCESSES
from config import SENTENCE_TRANSFORMER_PATH
from utils import setup_distributed


from causal_modeling import RollbackDecoderWithHead

from dataclasses import dataclass





'''
python make_embeddings.py \
--dataset race \
--encoder all-mpnet-base-v2 \
--hidden_size 768 \
--method auto_cot \
--output_dir experiment/race \
--max_length_cot 2048 \
--embedding_dir ./embeddings/race   

'''


'''
models = [  
    'all-MiniLM-L6-v2',        # 应该是384  
    'all-mpnet-base-v2',       # 应该是768  
    'bert-base-nli-mean-tokens'  # 应该是768  
]  
'''

@dataclass
class ChainEncodingArguments:
    """
    解析参数的类
    """
    random_seed:int = 42
    dataset:str = 'race'
    # encoder:str = "all-MiniLM-L6-v2"
    encoder:str = SENTENCE_TRANSFORMER_PATH
    hidden_size:int = 768   # 根据模型传进来
    method:str = "auto_cot"
    output_dir:str = "experiment/race"
    max_ra_len:int = 20
    max_length_cot:int = 2048
    max_length_direct:int = 32
    limit_dataset_size:int = 0
    api_time_interval:float = 1.0
    temperature:float = 0
    embedding_dir:str = "./embeddings/race"
    context_dir:str = "./context/race"
    direct_answer_trigger = "\nTherefore, the answer is"
    direct_answer_trigger_for_zeroshot_cot = direct_answer_trigger
    cot_trigger = "Let's think step by step."
    


def parse_arguments():
    
    parser = argparse.ArgumentParser(description="Zero-shot-CoT")
    
    parser.add_argument("--random_seed", type=int, default=42, help="random seed")
    
    parser.add_argument(
        "--dataset", type=str, default="race", choices=["race", "dream", "sciq", "commonsense_qa"], help="dataset used for experiment, select from [ race, dream, sciq, commonsense_qa ]"
    )
    
    parser.add_argument(
        "--encoder", type=str, default="all-MiniLM-L6-v2", help="which sentence-transformer encoder for reasoning chain encoding, select from, [all-MiniLM-L6-v2, all-mpnet-base-v2, roberta-large-nli-stsb-mean-tokens] "
    )
    parser.add_argument(
        "--hidden_size", type=int, default=0, help="the hidden_size that sentence-transformer encode to. if 0, use the default hidden_size of sentence-transformer."
    )
    parser.add_argument(
        "--method", type=str, default="auto_cot", choices=["zero_shot", "zero_shot_cot", "few_shot", "few_shot_cot", "auto_cot"], help="method"
    )
    parser.add_argument(
        "--output_dir", type=str, default="experiment/race", help="output directory"
    )
    parser.add_argument(
        "--max_ra_len", type=int, default=20, help="maximum length of reasoning steps"
    )
    parser.add_argument(
        "--max_length_cot", type=int, default=256, help="maximum length of output tokens by model for reasoning extraction"
    )
    parser.add_argument( # 不带推理链的答案长度 (the length after answer extraction)
        "--max_length_direct", type=int, default=32, help="maximum length of output tokens by model for answer extraction"
    )
    parser.add_argument(
        "--limit_dataset_size", type=int, default=0, help="whether to limit test dataset size. if 0, the dataset size is unlimited and we use all the samples in the dataset for testing."
    )
    parser.add_argument(
        "--api_time_interval", type=float, default=1.0, help="sleep between runs to avoid excedding the rate limit of openai api"
    )
    parser.add_argument(
        "--temperature", type=float, default=0, help="temperature for GPT-3"
    )
    parser.add_argument(
        "--embedding_dir", type=str, default="./embeddings/race", help="the location to store the embeddings"
    )
    
    parser.add_argument(
        "--context_dir", type=str, default="./context/race", help="the location to store the context of the aggregate embeddings"
    )
    
    
    args = parser.parse_args()
    
    args.direct_answer_trigger = "\nTherefore, the answer is"
    # "Therefore, the answer ..." -> "The answer ..."
    args.direct_answer_trigger_for_zeroshot_cot = args.direct_answer_trigger # Therefore, the answer is ...
    args.cot_trigger = "Let's think step by step."
    
    if args.dataset == "race":
        args.direct_answer_trigger_for_zeroshot_cot = "\nTherefore, among A through D, the answer is"
    elif args.dataset == "dream":
        args.direct_answer_trigger_for_zeroshot_cot = "\nTherefore, among A through C, the answer is"
    elif args.dataset == "sciq":
        args.direct_answer_trigger_for_zeroshot_cot = "\nTherefore, among A through D, the answer is"
    elif args.dataset == "commonsense_qa":
        args.direct_answer_trigger_for_zeroshot_cot = "\nTherefore, among A through E, the answer is"
    else:
        raise ValueError("dataset is not properly defined... please select from [ race, dream, sciq, commonsense_qa ]")

    
    return args





# 从experment/race提取rationale


def extract_k_reasoning_chains(args:ChainEncodingArguments):
    """  
    将K条推理链编码成向量表示  
    
    Args:  
        reduced_reasoning_steps: K条推理链，每条包含多个推理步骤  
        
    Returns:  
        torch.Tensor: 形状为(K, min_ra_len, 768)的张量，其中：  
            - K: 推理链数量  
            - min_ra_len: 所有推理链中最短的长度  
            - 768: 编码向量的维度  
        args: 解析参数
    """  
    # args =parse_arguments()
    
    rationales = [] # stores the reasoning chains of the K chosen questions
    with open(args.output_dir, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line:Dict = json.loads(line)
            if "rationale" in line:
                rationale = line['rationale'].strip()
            else:
                raise KeyError("rationale is not in line, when extracting reasoning steps")
            rationales.append(rationale)
    
    
    # 将每个rationale拆分为若干reasoning steps
    # 每个reasoning step之间用换行符分隔
    # 每个reasoning step的开头用 "step i:" 标识
    # 例如：
        # Let's think step by step.
        # 1. We know that the capital of California is Sacramento.
        # 2. We know that Sacramento is in the state of California.
        # 3. Therefore, the answer is California.
    # 拆分为：    
        # step 1: We know that the capital of California is Sacramento.
        # step 2: We know that Sacramento is in the state of California.
    
    min_ra_len = -1 # minimum reaoning steps among all the questions
    reasoning_steps:List[List[str]] = []
    for rationale in rationales:
        steps:List[str] = rationale.split("\n")
        if min_ra_len == -1:
            min_ra_len = len(steps)
        else:
            min_ra_len = min(min_ra_len, len(steps))    
        reasoning_steps.append(steps)

    print("minimum reasoning steps  = ", min_ra_len)
    args.min_ra_len = min_ra_len
    
    reduced_reasoning_steps = []
    for steps in reasoning_steps:
        reduced_steps = steps[:min_ra_len] 
        for i,step in enumerate(reduced_steps):
            reduced_steps[i] = "step {}: {}".format(i+1, step)
        
        reduced_reasoning_steps.append(reduced_steps)
    
    return reduced_reasoning_steps, args


def encode_k_reasoning_chains(
    reduced_reasoning_steps:List[List[str]],
    args:ChainEncodingArguments
    )->torch.Tensor:
    # args = parse_arguments()
    encoder = SentenceTransformer(args.encoder)
    sentence_embedding_dimension = encoder.get_sentence_embedding_dimension()
    
    if sentence_embedding_dimension != args.hidden_size and args.hidden_size != 0: # 0 means no dimension adaption
        print(f"the hidden_size of encoder {args.hidden_size} does not match the hidden_size of the model {sentence_embedding_dimension}")
        print("========= adding dimension adaption layer .......")
        
        word_embedding_model = models.Transformer(args.encoder)
        
        # 2. 添加池化层  
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        
        # 3. 添加降维层（例如将768维降至256维）  
        dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(),   
                                out_features=args.hidden_size,   
                                activation_function=torch.nn.Tanh())  
        
        encoder = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model])
        
    # encoder.encode()
    
    all_embeddings = []
    for chain in reduced_reasoning_steps:
        
        chain_embedding = encoder.encode(
            chain, # List[str]
            convert_to_tensor=True,
            show_progress_bar=True,
            batch_size=128,
            normalize_embeddings=True,
        ) # shape = (min_ra_len, 768)
        all_embeddings.append(chain_embedding)
    
    # 将所有编码堆叠成一个张量  
    # shape: (K, min_ra_len, 768)  
    embeddings = torch.stack(all_embeddings) 
    
    print("********** Encode K reasoning chains ********************8")
    print("type(embeddings) = ", type(embeddings))

    print("embeddings.shape = ", embeddings.shape)
    print("*****************************************************")
    print()
        
    # shape = (K, min_ra_len, 768)
    return embeddings, args




def save_embeddings(
    embeddings:torch.Tensor,
    args:ChainEncodingArguments
    ):
    '''
    这个函数只能从autocot目录调用
    
    先调用 encode_k_reasoning_chains() 函数，得到 embeddings, args,
    再调用此函数
    
    '''
    # args = parse_arguments()
    # 保存编码结果

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  
    metadata={}
    save_info = {  
        'timestamp': timestamp,  
        'model_name': args.encoder,
        'hidden_size': args.hidden_size,  
        'num_chains': len(embeddings),  
        'min_ra_len': args.min_ra_len,  
        'shape': list(embeddings.shape),  
        'metadata': metadata or {},
    }  
    
    # 保存embeddings和元数据  
    embeddings_path = os.path.join(args.embedding_dir, f'embeddings_{args.dataset}_{args.hidden_size}.pt')
    metadata_path = os.path.join(args.embedding_dir, f'metadata_{args.dataset}_{args.hidden_size}.json')

    if not os.path.exists(args.embedding_dir):
        os.makedirs(args.embedding_dir)
        print("已创建数据集={}, 且 hidden_size={} 的embedding的存储目录{}".format(args.dataset, args.hidden_size, args.embedding_dir))
        print("开始保存K个CoT chain embedding, metadata ~~~")
    else:
        print("数据集={}, 且 hidden_size={} 的embedding的存储目录{}已存在~~~~开始检查文件完整性！".format(args.dataset, args.hidden_size, args.embedding_dir))

        if Path(embeddings_path).exists() and Path(metadata_path).exists():
            print(f"数据集={args.dataset}, 且 hidden_size={args.hidden_size} 的embedding文件 和 metadata文件都已存在, 保存完整, 直接使用~~~")
            return save_info
        
        elif (Path(embeddings_path).exists() or Path(metadata_path).exists()):
            print(f"数据集={args.dataset}, 且 hidden_size={args.hidden_size} 的 embedding文件 和 metadata文件 保存不完整（只存在其中一个）。")
            print(f"删除不完整的文件，重新开始保存~~~")

    torch.save(embeddings, embeddings_path)
    
    # 保存元数据  
    with open(metadata_path, 'w', encoding='utf-8') as f:  
        json.dump(save_info, f, ensure_ascii=False, indent=2)  
        
    save_info['embeddings_path'] = str(embeddings_path)  
    save_info['metadata_path'] = str(metadata_path)  
    
    return save_info  




def load_embeddings(
    dataset_name:str = 'race',
    save_dir:str = "./autocot/embeddings"
    )->Dict[str,Union[torch.Tensor, Dict[str, Any]]]:
    
    '''
    这个函数需要从项目根目录调用
    '''
    
    save_dir = save_dir / dataset_name
    embeddings_path = save_dir / f'embeddings_{dataset_name}.pt'  
    metadata_path = save_dir / f'metadata_{dataset_name}.json'


    if not (embeddings_path.exists() and metadata_path.exists()):  
            raise FileNotFoundError(f"找不到数据集为 {dataset_name} 的embedding文件")  

    embeddings = torch.load(embeddings_path)
    # 加载元数据  
    with open(metadata_path, 'r', encoding='utf-8') as f:  
        metadata = json.load(f)  
        
    return {  
        'embeddings': embeddings,  
        'metadata': metadata  
    }  

def aggregate_cot_embeddings(
    embeddings:torch.Tensor, 
    args: ChainEncodingArguments,
    use_attention:bool=True,
    )->torch.Tensor:
    """  
    使用RoutedCrossAttention将K个embeddings融合为1个context  
    
    Args:  
        embeddings: 原始embeddings张量，shape=(K, min_ra_len, hidden_size)  
        num_attention_heads: 注意力头数量  
        dropout: dropout率  
        
    Returns:  
        torch.Tensor: 融合后的上下文嵌入矩阵，shape=(min_ra_len, hidden_size)  
    """  
    K, min_ra_len, hidden_size = embeddings.shape  
    
    if use_attention:
        # 初始化RoutedCrossAttention  
        context, attention_info = HierarchicalAttentionFusion(hidden_size=hidden_size)(  
            embeddings=embeddings,    
        )  # shape = (min_ra_len, hidden_size)
        
        print("============ Attention Info ================")
        print(attention_info.keys())
    else:
        # 使用 average pooling
        context = torch.mean(embeddings, dim=0) # shape = (min_ra_len, hidden_size)
    
    print("type(context) = ",type(context))
    print("context.shape = ", context.shape)
    
    
    # 保存context
    context_path = os.path.join(args.context_dir, f'context_{args.dataset}_{args.hidden_size}.pt')
    parent_dir = os.path.dirname(context_path)
    
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
        print("已创建context embedding的存储目录{}".format(parent_dir))
        print("开始保存新的context~~~")
    else:
        # 如果embedding文件已存在
        print("context embedding的存储目录{}已存在".format(parent_dir))
        if Path(context_path).exists():
             print("数据集={},且 hidden_size={} 的context embedding文件已存在，无法覆盖".format(args.dataset,args.hidden_size))
             print("读取已保存的context~~~~")
             device, local_rank = setup_distributed()
             context = torch.load(context_path,map_location=device)
             return context
        else:
            print("数据集={},且 hidden_size={} 的context embedding文件不存在，开始保存~~~".format(args.dataset,args.hidden_size))

    torch.save(context, context_path)
    
    return context  



def get_cot_context(args:ChainEncodingArguments)->torch.Tensor:
    '''
    return context, shape=(min_ra_len, hidden_size)
    '''
    reasoning_chains, args =extract_k_reasoning_chains(args)

    embeddings,args = encode_k_reasoning_chains(reasoning_chains, args)
    
    save_info = save_embeddings(embeddings, args)
    print("=========== embedding save info ===============")
    print(save_info)
    
    context = aggregate_cot_embeddings(embeddings, args, True)
    
    print("******************** Get CoT Context Embedding ************************")
    print("type(context) = ", type(context))
    print("context.shape = ", context.shape)
    print("context = ", context)
    print("******************************************************************\n")

    return context


def rollback_one_step_extend(
        target_steps:int, 
        args:ChainEncodingArguments, 
        model:AutoModelForSequenceClassification,
        decoder:RollbackDecoderWithHead=None
    )->torch.Tensor:
    '''
    :param: model 这里的模型使用 BaasPromptModel初始化时赋值的self.model 传入
    
    :target_steps: num_suffix_tokens
    
    return context, shape=(target_steps, hidden_size)
    '''
    context:torch.Tensor = get_cot_context(args)
    source_steps = len(context)
    
    if target_steps <= source_steps:
        print("截断推理链context steps from {} to {}".format(source_steps, target_steps))
        return context[:target_steps]    
    else:
        print("扩展推理链context steps from {} to {}".format(source_steps, target_steps))
        context = context[:source_steps-1] # rollback 1 step
        
        # use multihead-attention to generate new steps using causual language modeling
        for i in range(source_steps-1, target_steps):
            # 生成新的推理步骤
            token_id, new_step_embedding = generate_new_step(context,model,decoder)
            new_step_embedding = new_step_embedding.unsqueeze(0) # [1, hidden_dim]
            context = torch.cat([context, new_step_embedding], dim=0)
        
        return context


def generate_new_step(context: torch.Tensor,
                      model:AutoModelForSequenceClassification,  
                     decoder: RollbackDecoderWithHead = None,  
                     temperature: float = 0.7) -> Tuple[int, torch.Tensor]:  
    """生成下一个token并返回其embedding  
    
    Args:  
        context: 形状为 [seq_len, hidden_dim] 的上下文张量  
        model: 带有输出头的解码器模型  
        temperature: 采样温度  
        : 词表大小（默认使用BERT的词表大小）  
    
    Returns:  
        tuple: (next_token_id, next_token_embedding)  
            - next_token_id: 下一个token的ID  
            - next_token_embedding: 下一个token的embedding向量 [hidden_dim]  
    """  
    if decoder is None:  
        # 如果没有传入模型，创建一个新的解码器层  
        d_model = context.size(-1)  # hidden_dim  
        decoder = RollbackDecoderWithHead(  
            model = model,
            d_model=d_model,  
            d_ff=d_model * 4, 
            num_heads=8,  
            dropout=0.1  
        )  
    
    device = context.device  
    decoder = decoder.to(device)  
    
    # 创建因果掩码  
    seq_len = context.size(0)  
    causal_mask = torch.triu(torch.ones((seq_len, seq_len), device=device), diagonal=1).bool()  
    causal_mask = ~causal_mask.unsqueeze(0).unsqueeze(1)  # [1, 1, seq_len, seq_len]  
    
    '''
    causal_mask = torch.triu(torch.ones((seq_len, seq_len), device=device), diagonal=1).bool()：
        创建一个上三角矩阵，其中主对角线以上的元素为 1，主对角线及以下的元素为 0。
        torch.triu 函数用于生成上三角矩阵，diagonal=1 表示从主对角线开始的位置。bool() 函数将结果转换为布尔类型。

    causal_mask = ~causal_mask.unsqueeze(0).unsqueeze(1)：
        对因果掩码进行取反操作，即将 1 变为 0，将 0 变为 1。
        然后，使用 unsqueeze(0) 和 unsqueeze(1) 在维度 0 和 1 上分别添加一个维度，
        使得因果掩码的形状变为 [1, 1, seq_len, seq_len]。
    '''
    
    
    # 将context扩展为batch维度
    if context.dim()==2:
        context = context.unsqueeze(0)  # [1, seq_len, hidden_dim]  
    
    with torch.no_grad():  
        # 通过解码器生成新的隐藏状态  
        next_token_logits  = decoder.forward(context, causal_mask)  # shape = [1, vocab_size]
        
        print("next_token_logits.shape = ", next_token_logits.shape)
        print("next_token_logits = ", next_token_logits)
        
        # 应用温度缩放  
        # # temperature 影响概率分布的"锐利度", <1 使分布更尖锐，>1 使分布更平缓    
        if temperature != 1.0:  
            next_token_logits = next_token_logits / temperature  
        
        # 计算概率分布  
        probs = F.softmax(next_token_logits, dim=-1)  
        
        print("probs.shape = ", probs.shape)
        print("probs = ", probs)    
        # 采样下一个token  
        next_token = torch.multinomial(probs, num_samples=1)  # [1, 1] 
        
        
        '''
        # multinomial 根据概率分布进行采样  
        # - probs: 概率分布 [1, vocab_size]  
        # - num_samples=1: 采样一个token  
        # - 返回的是词表中的索引（token ID）  

        # 例如，对于概率分布 [0.01, 0.31, 0.42, 0.26]：  
        # - 42% 的概率选择索引 2  
        # - 31% 的概率选择索引 1  
        # - 26% 的概率选择索引 3  
        # - 1% 的概率选择索引 0  
        
        '''
        

        # 获取token的embedding  
        next_token_embedding = decoder.embed_tokens(next_token)  # [1, 1, hidden_dim]  
        next_token_embedding = next_token_embedding.squeeze()  # [hidden_dim]  
        
        print("next_token_embedding.shape = ", next_token_embedding.shape)
        print("next_token_embedding = ", next_token_embedding)

    # return new_step.squeeze(0)  # [hidden_dim]  
    return next_token.squeeze().item(), next_token_embedding

        


if __name__ == '__main__':
    # context = get_cot_context()
    args = ChainEncodingArguments(
        hidden_size=768
    )
    context = rollback_one_step_extend(20,args)
    print("context.shape = ", context.shape)
    print("context = \n", context)
    
    
    