'''
Adapted from https://github.com/kojima-takeshi188/zero_shot_cot
'''


'''

python run_autocot_inference.py \
--dataset race \
--demo_path demos/race \
--output_dir experiment/race \
--method auto_cot \
--max_length_cot 2048

python run_autocot_inference.py \
--dataset sciq \
--demo_path demos/sciq \
--output_dir experiment/sciq \
--method auto_cot

python run_autocot_inference.py \
--dataset dream \
--demo_path demos/dream \
--output_dir experiment/dream \
--method auto_cot

python run_autocot_inference.py \
--dataset commonsense_qa \
--demo_path demos/commonsense_qa \
--output_dir experiment/commonsense_qa \
--method auto_cot

'''

import argparse
from autocot_utils import *
from pathlib import Path  

# 获取当前文件所在目录的父目录  
parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  

# 将父目录添加到sys.path  
sys.path.insert(0, parent_directory) 

from config import Config
from config import NUM_CPU_PROCESSES


def parse_arguments():
    
    parser = argparse.ArgumentParser(description="Zero-shot-CoT")
    
    parser.add_argument("--random_seed", type=int, default=42, help="random seed")
    
    parser.add_argument(
        "--dataset", type=str, default="race", choices=["race", "dream", "sciq", "commonsense_qa"], help="dataset used for experiment, select from [ race, dream, sciq, commonsense_qa ]"
    )
    parser.add_argument(
        "--demo_path", type=str, default="demos/race", help="pre-generated demos used for experiment"
    )
    parser.add_argument(
        "--resume_id", type=int, default=0, help="resume from which question id (current line number in the output file), if the experiment fails accidently (e.g., network error)"
    )
    parser.add_argument("--minibatch_size", type=int, default=1, choices=[1], help="minibatch size should be 1 because GPT-3 API takes only 1 input for each request")
    
    parser.add_argument("--max_num_worker", type=int, default=NUM_CPU_PROCESSES, help="maximum number of workers for dataloader")
    
    parser.add_argument(
        "--model", type=str, default="gpt-4o", choices=["gpt-4o", "gpt4o-mini", "Qwen2.5-3B"], help="model used for decoding. Select from [ gpt-4o, gpt-4o-mini, Qwen2.5-3B ]"
    )
    
    parser.add_argument(
        "--encoder", type=str, default="all-MiniLM-L6-v2", help="which sentence-transformer encoder for clustering"
    )
    
    parser.add_argument(
        "--method", type=str, default="auto_cot", choices=["zero_shot", "zero_shot_cot", "few_shot", "few_shot_cot", "auto_cot"], help="method"
    )
    parser.add_argument(
        "--output_dir", type=str, default="experiment/race", help="output directory"
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
        "--log_dir", type=str, default="./autocot_log/", help="log directory"
    )
    
    
    args = parser.parse_args()
    
    args.direct_answer_trigger = "\nTherefore, the answer is"
    # "Therefore, the answer ..." -> "The answer ..."
    trigger = args.direct_answer_trigger.replace("\nTherefore, ", "")
    args.direct_answer_trigger_for_zeroshot = trigger[0].upper() + trigger[1:] # "The answer is"
    args.direct_answer_trigger_for_zeroshot_cot = args.direct_answer_trigger # Therefore, the answer is ...
    args.direct_answer_trigger_for_fewshot = "The answer is"
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



def main():
    args = parse_arguments()
    print('*****************************')
    print(args)
    print('*****************************')
    
    fix_seed(args.random_seed)
    print("OPENAI_API_KEY:")
    print(os.getenv("OPENAI_API_KEY")[0:5] + '**********')
    
    # Initialize decoder class (load model and tokenizer) ...
    decoder = Decoder()
    
    print("setup data loader ...")
    dataloader, config = setup_data_loader(args)
    
    # 我们要从dataloader中抽取k个样本进行推理。
    # 获取一个只包含k个question的新dataloader
    k = 8
    labels, kmeams, sorted_clusters = cluster_dataloader(dataloader, args, config, begin_example=300, num_example=300, n_clusters=k)
    
    dataloader_k = get_k_questions_dataloader_from_clusters(sorted_clusters, args, config, n_clusters=k)
    
    if args.method == "few_shot":
        demo = create_demo_text(args, cot_flag=False)
    elif args.method == "few_shot_cot" or args.method == "auto_cot":
        demo = create_demo_text(args, cot_flag=True)
        print("demo length = ", len(demo))
    else:
        pass
    
    
    total = 0 # 记录数据集中的总共问题数量
    correct_list = [] # 记录 1,0,1,0 ... 的列表
    
    if not os.path.exists(args.output_dir):
        try:
            directory = os.path.dirname(args.output_dir)
        
            # 创建输出目录 experiment
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
                print("create the directory [experiment]")

            # 创建实验文件
            # Path对象可以更方便地处理文件路径  
            path = Path(args.output_dir)  
            # touch() 方法会创建一个空文件，如果父目录不存在会抛出异常  
            path.touch(exist_ok=True)  # exist_ok=False 表示如果文件存在则抛出异常  
            print("创建实验文件{}成功！".format(args.output_dir))
        except Exception as e:
            raise RuntimeError(f"创建实验文件失败：{str(e)}")
    else:
        print("The file [{}] already exists!".format(args.output_dir))
        threshold = 5
        with open(args.output_dir, 'r', encoding='utf-8') as f:  
            # readlines() 读取所有行到列表中  
            lines = f.readlines()  
            line_count = len(lines)  
            if line_count>threshold:
                raise RuntimeError("The file [{}] already has data in it (lines > 5)! \n\n Abort the inference process...".format(args.output_dir))
            else:
                print("The file [{}] is similar to empty (lines <= 5)! \n\n We continue to write data in ...".format(args.output_dir))
    
    
    
    with open(args.output_dir, "w", encoding='utf-8') as wp:

        for i, data in enumerate(dataloader_k):
            
            
            # data.type = dict{'question':["xxxx", "xxxx"] , 'answer':["A", "B", ...] }
            
            if i < args.resume_id - 1: # 断点续传， 实际需要续传的断点index = resume_id -1
                #  跳过不需要的问题
                continue
            
            
            # 验证data的格式和键值 
            if not isinstance(data, dict) or not data:  
                print(f"Warning: Invalid data format at index {i}")  
                print(f"Data: {data}")  
                raise RuntimeError(f"Warning: Invalid data format or empty at index {i}")  
                
            if config.question_key not in data: #  or config.label_key not in data:  
                print(f"Warning: Missing required keys at index {i}")  
                print(f"Available keys: {data.keys()}")  
                raise RuntimeError(f"Warning: Missing required keys at index {i}") 
            
            
            
            output_line = {} # outputline 是一个仅占一行的json对象，因此不能包括换行符
            
            
            print('*************************')
            print("{}st data".format(i+1))
            
            print("type(data) = ", type(data))
            # print("data = \n",data)
            
            # Prepare question template ...
            # x, y = data
            x = data[config.question_key]
            
            # y = data[config.label_key]
            # x = "Q: " + x[0] + "\n" + "A:"
            x = "Q: " + x + " " + "A: "

            # y = y[0].strip()            
            # y = y.strip()            

            
            output_line["question"] = x
            # output_line["gold_ans"] = y
            
            # adding different answer triggers
            if args.method == "zero_shot":
                x= x + " " + args.direct_answer_trigger_for_zeroshot
            elif args.method == "zero_shot_cot":
                x= x + " " + args.cot_trigger # Let's think step by step
            elif args.method == "few_shot": # input + demonstrations + A:
                x = demo + x
            elif args.method == "few_shot_cot":  # input + demonstrations + A:
                x = demo + x
            elif args.method == "auto_cot":
                x = demo + x + " " + args.cot_trigger
            else:
                raise ValueError("method is not properly defined ... please select from [ zero_shot, zero_shot_cot, auto_cot, few_shot, few_shot_cot ]")

            # use LLM to answer the question ...
            # zero-shot, few-shot 输出长度都是 1
            max_length = args.max_length_cot if "cot" in args.method else args.max_length_direct
            
            # 进行 Auto-CoT 推理
            z = decoder.decode(args, x, max_length)
            
            print("rationle = \n", z)
            
            
            # Answer extraction for zero-shot-cot ...
            if args.method == "zero_shot_cot":
                # 将input+rationale+answer + "Therefore the answer is:" 重新给到模型，引导模型生成精确的答案 (e.g. A,B,C,D)
                z2 = x + z + " " + args.direct_answer_trigger_for_zeroshot_cot # i.e. "\nTherefore, among A through D, the answer is"
                max_length = args.max_length_direct
                pred = decoder.decode(args, z2, max_length) # output ["A", "B", "C", "D"], 
                print("z2 + pred = ", z2 + pred)
            else:
                '''
                do nothing ...
                '''
                # 进行答案抽取
                z2 = x + "\n" + z + "\n" + args.direct_answer_trigger_for_zeroshot_cot # i.e. "\nTherefore, among A through D, the answer is"
                pred = decoder.decode(args, z2, args.max_length_direct) # output ["A", "B", "C", "D"],
                pred = extract_answer(pred)
                
                # print("===================================")
                # print("x + rationale = \n", z2)
                # print("===================================")
            
            # Cleansing of predicted answer ...
            
            # 推理后去除推理链中的换行符, 使得output_line只占一行
            z = z.replace("\n\n","\n").replace("\n", '\n') # r'\n'
            
            # print(" =====================")
            # print("after replacement, the rationale  =  \n", z)
            # print("=================================")
            
            output_line['rationale'] = z
            
            # 原项目中，使用regex规则解析最终答案，减少GPT调用次数
            # race这样的数据集用不了，它的推理链基本无规则。
            # pred = answer_cleansing(args, z)
            
            
            output_line['pred_ans'] = pred
            
            # wrapped question
            output_line['wrap_que'] = x
            
            # 把对一个样本的完整推理过程写入到输出文件中
            output_json = json.dumps(output_line) # transform to json string
            
            wp.write(output_json + '\n')
            
            
            # Choose the most frequent answer from the list ...
            print("pred : {}".format(pred))
            # print("GT : " + y)
            print('*************************')
            
            # checking answer
            # correct = (np.array([pred])==np.array([y])).sum().item()
            # correct_list.append(correct)
            
            total += 1
            
            # cot推理的样本数量达到上限，直接停止
            if (args.limit_dataset_size != 0) and ((i+1) >= args.limit_dataset_size):
                break
            
    # Calculate accuracy ...
    # accuracy = (sum(correct_list)*1.0/total) * 100
    # print("accuracy : {}%".format(accuracy))
            
if __name__ == "__main__":
    main()