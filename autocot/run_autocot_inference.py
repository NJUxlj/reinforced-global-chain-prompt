'''
Adapted from https://github.com/kojima-takeshi188/zero_shot_cot
'''


'''

python run_inference.py \
--dataset multiarith \
--demo_path demos/multiarith \
--output_dir experiment/multiarith

'''

import argparse
from autocot_utils import *

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
        "--log_dir", type=str, default="./log/", help="log directory"
    )
    
    
    args = parser.parse_args()
    
    if args.dataset == "race":
        args.dataset_path = "./dataset/AQuA/test.json"
        args.direct_answer_trigger = "\nTherefore, among A through D, the answer is"
    elif args.dataset == "dream":
        args.dataset_path = "./dataset/grade-school-math/test.jsonl"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "sciq":
        args.dataset_path = "./dataset/CommonsenseQA/dev_rand_split.jsonl"
        args.direct_answer_trigger = "\nTherefore, among A through E, the answer is"
    elif args.dataset == "commonsense_qa":
        args.dataset_path = "./dataset/AddSub/AddSub.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    

    # "Therefore, the answer ..." -> "The answer ..."
    trigger = args.direct_answer_trigger.replace("\nTherefore, ", "")
    args.direct_answer_trigger_for_zeroshot = trigger[0].upper() + trigger[1:] # "The answer is"
    args.direct_answer_trigger_for_zeroshot_cot = args.direct_answer_trigger
    args.direct_answer_trigger_for_fewshot = "The answer is"
    args.cot_trigger = "Let's think step by step."
    
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
    dataloader = setup_data_loader(args)
    
    
    if args.method == "few_shot":
        demo = create_demo_text(args, cot_flag=False)
    elif args.method == "few_shot_cot" or args.method == "auto_cot":
        demo = create_demo_text(args, cot_flag=True)
    else:
        pass
    
    
    total = 0 # 记录数据集中的总共问题数量
    correct_list = [] # 记录 1,0,1,0 ... 的列表
    
    with open(args.output_dir, "a") as wp:

        for i, data in enumerate(dataloader):
            if i < args.resume_id - 1: # 断点续传， 实际需要续传的断点index = resume_id -1
            # if i < 297:
                continue
            
            output_line = {} 
            
            print('*************************')
            print("{}st data".format(i+1))
            
            
            # Prepare question template ...
            x, y = data
            x = "Q: " + x[0] + "\n" + "A:"
            y = y[0].strip()            
            
            output_line["question"] = x
            output_line["gold_ans"] = y
            
            # adding different answer triggers
            if args.method == "zero_shot":
                x= x + " " + args.direct_answer_trigger_for_zeroshot
            elif args.method == "zero_shot_cot":
                x= x + " " + args.cot_trigger
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
            
            z = decoder.decode(args, x, max_length)
            
            output_line['rationale'] = z
            
            # Answer extraction for zero-shot-cot ...
            if args.method == "zero_shot_cot":
                # 将input+rationale+answer + "Therefore the answer is:" 重新给到模型，引导模型生成精确的答案 (e.g. A,B,C,D)
                z2 = x + z + " " + args.direct_answer_trigger_for_zeroshot_cot # i.e. "\nTherefore, among A through D, the answer is"
                max_length = args.max_length_direct
                pred = decoder.decode(args, z2, max_length) # output ["A", "B", "C", "D"], 
                print("z2 + pred = ", z2 + pred)
            else:
                pred = z
                print("x + pred = ", x + pred)
            
            # Cleansing of predicted answer ...
            pred = answer_cleansing(args, pred)
            
            
            output_line['pred_ans'] = pred
            output_line['wrap_que'] = x
            
            # 把对一个样本的完整推理过程写入到输出文件中
            output_json = json.dumps(output_line)
            
            wp.write(output_json + '\n')
            
            
            # Choose the most frequent answer from the list ...
            print("pred : {}".format(pred))
            print("GT : " + y)
            print('*************************')
            
            # checking answer
            correct = (np.array([pred])==np.array([y])).sum().item()
            correct_list.append(correct)
            total += 1
            
            if (args.limit_dataset_size != 0) and ((i+1) >= args.limit_dataset_size):
                break
    # Calculate accuracy ...
    accuracy = (sum(correct_list)*1.0/total) * 100
    print("accuracy : {}%".format(accuracy))
            
if __name__ == "__main__":
    main()