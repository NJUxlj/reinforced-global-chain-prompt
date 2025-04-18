
import argparse
import os
import sys
from dataclasses import dataclass

# 获取当前文件所在目录的父目录  
parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
# 将父目录添加到sys.path  
sys.path.insert(0, parent_directory) 
# 获取当前文件所在目录  
current_directory = os.path.dirname(os.path.abspath(__file__)) 
# 将当前目录添加到sys.path  
sys.path.insert(0, current_directory) 


# from autocot_utils import *
from data_utils.load import *
from config.config import NUM_CPU_PROCESSES
from autocot_utils import (
    setup_logger,
    LoggerWriter,
    logging,
    extract_answer,
    Decoder,
    create_demo_text
)







'''
python autocot_api.py \
--dataset race \
--method zero_shot_cot



python autocot_api.py \
--dataset sciq \
--method zero_shot_cot

python autocot_api.py \
--dataset dream \
--method zero_shot_cot


python autocot_api.py \
--dataset commonsense_qa \
--method zero_shot_cot

'''





def cot(method, question):
    args = parse_arguments()
    decoder = Decoder()

    args.method = method
    if args.method != "zero_shot_cot":
        if args.method == "auto_cot":
            args.demo_path = "demos/multiarith_auto"
        else:
            args.demo_path = "demos/multiarith_manual"
        demo = create_demo_text(args, cot_flag=True)
    else:
        demo = None

    x = "Q: " + question + "\n" + "A:"
    print('*****************************')
    print("Test Question:")
    print(question)
    print('*****************************')

    if args.method == "zero_shot":
        x = x + " " + args.direct_answer_trigger_for_zeroshot
    elif args.method == "zero_shot_cot":
        x = x + " " + args.cot_trigger
    elif args.method == "manual_cot":
        x = demo + x
    elif args.method == "auto_cot":
        x = demo + x + " " + args.cot_trigger
    else:
        raise ValueError("method is not properly defined ...")

    print("Prompted Input:")
    print(x.replace("\n\n", "\n").strip())
    print('*****************************')

    max_length = args.max_length_cot if "cot" in args.method else args.max_length_direct
    z = decoder.decode(args, x, max_length)
    z = z.replace("\n\n", "\n").replace("\n", "").strip()
    if args.method == "zero_shot_cot":
        z2 = x + z + " " + args.direct_answer_trigger_for_zeroshot_cot
        max_length = args.max_length_direct
        pred = decoder.decode(args, z2, max_length)
        print("Output:")
        print(z + " " + args.direct_answer_trigger_for_zeroshot_cot + " " + pred)
        print('*****************************')
    else:
        pred = z
        print("Output:")
        print(pred)
        print('*****************************')



# This script doesn't depend on parseargs, I've packed the required arguments into the Arguments class and put them in autocot_utils.py

def parse_arguments():
    parser = argparse.ArgumentParser(description="Zero-shot-CoT")

    parser.add_argument("--max_num_worker", type=int, default=NUM_CPU_PROCESSES, help="maximum number of workers for dataloader")
    parser.add_argument(
        "--model", type=str, default="gpt-4o", help="model used for decoding. Please select from [gpt-4o, gpt-4o-mini]"
    )
    
    parser.add_argument(
        "--dataset", type=str, default="race", help="select a dataset from ['race','sciq','dream','commonsense_qa']"
    )
    
    parser.add_argument(
        "--dataset_path", type=str, default="./data/race", help="dataset path"
    )
    
    parser.add_argument(
        "--method", type=str, default="zero-shot-cot", choices=["zero_shot", "zero_shot_cot", "few_shot", "few_shot_cot", "auto_cot"], help="method"
    )
    parser.add_argument(
        "--cot_trigger_no", type=int, default=1, help="A trigger sentence that elicits a model to execute chain of thought"
    )
    parser.add_argument(
        "--max_length_cot", type=int, default=256, help="maximum length of output tokens by model for reasoning extraction"
    )
    parser.add_argument(
        "--max_length_direct", type=int, default=32, help="maximum length of output tokens by model for answer extraction"
    )
    parser.add_argument(
        "--limit_dataset_size", type=int, default=0, help="whether to limit test dataset size. if 0, the dataset size is unlimited and we use all the samples in the dataset for testing."
    )
    parser.add_argument(
        "--api_time_interval", type=float, default=1.0, help=""
    )
    parser.add_argument(
        "--temperature", type=float, default=0, help=""
    )
    parser.add_argument(
        "--log_dir", type=str, default="./cot_log/", help="log directory"
    )
    args = parser.parse_args()

    args.direct_answer_trigger_for_fewshot = "The answer is"
    args.direct_answer_trigger_for_zeroshot = "The answer is"
    args.direct_answer_trigger_for_zeroshot_cot = "Therefore, the answer is"
    
    if args.dataset == 'race':
        args.direct_answer_trigger_for_zeroshot_cot = "\nTherefore, among A through D, the answer is"
    elif args.dataset=='sciq':
        args.direct_answer_trigger_for_zeroshot_cot = "\nTherefore, among A through D, the answer is"
    elif args.dataset=='dream':
        args.direct_answer_trigger_for_zeroshot_cot = "\nTherefore, among A through C, the answer is"
    elif args.dataset=='commonsense_qa':
        args.direct_answer_trigger_for_zeroshot_cot = "\nTherefore, among A through E, the answer is"
    else:
        raise ValueError("dataset is not properly defined... check args.dataset")
        
    args.cot_trigger = "Let's think step by step."

    return args




def zero_shot_cot(question, answer, args):
    decoder = Decoder()
    # cot_trigger = "Let's think step by step."
    
    question = question.replace("\n\n", "\n").replace("\n", " ").strip()
    x = "Q: " + question + "\n" + "A:"
    # print('*****************************')
    # print("Test Question:")
    # print(question)
    # print('*****************************')


    x:str = x + " " + args.cot_trigger
    '''
    Q: xxxxx
    A: Let's think step by step
    '''

    # print("Prompted Input:")
    # we define the separatpor of reasoning steps is "\n"
    print(x)
    print()
   

    # generate the reasoning chain including the final answer
    z = decoder.decode(args, x, args.max_length_cot)
    # later, we need to use the '\n' to separate the reasoning steps
    z = z.replace("\n\n", "\n").strip()

    # let the model extract the final answer from the reasoning chain
    z2 = x + z + " " + args.direct_answer_trigger_for_zeroshot_cot
    
    pred = decoder.decode(args, z2, args.max_length_direct)
    # print("Reasoning Chain Output:")
    print(z + "\n" + args.direct_answer_trigger_for_zeroshot_cot + " " + pred)
    # print('*****************************')

    
    # 提取预测结果  
    pred_before = extract_answer(z + " " + args.direct_answer_trigger_for_zeroshot_cot + " " + pred)  
    pred_after = pred_before.rstrip('.')  
    pred_list = [pred_after]  
    pred_mode = pred_after  

    # print("Output:")  
    # print(z + " " + args.direct_answer_trigger_for_zeroshot_cot + " " + pred)  
    print(f"pred_before : {pred_before}")   # A.
    print(f"pred_after : {pred_after}")    # A
    print(f"pred_list : {pred_list}")  
    print(f"pred_mode : {pred_mode}")  
    print(f"GT : {answer}")  
    print('*' * 25) 
    print('*' * 25) 
        
        
    
        

def cot_log_generator():
    '''
        在某个数据集上生成zero-shot-cot日志
    '''
    args = parse_arguments()
    # args = Arguments(
    #     dataset_path='../data/race/',
    #     dataset='race',
    # )
    
    args = parse_arguments()
    
    logger = setup_logger(args.dataset, args)
    
    # 保存原始的stdout  
    original_stdout = sys.stdout  
    
    train_ds, data_config = preprocess_dataset_autocot(args.dataset)

    question_key = data_config.question_key
    answer_key = data_config.label_key
    questions = train_ds[question_key]
    answers = train_ds[answer_key]
        
    train_ds = [  
            {question_key: q, answer_key: a}   
            for q, a in zip(questions, answers)  
    ]  
    
    try:
        sys.stdout = LoggerWriter(logger, logging.INFO)

        # generate logs
        for i, example in enumerate(train_ds[:300]):
            print(f"{i}st data")  
            print("1_th_sampling")  
            
            # print("example = \n", example)
            
            # only pick the question and the answer field in the dataset
            # zero_shot_cot(example[first_four_columns[1]], example[first_four_columns[3]], args)  
            
            zero_shot_cot(example[question_key], example[answer_key], args)  
            
            # print('*' * 25)   
        
    finally:
        sys.stdout = original_stdout
    

if __name__ == "__main__":
    cot_log_generator()
    # train_ds, first_four_columns = preprocess_dataset_autocot('race')
    # print(first_four_columns)
    
    # for i, example in enumerate(train_ds[:5]):
    #     print(f"=============={i+1}===============")
    #     print(example)