'''
Adapted from https://github.com/kojima-takeshi188/zero_shot_cot
'''

import argparse
from autocot_utils import *



def parse_arguments():
    
    parser = argparse.ArgumentParser(description="Zero-shot-CoT")
    
    parser.add_argument("--random_seed", type=int, default=1, help="random seed")
    
    parser.add_argument(
        "--dataset", type=str, default="race", choices=["race", "record", "multirc", "arc"], help="dataset used for experiment, select from [ race, record, multirc, arc ]"
    )
    parser.add_argument(
        "--demo_path", type=str, default="demos/race", help="pre-generated demos used for experiment"
    )
    parser.add_argument(
        "--resume_id", type=int, default=0, help="resume from which question id (current line number in the output file), if the experiment fails accidently (e.g., network error)"
    )
    parser.add_argument("--minibatch_size", type=int, default=1, choices=[1], help="minibatch size should be 1 because GPT-3 API takes only 1 input for each request")
    
    parser.add_argument("--max_num_worker", type=int, default=0, help="maximum number of workers for dataloader")
    
    parser.add_argument(
        "--model", type=str, default="gpt4o", choices=["gpt4o", "gpt4o-mini", "Qwen2.5-3B"], help="model used for decoding. Select from [ gpt4o, gpt4o-mini, Qwen2.5-3B ]"
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
    parser.add_argument(
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
    
    
    






def main():
    args = parse_arguments()
    
    
    
if __name__ == "__main__":
    main()