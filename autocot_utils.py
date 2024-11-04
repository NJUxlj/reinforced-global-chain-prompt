
'''
Adapted from https://github.com/kojima-takeshi188/zero_shot_cot
'''

from statistics import mean
from torch.utils.data import Dataset
import openai
import os
import multiprocessing
import json
import numpy as np
import torch
import re
import random
import time
import datetime


def decoder_for_gpt4(args, input, max_length):
    time.sleep(args.api_time_interval)
    
    
    if args.model == "gpt4o":
        engine = "gpt-4-0613"
    
    elif args.model == "gpt4o-mini":
        engine = "gpt-4-32k-0613"
    else:
        raise ValueError("GPT model is not properly defined ... please select gpt4o or gpt4o-mini")
    
    if ("few_shot" in args.method or "auto" in args.method)  and engine == "code-davinci-002":
        response = openai.Completion.create(
          engine=engine,
          prompt=input,
          max_tokens=max_length,
          temperature=args.temperature,
          top_p=1,
          frequency_penalty=0,
          presence_penalty=0,
          stop=["\n"]
        )
    else:
        response = openai.Completion.create(
            engine=engine,
            prompt=input,
            max_tokens=max_length,
            temperature=args.temperature,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None
        )

    return response["choices"][0]["text"]