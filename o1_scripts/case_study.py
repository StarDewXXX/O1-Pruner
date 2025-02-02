import torch
import json
from transformers import AutoTokenizer
from tqdm import tqdm
import random
from utils import extract_answer, get_example
from grader import grade_answer
from collections import defaultdict
from datasets import load_dataset
import os
import argparse
import json
import time
import os
import sys
import re
import random
import numpy as np


# data_base = json.load(open("./data/model_generated/Marco_math_test_8192_normal_K-1.json"))
# data_pruned = json.load(open("./data/model_generated/Marco-lht-alpha-2_math_test_8192_normal_K-1.json"))
data_base = json.load(open("./data/model_generated/..."))
data_pruned = json.load(open("./data/model_generated/..."))
tokenizer = AutoTokenizer.from_pretrained("Qwen/QwQ-32B-Preview")

for i in range(len(data_base)):
    problem = data_base[i]['problem']
    real_answer = data_base[i]['ground_truth_answer']
    base_solution = data_base[i]['solution']
    pruned_solution = data_pruned[i]['solution']

    base_length = len(tokenizer(base_solution)['input_ids'])
    pruned_length = len(tokenizer(pruned_solution)['input_ids'])

    print(f"[problem {i}]\n",problem)
    print("[base_solution]\n",base_solution)
    print("-"*100)
    print("[pruned_solution]\n",pruned_solution)

    
    print("[real_answer]:",real_answer)
    print("[base answer]:",extract_answer(base_solution),"[pruned answer]:",extract_answer(pruned_solution))
    print("[base_length]:",base_length)
    print("[pruned_length]:",pruned_length)
    print("="*100)
    input("continue?")