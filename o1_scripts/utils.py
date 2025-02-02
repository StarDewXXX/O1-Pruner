import random
import re
from datasets import load_dataset
def try_extract(output, pattern):
    matches = re.findall(pattern, output, re.DOTALL)
    answers = [match.strip() for match in matches]
    if len(answers) > 0:
        return answers[-1]
    else:
        return "None"

def extract_answer(output, model="llama"):

    answers = []
    for piece in output.split('boxed{')[1:]:
        n = 0
        for i in range(len(piece)):
            if piece[i] == '{':
                n += 1
            elif piece[i] == '}':
                n -= 1
                if n < 0:
                    if i + 1 < len(piece) and piece[i + 1] == '%':
                        answers.append(piece[: i + 1])
                    else:
                        answers.append(piece[:i])
                    break
    if len(answers) > 0:
        return answers[0]
    else:
        return "None"

    return "None"

def extract_all_answers(output):
    all_substrs = [
        "Answer",
        "ANSWER",
        "The",
        "Task",
        "Finished"
    ]
    for sub_str in all_substrs:
        output = output.replace(sub_str, sub_str.lower())
    # print(output)
    if output[-1] != "\n":
        output += "\n"
    # 匹配所有以 '# Answer' 开头，答案位于下一行之前的内容
    pattern = r"# answer\s*(.*?)\n"
    matches = re.findall(pattern, output, re.DOTALL)
    # 返回所有匹配的答案
    answers = [match.strip() for match in matches]
    
    return answers