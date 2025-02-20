import random
import re

def try_extract(output, pattern):
    matches = re.findall(pattern, output, re.DOTALL)
    answers = [match.strip() for match in matches]
    if len(answers) > 0:
        return answers[-1]
    else:
        return "None"

def extract_answer(output):

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