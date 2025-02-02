import json
from grader import grade_answer, extract_answer

K = 4
data = json.load(open("./data/o1_cot_k4_llama-70b.json","r"))[0:K*200]

# data = json.load(open("./data/normal_output_8b.json","r"))
# K = 1

# print(data[0]['messages'][-1]['content'])
# print(data[1]['messages'][-1]['content'])
results = [item['messages'][-1]['content'] for item in data]
extracted_results = []
real_answers = [item['ground_truth_answer'] for item in data]
print(len(results))
print(len(real_answers))
wanted = 0

for index in range(len(results)):

    r = results[index]
    r = r.lower()
    
    extracted_r = r.split("# answer")[-1].strip()
    extracted_r = extracted_r.split("the answer is")[-1].strip()
    extracted_r = extracted_r.split("task finished.")[-1].strip()
    extracted_r = extracted_r.split("answer:")[-1].strip()
    extracted_r = extracted_r.split("the answer")[-1].strip()

    # for m in data[index]['messages']:
    #     print(f"[{m['role']}]: {m['content']}")
    # print("-"*50)
    # input("continue?")

    # if grade_answer(extracted_r, real_answers[index]):
    #     output = data[index]['messages'][-1]['content']
    #     if output.find("incorrect") != -1 or output.find("wrong") != -1 or output.find("mistake") != -1 or output.find("partially") != -1:
    #         wanted += 1
            # print("[Question]",data[index]['problem'])
            # print("[Answer]",output)
            # print("-"*50)
            # input("continue?")
            # for m in data[index]['messages']:
            #     print(f"[{m['role']}]: {m['content']}")
            # print("-"*50)
            # input("continue?")

    if len(extracted_r) > 50:
    #     print(extracted_r)
        extracted_r = extracted_r[0:50]
    #     for m in data[index]['messages']:
    #         print(f"[{m['role']}]: {m['content']}")
    #     print("-"*50)
    #     input("continue?")

    extracted_results.append(extracted_r)
    
    
    # print(extracted_r)
    # print("----------------------------------")

count_of_successful_reflection = 0
count = 0
correct = 0

print("start")


for batch in range(len(data)//K):
    correct_batch = 0
    correct_batch_have_reflection = 0
    for i in range(batch*K,(batch+1)*K):
        if grade_answer(extracted_results[i], real_answers[i]):
            correct_batch += 1
            output = data[i]['messages'][-1]['content']
            if output.find("incorrect") != -1 or output.find("wrong") != -1 or output.find("mistake") != -1 or output.find("partially") != -1:
                correct_batch_have_reflection += 1
            #     print(output)
            #     input("continue?")
    # for i in range(batch*K,(batch+1)*K):
    #     if grade_answer(extracted_results[i], real_answers[i]):
    #         correct_batch += 1
    # input("continue?")
    if correct_batch > 0:
        correct += 1
    
    if correct_batch_have_reflection:
        count_of_successful_reflection += 1
    count += 1

# for batch in range(len(data)//K):
#     correct_batch = 0
#     for i in range(batch*K,(batch+1)*K):
#         if grade_answer(extracted_results[i], real_answers[i]):
#             correct_batch += 1
    
#     if correct_batch > 0:
#         correct += 1 
#     count += 1


# for i in range(len(data)):
#     if grade_answer(extracted_results[i], real_answers[i]):
#         correct += 1
#     count += 1

print("ratio:",correct / count, f"correct:{correct} count_successful_reflection:{count_of_successful_reflection} count:{count} ")
print("wanted ratio:",wanted / len(data))
