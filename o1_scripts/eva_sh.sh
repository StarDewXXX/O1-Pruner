model="Marco" #"Marco" #SimpleO1/data/model_evalution/QwQ-loft_math_test_8192_normal_K-1
model_name="Marco_0_40_acc-iter0-MATH-gated" #"QwQ-lht-alpha_2" # "Marco-loft-iter0-alpha_0-ppo-normed" #"Marco-lht-alpha-5" #"Marco-lht-alpha-2"
model="QwQ"
model_name="QwQ-loft-iter0-alpha_3-v3"

HF_ENDPOINT=https://hf-mirror.com python v4_scripts/result_analysis.py --file_name "${model_name}_math_test_8192_normal_K-1" --model ${model} --K 1
HF_ENDPOINT=https://hf-mirror.com python v4_scripts/result_analysis.py --file_name "${model_name}_gsm8k_8192_normal_K-1" --model ${model} --K 1
HF_ENDPOINT=https://hf-mirror.com python v4_scripts/result_analysis.py --file_name "${model_name}_gaokao_8192_normal_K-1" --model ${model} --K 1

