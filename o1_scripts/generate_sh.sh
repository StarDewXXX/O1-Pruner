model_name="QwQ-loft-iter0-alpha_3-v2" #"Marco-loft-iter0-alpha_0-ppo-normed" #"Marco"
model_path=./saves/math/QwQ-loft-iter0-alpha_3-v2 #Qwen/QwQ-32B-Preview  #"AIDC-AI/Marco-o1" 
CUDA_VISIBLE_DEVICES=4,5,6,7 HF_ENDPOINT=https://hf-mirror.com python o1_scripts/inference.py --K 1 --num_samples 1400 --dataset math_test --model ${model_name} --speed normal --model_path ${model_path}
# CUDA_VISIBLE_DEVICES=4,5,6,7 HF_ENDPOINT=https://hf-mirror.com python v4_scripts/inference.py --K 1 --num_samples 1400 --dataset gsm8k --model ${model_name} --speed normal --model_path ${model_path}
# CUDA_VISIBLE_DEVICES=4,5,6,7 HF_ENDPOINT=https://hf-mirror.com python v4_scripts/inference.py --K 1 --num_samples 1400 --dataset gaokao --model ${model_name} --speed normal --model_path ${model_path}