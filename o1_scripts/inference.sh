# model_name="Marco"
# model_path="AIDC-AI/Marco-o1" 
# CUDA_VISIBLE_DEVICES=4,5,6,7 HF_ENDPOINT=https://hf-mirror.com python v4_scripts/inference.py --K 512 --num_samples 24 --dataset math_train --model ${model_name} --speed normal --model_path ${model_path}
model_name="QwQ-loft-iter0"
model_path="./saves/math/QwQ-loft-iter0-alpha_5" 
CUDA_VISIBLE_DEVICES=4,5,6,7 HF_ENDPOINT=https://hf-mirror.com python o1_scripts/inference.py --K 12 --num_samples 5000 --dataset math_train --model ${model_name} --speed normal --model_path ${model_path}