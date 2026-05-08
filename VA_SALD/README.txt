example


Open VA_SALD



Run below

conda env create -f environment.yml

export PYTHONPATH=$(pwd)
CUDA_VISIBLE_DEVICES=3 accelerate launch --num_processes=1  --main_process_port=0 trainer/VA_SALD_Guidance_zerothorder.py  --config config/VA_SALD_Guidance_zerothorder.py:base
