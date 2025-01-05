# CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-125m c4

# CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-125m c4 --wbits 4 --nearest

# CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-125m c4 --wbits 4 --groupsize 1024

# CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-350m c4

# CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-350m c4 --wbits 4 --nearest

# CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-350m c4 --wbits 4 --groupsize 1024

# CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-1.3b c4

# CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-1.3b c4 --wbits 4 --nearest

# CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-1.3b c4 --wbits 4 --groupsize 1024

# CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-2.7b c4 > log/opt2.7b_full.txt

# CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-2.7b c4 --wbits 4 --nearest > log/opt2.7b_rtn.txt

# CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-2.7b c4 --wbits 4 --groupsize 1024 > log/opt2.7b_gptq.txt

CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-6.7b c4 > log/opt6.7b_full.txt

CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-6.7b c4 --wbits 4 --nearest > log/opt6.7b_rtn.txt

CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-6.7b c4 --wbits 4 --groupsize 1024 > log/opt6.7b_gptq.txt

CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-13b c4 > log/opt13b_full.txt

CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-13b c4 --wbits 4 --nearest > log/opt13b_rtn.txt

CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-13b c4 --wbits 4 --groupsize 1024 > log/opt13b_gptq.txt

CUDA_VISIBLE_DEVICES=0 python llama.py huggyllama/llama-13b c4 --new-eval > log/llama13b_full.txt

CUDA_VISIBLE_DEVICES=0 python llama.py huggyllama/llama-13b c4 --wbits 4 --true-sequential --act-order --new-eval --nearest > log/llama13b_rtn_4b.txt

CUDA_VISIBLE_DEVICES=0 python llama.py huggyllama/llama-13b c4 --wbits 4 --true-sequential --act-order --new-eval --groupsize 1024 > log/llama13b_gptq_4b.txt

CUDA_VISIBLE_DEVICES=0 python llama.py huggyllama/llama-13b c4 --wbits 3 --true-sequential --act-order --new-eval --nearest > log/llama13b_rtn_3b.txt

CUDA_VISIBLE_DEVICES=0 python llama.py huggyllama/llama-13b c4 --wbits 3 --true-sequential --act-order --new-eval --groupsize 1024 > log/llama13b_gptq_3b.txt

CUDA_VISIBLE_DEVICES=0 python llama.py huggyllama/llama-13b c4 --wbits 8 --true-sequential --act-order --new-eval --nearest > log/llama13b_rtn_8b.txt

CUDA_VISIBLE_DEVICES=0 python llama.py huggyllama/llama-13b c4 --wbits 8 --true-sequential --act-order --new-eval --groupsize 1024 > log/llama13b_gptq_8b.txt

CUDA_VISIBLE_DEVICES=0 python bloom.py bigscience/bloom-7b1 c4 > log/bloom7.1b_full.txt

CUDA_VISIBLE_DEVICES=0 python bloom.py bigscience/bloom-7b1 c4 --wbits 4 --nearest > log/bloom7.1b_rtn.txt

CUDA_VISIBLE_DEVICES=0 python bloom.py bigscience/bloom-7b1 c4 --wbits 4 --groupsize 1024 > log/bloom7.1b_gptq.txt