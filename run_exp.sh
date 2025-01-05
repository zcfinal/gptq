# CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-125m c4

# CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-125m c4 --wbits 4 --nearest

# CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-125m c4 --wbits 4 --groupsize 1024

# CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-350m c4

# CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-350m c4 --wbits 4 --nearest

# CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-350m c4 --wbits 4 --groupsize 1024

# CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-1.3b c4

# CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-1.3b c4 --wbits 4 --nearest

# CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-1.3b c4 --wbits 4 --groupsize 1024

CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-2.7b c4 > log/opt2.7b_full.txt

CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-2.7b c4 --wbits 4 --nearest > log/opt2.7b_rtn.txt

CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-2.7b c4 --wbits 4 --groupsize 1024 > log/opt2.7b_gptq.txt