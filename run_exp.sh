# CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-125m c4

# CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-125m c4 --wbits 4 --nearest

CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-125m c4 --wbits 4 --groupsize 1024