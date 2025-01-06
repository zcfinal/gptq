import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig
import time

def load_models(model_id):
    """加载fp16模型和GPTQ量化模型"""
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # 加载fp16模型
    fp16_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # 配置GPTQ
    dataset = ["auto-gptq is an easy-to-use model quantization library with user-friendly apis."]
    gptq_config = GPTQConfig(
        bits=4,
        dataset=dataset,
        tokenizer=tokenizer,
        exllama_config={"version": 2}  # 使用ExLlamaV2加速
    )
    
    # 加载GPTQ量化模型
    gptq_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        quantization_config=gptq_config
    )
    
    return tokenizer, fp16_model, gptq_model

def generate_text(model, tokenizer, prompt, max_length=100):
    """生成文本"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
        )
    end_time = time.time()
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generation_time = end_time - start_time
    
    return generated_text, generation_time

def evaluate_models(model_id, test_prompts):
    """评测fp16和GPTQ模型"""
    print(f"Loading models from {model_id}...")
    tokenizer, fp16_model, gptq_model = load_models(model_id)
    
    results = []
    
    for i, prompt in enumerate(test_prompts):
        print(f"\nProcessing prompt {i+1}/{len(test_prompts)}")
        print(f"Prompt: {prompt}")
        
        # FP16推理
        print("\nFP16 Model Generation:")
        fp16_text, fp16_time = generate_text(fp16_model, tokenizer, prompt)
        print(f"Generated text: {fp16_text}")
        print(f"Generation time: {fp16_time:.2f} seconds")
        
        # GPTQ推理
        print("\nGPTQ Model Generation:")
        gptq_text, gptq_time = generate_text(gptq_model, tokenizer, prompt)
        print(f"Generated text: {gptq_text}")
        print(f"Generation time: {gptq_time:.2f} seconds")
        
        results.append({
            "prompt": prompt,
            "fp16_result": {
                "text": fp16_text,
                "time": fp16_time
            },
            "gptq_result": {
                "text": gptq_text,
                "time": gptq_time
            }
        })
        
    return results

def main():
    # 设置测试参数
    model_id = "huggyllama/llama-13b"
    # model_id = "facebook/opt-6.7b"  # 可以替换为其他模型
    # model_id = "facebook/opt-125m"
    test_prompts = [
        "The best way to learn programming is",
        "In the future, artificial intelligence will",
        "The most important invention in history is"
    ]
    
    # 运行评测
    try:
        results = evaluate_models(model_id, test_prompts)
        
        # 打印汇总结果
        print("\n=== Summary ===")
        fp16_times = [r["fp16_result"]["time"] for r in results]
        gptq_times = [r["gptq_result"]["time"] for r in results]
        
        print(f"Average FP16 generation time: {sum(fp16_times)/len(fp16_times):.2f} seconds")
        print(f"Average GPTQ generation time: {sum(gptq_times)/len(gptq_times):.2f} seconds")
        print(f"GPTQ speedup: {sum(fp16_times)/sum(gptq_times):.2f}x")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()