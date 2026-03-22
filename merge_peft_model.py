import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

def merge_lora_model():
    # 1. 设置参数解析
    parser = argparse.ArgumentParser(description="Merge PEFT model with base model")
    parser.add_argument("--base_model", type=str, required=True, help="Path to the base model")
    parser.add_argument("--peft_model", type=str, required=True, help="Path to the LoRA/PEFT checkpoint")
    parser.add_argument("--save_location", type=str, required=True, help="Path to save the merged model")

    args = parser.parse_args()

    base_model_path = args.base_model
    peft_model_id = args.peft_model
    save_location = args.save_location

    print(f"Loading base model from: {base_model_path}")

    # 2. 加载基础模型
    # 注意：device_map="cpu" 或 "auto" 均可，合并操作通常在内存中完成
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="auto",
        trust_remote_code=True
    )

    # 3. 【核心修复】调整词表大小以匹配 LoRA 权重
    # 错误提示显示 checkpoint 里的维度是 151665，而 Qwen 默认加载是 152064
    # print("Resizing token embeddings to 151665...")
    if "Qwen2.5-Coder-7B" in base_model_path:
        model.resize_token_embeddings(151665)

    # 4. 加载 Tokenizer
    # 建议从 peft_model 目录加载，以防训练时添加了特殊 Token
    print(f"Loading tokenizer from: {peft_model_id}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(peft_model_id, trust_remote_code=True)
    except:
        print("Tokenizer not found in PEFT dir, loading from base model...")
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

    # 5. 加载 PEFT (LoRA) 模型
    print(f"Loading PEFT model from: {peft_model_id}")
    inference_model = PeftModel.from_pretrained(
        model, 
        peft_model_id,
        device_map="auto"
    )

    # 6. 合并权重
    print("Merging LoRA weights into base model...")
    merged_model = inference_model.merge_and_unload()

    # 7. 保存合并后的模型和 Tokenizer
    print(f"Saving merged model to: {save_location}")
    merged_model.save_pretrained(save_location, safe_serialization=True)
    tokenizer.save_pretrained(save_location)

    print(f"Successfully merged and saved to {save_location}!")

if __name__ == "__main__":
    merge_lora_model()
