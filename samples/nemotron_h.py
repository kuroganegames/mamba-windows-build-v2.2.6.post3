# coding:utf-8

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

dir_cache = r"I:\cache"
model_id = "nvidia/NVIDIA-Nemotron-Nano-9B-v2"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=dir_cache)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
    cache_dir=dir_cache
)


messages = [
    {"role": "system", "content": "/think"},
    {"role": "user", "content": "Write a haiku about GPUs"},
]


tokenized_chat = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

outputs = model.generate(
    tokenized_chat,
    max_new_tokens=500,
    eos_token_id=tokenizer.eos_token_id
)
print(tokenizer.decode(outputs[0]))
