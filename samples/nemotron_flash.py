# coding:utf-8

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

dir_cache = r"I:\cache"
model_id = "nvidia/Nemotron-Flash-3B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, cache_dir=dir_cache)
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, cache_dir=dir_cache)
model = model.cuda().to(torch.bfloat16)

max_new_tokens = 256

print('Initializing generation state...')
generation_state = model.init_cuda_graph_generation(
    max_new_tokens=max_new_tokens,
    batch_size=1,
    device='cuda',
)

prompt = input("User:")
prompt = "User: " + prompt + "\nAssistant:"
inputs = tokenizer(prompt, return_tensors="pt").to('cuda')

print(f"Generating with CUDA graph acceleration...")
outputs = model.generate_with_cuda_graph(
    input_ids=inputs["input_ids"],
    generation_state=generation_state,
    max_new_tokens=max_new_tokens,
    temperature=0,
    eos_token_id=tokenizer.eos_token_id,
)

response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
print(f"Response: {response}")
