import time, torch
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B-Base",
    torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
m.gradient_checkpointing_enable(); m.enable_input_require_grads()
m = get_peft_model(m, LoraConfig(r=16, lora_alpha=64, lora_dropout=0.1, target_modules=
    ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
m.train()
x = torch.randint(0, 150000, (2, 3000), device="cuda")
for i in range(4):
    torch.cuda.synchronize(); t = time.time()
    h = m.base_model.model.model(input_ids=x).last_hidden_state
    h.float().pow(2).mean().backward()
    torch.cuda.synchronize(); print(f"step {i}: {time.time()-t:.2f} s")
print("peak GB", torch.cuda.max_memory_allocated() / 1e9)
