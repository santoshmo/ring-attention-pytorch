from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

# 4-bit quant config
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    llm_int8_enable_fp32_cpu_offload=True  # keep CPU-offloaded modules in fp32
)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-405B",
    quantization_config=bnb_cfg,
    device_map="auto"       # HF will shard / offload automatically
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-405B")

# Now you can run inference on a single 96 GB GPU
inputs = tokenizer("Hello, world!", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=64)
print(tokenizer.decode(outputs[0]))
