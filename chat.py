from huggingface_hub import login
login("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")

import torch
print("CUDA available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU found")

from PIL import Image

from transformers import AutoModel, AutoTokenizer,BitsAndBytesConfig


bnb_config = BitsAndBytesConfig( load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16, )

model = AutoModel.from_pretrained( "ContactDoctor/Bio-Medical-MultiModal-Llama-3-8B-V1", quantization_config=bnb_config, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True, attn_implementation="flash_attention_2", )

tokenizer = AutoTokenizer.from_pretrained("ContactDoctor/Bio-Medical-MultiModal-Llama-3-8B-V1", trust_remote_code=True)

image = Image.open("Path to Your image").convert('RGB')

question = 'Give the modality, organ, analysis, abnormalities (if any), treatment (if abnormalities are present)?'

msgs = [{'role': 'user', 'content': [image, question]}]

res = model.chat( image=image, msgs=msgs, tokenizer=tokenizer, sampling=True, temperature=0.95, stream=True )

generated_text = ""

for new_text in res:
    generated_text += new_text
    print(new_text, flush=True, end='')
