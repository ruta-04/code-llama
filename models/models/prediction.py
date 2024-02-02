import torch
from transformers import AutoTokenizer
import transformers
from huggingface_hub.hf_api import HfFolder

token="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"
HfFolder.save_token(token)

tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-Instruct-hf", token=True)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
pipeline = transformers.pipeline(
    "text-generation",
    model="codellama/CodeLlama-7b-Instruct-hf",
    torch_dtype=torch.float16,
    device_map="auto",
    token=True,
    pad_token_id=tokenizer.eos_token_id,
)

def predict(input):  
    #temprature and top_p generally shouldn't be modified at the same time. they could cancel out each other
    sequences = pipeline(
        input,
        do_sample=True,
        temperature=0.2,
        #top_p=0.9,
        eos_token_id=tokenizer.eos_token_id,
        max_length=100,
    )
    return sequences[0]['generated_text']
    # for seq in sequences:
    #     print(f"Result: {seq['generated_text']}") 

#print(predict("def fibonacci("))