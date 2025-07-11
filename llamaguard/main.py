import argparse
import os
import json
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve, auc

# Function to evaluate guardrails
def evaluate_guardrail(model, tokenizer, ids, prompts, labels, categories, lanugages, dataset_name):
    dataset_results = []
    preds = []

    formatted_prompts = transform_dataset_prompt(prompts)

    for chat in tqdm(formatted_prompts):
        pred = moderate(chat, tokenizer, model)
        print(pred)
        if 'unsafe' in pred.lower():
            preds.append(1)
        else:
            preds.append(0)

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    print(acc)
    print("---")
    print(f1)

    for idx, prompt in enumerate(prompts):
        result = {
            'prompt': prompt,
            'category': categories[idx],
            'ID': ids[idx],
            'pred': preds[idx],
            'labels': labels[idx],
            'languages': lanugages[idx]
        }
        dataset_results.append(result)
    result_df = pd.DataFrame(dataset_results)
    result_df.to_csv("./llamaguard_results.csv")


# Define a function to transform dataset prompts
def transform_dataset_prompt(prompts):
    transformed_data=[]
    for x in prompts:
        transformed_data.append([{"role": "user", "content": x}])
    return transformed_data

# Define the moderation function
def moderate(chat, tokenizer, model, device='cuda'):
    input_ids = tokenizer.apply_chat_template(chat, return_tensors="pt").to(device)
    output = model.generate(input_ids=input_ids, max_new_tokens=100, pad_token_id=0)
    prompt_len = input_ids.shape[-1]
    return tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)

def main():
    test_df = pd.read_csv("../data/test.csv")
    
    test_plain = test_df["prompt"].tolist()#[:100]
    categories = test_df['category'].tolist()#[:100]
    languages = test_df['target_language'].tolist()#[:100]
    ids = test_df['ID'].tolist()#[:100]
    labels = test_df['label_index'].tolist()#[:100]
    labels = [int(lab) for lab in labels]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    hf_token = ''#TODO Replace with your token
    model_id = 'meta-llama/Llama-Guard-3-8B' # 'meta-llama/Llama-Guard-3-8B', 'meta-llama/Llama-Guard-3-1B'

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(model_id, token=hf_token, torch_dtype=dtype, device_map=device)

    evaluate_guardrail(model, tokenizer, ids, test_plain, labels, categories, languages, 'testing_set')
    print("done")

if __name__ == '__main__':
    main()