from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict, load_dataset, interleave_datasets, load_from_disk
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig, TrainingArguments, Trainer
import torch
import time
import evaluate
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")


#si occupa di trasformare dati testuali in un formato numerico adatto a dare in pasto alla LLM per l'addestramento
def tokenize_function(example):
    
    start_prompt = "Tables:\n"
    middle_prompt = "\n\nQuestion:\n"
    end_prompt = "\n\nAnswer:\n"
  
    data_zip = zip(example['context'], example['question'])
    prompt = [start_prompt + context + middle_prompt + question + end_prompt for context, question in data_zip]
    example['input_ids'] = tokenizer(prompt, padding="max_length", truncation=True, return_tensors="pt").input_ids
    example['labels'] = tokenizer(example['answer'], padding="max_length", truncation=True, return_tensors="pt").input_ids

    return example


print(torch.cuda.is_available())
print(torch.version.cuda)
model_name='t5-small'
tokenizer=AutoTokenizer.from_pretrained(model_name)
original_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
original_model = original_model.to('cuda')


#carico i due dataset
try:
    dataset = load_from_disk("merged_dataset")
    print("Loaded Merged Dataset")
except:
    dataset_scc_train = load_dataset("b-mc2/sql-create-context", split='train[:80%]')
    dataset_scc_test  = load_dataset("b-mc2/sql-create-context", split='train[-20%:-10%]')
    dataset_scc_val   = load_dataset("b-mc2/sql-create-context", split='train[-10%:]')

    dataset_tts_train = load_dataset("Clinton/Text-to-sql-v1", split='train[:80%]')
    dataset_tts_train = dataset_tts_train.remove_columns(['source', 'text'])
    dataset_tts_train = dataset_tts_train.rename_columns({'instruction': 'question', 'input': 'context', 'response': 'answer'})
    dataset_tts_test  = load_dataset("Clinton/Text-to-sql-v1", split='train[-20%:-10%]')
    dataset_tts_test  = dataset_tts_test.remove_columns(['source', 'text'])
    dataset_tts_test  = dataset_tts_test.rename_columns({'instruction': 'question', 'input': 'context', 'response': 'answer'})
    dataset_tts_val   = load_dataset("Clinton/Text-to-sql-v1", split='train[-10%:]')
    dataset_tts_val   = dataset_tts_val.remove_columns(['source', 'text'])
    dataset_tts_val   = dataset_tts_val.rename_columns({'instruction': 'question', 'input': 'context', 'response': 'answer'})

    dataset_ks_train  = load_dataset("knowrohit07/know_sql", split='validation[:80%]')
    dataset_ks_test   = load_dataset("knowrohit07/know_sql", split='validation[-20%:-10%]')
    dataset_ks_val    = load_dataset("knowrohit07/know_sql", split='validation[-10%:]')

    dataset = DatasetDict({ 'train': interleave_datasets([dataset_scc_train, dataset_tts_train, dataset_ks_train]),
                            'test': interleave_datasets([dataset_scc_test, dataset_tts_test, dataset_ks_test]),
                            'validation': interleave_datasets([dataset_scc_val, dataset_tts_val, dataset_ks_val])})

    dataset.save_to_disk("merged_dataset")
    print("Merged and Saved Dataset")

try:
    tokenized_datasets = load_from_disk("tokenized_datasets")
    print("Loaded Tokenized Dataset")
except:
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(['question', 'context', 'answer'])
    
    tokenized_datasets.save_to_disk("tokenized_datasets")
    print("Tokenized and Saved Dataset")

print(f"Shapes of the datasets:")
print(f"Training: {tokenized_datasets['train'].shape}")
print(f"Validation: {tokenized_datasets['validation'].shape}")
print(f"Test: {tokenized_datasets['test'].shape}")

print(tokenized_datasets)

#testing senza training
index=0
question=dataset['test'][index]['question']
context=dataset['test'][index]['context']
answer=dataset['test'][index]['answer']

prompt=f"""Tables:
{context}
Question:{question}
Answer:
"""
inputs=tokenizer(prompt,return_tensors='pt')
inputs=inputs.to('cuda')
output=tokenizer.decode(
    original_model.generate(
        inputs["input_ids"],
        max_new_tokens=200,
    )[0],
    skip_special_tokens=True
)
dash_line='-'.join('' for x in range(100))
print(dash_line)
print(f'INPUT PROMPT:\n{prompt}')
print(dash_line)
print(f'BASELINE HUMAN ANSWER: \n{answer}')
print(dash_line)
print(f'MODEL GENERATION - ZERO SHOT:\n{output}')

#-----Fine-Tuning-----
finetuned_model=AutoModelForSeq2SeqLM.from_pretrained(model_name,torch_dtype=torch.bfloat16)
finetuned_model=finetuned_model.to('cuda')
tokenizer=AutoTokenizer.from_pretrained(model_name)

output_dir=f'./sql-training-1741776900'
training_args=TrainingArguments(
    output_dir=output_dir,
    #resume_from_checkpoint="C:\Users\rossi\OneDrive\Documents\Universita\Tirocinio Intergea\Demo_LangChain\sql-training-1741776900\checkpoint-5500",
    learning_rate=5e-3,
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    logging_steps=50,
    evaluation_strategy='steps',
    eval_steps=500,
)
trainer=Trainer(
    model=finetuned_model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
)

trainer.train(resume_from_checkpoint=True)


