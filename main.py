from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict, load_dataset, interleave_datasets, load_from_disk
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig, TrainingArguments, Trainer, pipeline
import torch
import time
import evaluate
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")
PYTORCH_CUDA_ALLOC_CONF=expandable_segments=True


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


#print(torch.cuda.is_available())
#print(torch.version.cuda)
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

    dataset_synt_train = load_dataset("gretelai/synthetic_text_to_sql", split='train[:80%]')
    dataset_synt_train = dataset_synt_train.remove_columns(['id','domain','domain_description','sql_complexity','sql_complexity_description','sql_task_type','sql_task_type_description','sql_explanation'])
    dataset_synt_train = dataset_synt_train.rename_columns({'sql_prompt': 'question', 'sql_context': 'context', 'sql': 'answer'})
    dataset_synt_test  = load_dataset("gretelai/synthetic_text_to_sql", split='train[-20%:-10%]')
    dataset_synt_test = dataset_synt_test.remove_columns(['id','domain','domain_description','sql_complexity','sql_complexity_description','sql_task_type','sql_task_type_description','sql_explanation'])
    dataset_synt_test = dataset_synt_test.rename_columns({'sql_prompt': 'question', 'sql_context': 'context', 'sql': 'answer'})
    dataset_synt_val   = load_dataset("gretelai/synthetic_text_to_sql", split='train[-10%:]')
    dataset_synt_val = dataset_synt_val.remove_columns(['id','domain','domain_description','sql_complexity','sql_complexity_description','sql_task_type','sql_task_type_description','sql_explanation'])
    dataset_synt_val = dataset_synt_val.rename_columns({'sql_prompt': 'question', 'sql_context': 'context', 'sql': 'answer'})
    print(dataset_synt_train.head(5))
    print(dataset_synt_val.head(5))

    dataset = DatasetDict({ 'train': interleave_datasets([dataset_scc_train, dataset_tts_train, dataset_ks_train,dataset_synt_train]),
                            'test': interleave_datasets([dataset_scc_test, dataset_tts_test, dataset_ks_test,dataset_synt_test]),
                            'validation': interleave_datasets([dataset_scc_val, dataset_tts_val, dataset_ks_val,dataset_synt_val])})

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



#-----Fine-Tuning-----
try:
    finetuned_model = AutoModelForSeq2SeqLM.from_pretrained("finetuned_codet5_2epoch")
    finetuned_model = finetuned_model.to('cuda')
    to_train = False

except:
    to_train = True
    finetuned_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    finetuned_model = finetuned_model.to('cuda')
    tokenizer = AutoTokenizer.from_pretrained(model_name)

output_dir = f'./sql-training-code-t5-2epoch'
if to_train:
    training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=4,
    )

    trainer = Trainer(
        model=finetuned_model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
    )
    
    trainer.train()
    
    finetuned_model.save_pretrained("finetuned_codet5_2epoch")
    tokenizer.save_pretrained("finetuned_codet5_2epoch")

#tokenizer = AutoTokenizer.from_pretrained("mohsin-aslam/text2sql-finetuned-60M")
#model = AutoModelForSeq2SeqLM.from_pretrained("mohsin-aslam/text2sql-finetuned-60M")
#model = model.to('cuda')
#finetuned_model = finetuned_model.to('cuda')
model = AutoModelForSeq2SeqLM.from_pretrained("mohsin-aslam/text2sql-finetuned-60M")
model=model.to('cuda')

#index indica l'indice del dataset da testare
index=2
#testing senza training (Zero Shot Inferencing)
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
print(f'#####Zero_Shot#####')
print(f'INPUT PROMPT:\n{prompt}')
print(dash_line)
print(f'BASELINE HUMAN ANSWER: \n{answer}')
print(dash_line)
print(f'MODEL GENERATION - ZERO SHOT:\n{output}')

#Risposta con modello finetuned
question = dataset['test'][index]['question']
context = dataset['test'][index]['context']
answer = dataset['test'][index]['answer']

prompt = f"""Tables:
{context}

Question:
{question}

Answer:
"""

inputs = tokenizer(prompt, return_tensors='pt')
inputs = inputs.to('cuda')

output = tokenizer.decode(
    model.generate(
        inputs["input_ids"], 
        max_new_tokens=200,
    )[0], 
    skip_special_tokens=True
)

dash_line = '-'.join('' for x in range(100))
print(dash_line)
print(f'#####Fine_Tuned#####')
print(f'INPUT PROMPT:\n{prompt}')
print(dash_line)
print(f'BASELINE HUMAN ANSWER:\n{answer}\n')
print(dash_line)
print(f'MODEL GENERATION - Fine_Tuned:\n{output}')

#-----Valutazione Performance-----
question=dataset['test'][0:14000]['question']
context=dataset['test'][0:14000]['context']
human_baseline_answers=dataset['test'][0:14000]['answer']

original_model_answers=[]
model_answers=[]

for idx, question in enumerate(question):
    prompt=f"""Tables:
    {context[idx]}
    Question:{question}
    Answer:
    """
input_ids=tokenizer(prompt,return_tensors='pt').input_ids
input_ids=input_ids.to('cuda')

human_baseline_text_output=human_baseline_answers[idx]

original_model_outputs = original_model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=300))
original_model_text_output = tokenizer.decode(original_model_outputs[0], skip_special_tokens=True)
original_model_answers.append(original_model_text_output)

model_outputs = model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=300))
model_text_output = tokenizer.decode(model_outputs[0], skip_special_tokens=True)
model_answers.append(model_text_output)

zipped_summaries=list(zip(human_baseline_answers,original_model_answers,model_answers))
df=pd.DataFrame(zipped_summaries,columns=['human_baseline_answers','original_model_answers','finetuned_model_answers'])

rouge=evaluate.load('rouge')

original_model_results=rouge.compute(predictions=original_model_answers,references=human_baseline_answers[0:len(original_model_answers)],
use_aggregator=True,use_stemmer=True)
print('Original Model:')
print(original_model_results)

finetuned_model_results=rouge.compute(predictions=model_answers,references=human_baseline_answers[0:len(model_answers)],
use_aggregator=True,use_stemmer=True)
print('Fine-Tuned Model:')
print(finetuned_model_results)



