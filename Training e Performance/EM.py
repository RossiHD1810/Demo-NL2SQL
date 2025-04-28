import torch
from datasets import load_from_disk
import numpy as np
from tqdm import tqdm
import re
import random
import os
from transformers import AutoTokenizer
from llama_cpp import Llama  


#Funzione per rimuovere carattere • generato da un bug della libreria llama_cpp
def clean_response(sql_response):
    return sql_response.replace('•', '').strip()

# Percorsi da configurare
model_path = "ggufmodel_llama_3.1_v3/unsloth.Q4_K_M.gguf"  
dataset_path = "merged_dataset_optimized"  

#Carico il dataset e creo la stessa partizione di test usata nel training
try:
    dataset = load_from_disk(dataset_path)
    split_dataset = dataset["train"].train_test_split(test_size=0.1, seed=3407)
    test_dataset = split_dataset["test"]
    print(f"Test dataset loaded: {len(test_dataset)} samples")
    
    #
    n=300 
    random.seed(52)  
    test_indices = random.sample(range(len(test_dataset)), n)
    test_subset = test_dataset.select(test_indices)
    print(f"N°{n} samples selected for quick testing")
    
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit(1)

# Carica il modello GGUF
try:
    
    tokenizer_path = "ggufmodel_llama_3.1_v3"
    if os.path.exists(tokenizer_path):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        print("Tokenizer loaded")
    else:
        
        tokenizer = AutoTokenizer.from_pretrained("unsloth/Meta-Llama-3.1-8B-bnb-4bit")
        print("Tokenizer loaded of the original model")

    
    model = Llama(
        model_path=model_path,
        n_ctx=2048,       
        n_batch=512,      
        n_gpu_layers=-1,  
        verbose=False     
    )
    print("GGUF Model Loaded")
except Exception as e:
    print(f"Error loading GGUF Model: {e}")
    exit(1)


def format_prompt(schema, question):
    prompt = f"""
    You are a SQL Expert. Given the following database schema, answer the question in natural language and provide the SQL query to retrieve the answer.
    ### Database_schema:
    {schema}
    ### Question:
    {question}
    ### Answer:
    """
    return prompt

# Funzione per estrarre la risposta generata dal modello
def extract_response(output_text):
    # Estraggo solo la risposta SQL dopo il "### Response:"
    match = re.search(r"#+\s*Response:\s*(.*?)(?:\s*#|$)", output_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return output_text.strip()

# Funzione di generazione con GGUF
def generate_sql(schema, question, max_tokens=256):
    prompt = format_prompt(schema, question)
    
    # Genera risposta usando llama.cpp
    response = model(
        prompt,
        max_tokens=max_tokens,
        temperature=0.1,  
        top_p=1.0,        
        top_k=1,          
        echo=False        
    )
    
    # Estraggo il testo generato
    generated_text = response["choices"][0]["text"]
    sql_response = extract_response(generated_text)
    sql_response = clean_response(sql_response)
    return sql_response
    return sql_response

# Valutazione
def compute_exact_match(predictions, references):
    exact_matches = [
        int(pred.strip() == ref.strip())
        for pred, ref in zip(predictions, references)
    ]
    return sum(exact_matches) / len(exact_matches) if exact_matches else 0

# Eseguo la valutazione sul subset
print("\nStarting EM performance testing on test dataset...")
predictions = []
references = []

for i, example in enumerate(tqdm(test_subset,ascii=True)):
    schema = example["context"]
    question = example["question"]
    reference = example["answer"]
    
    try:
        prediction = generate_sql(schema, question)
        predictions.append(prediction)
        references.append(reference)
        
        # Mostra il risultato per ogni esempio
        print(f"\n--- Example {i+1} ---")
        print(f"Input: {question}")
        print(f"Prediction: {prediction}")
        print(f"Reference : {reference}")
        print(f"Match: {prediction.strip() == reference.strip()}")
        
    except Exception as e:
        print(f"Error generating response for this sample {i}: {e}")

# Calcolo e mostro le metriche finali
em_score = compute_exact_match(predictions, references)
print(f"\n--- Final Result ---")
print(f"Number of testing samples: {len(predictions)}")
print(f"Exact Match (EM): {em_score:.4f} ({em_score*100:.2f}%)")

# Salva i risultati
with open("evaluation_results_gguf.txt", "w", encoding="utf-8") as f:
    f.write(f"Number of testing samples: {len(predictions)}\n")
    f.write(f"Exact Match (EM): {em_score:.4f} ({em_score*100:.2f}%)\n\n")
    
    # Salva tutti gli esempi testati
    f.write("--- All Samples Tested ---\n")
    for i in range(len(predictions)):
        f.write(f"Example {i+1}:\n")
        f.write(f"Input: {test_subset['question'][i]}\n")
        f.write(f"Prediction: {predictions[i]}\n")
        f.write(f"Reference : {test_subset['answer'][i]}\n")
        f.write(f"Match: {predictions[i].strip() == test_subset['answer'][i].strip()}\n\n")

print(f"Results saved on 'evaluation_results_gguf.txt'")