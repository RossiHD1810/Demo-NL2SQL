# -*- coding: utf-8 -*-
from unsloth import FastLanguageModel, is_bfloat16_supported
import torch
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import re
import random
import os
from transformers import AutoTokenizer

from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk


try:
   
    nltk.data.find('tokenizers/punkt')
except LookupError: 
    
    print("LookupError caught. Downloading NLTK 'punkt' tokenizer data...")
    nltk.download('punkt')
    print("Download completato.")


model_output_dir_phase2 = "./output_phase2/checkpoint-6100"
# Percorso del dataset JSON della Fase 2
phase2_dataset_path = "2_dataset.json"
# Percorso dello schema SQL Server semplificato
schema_file_path = "schemi_e_report/Views_Owner_BI_simplified.txt"

max_seq_length = 8192
load_in_4bit = True
dtype = None


def clean_response(sql_response):
    cleaned = sql_response
    if "### Answer:" in cleaned:
        cleaned = cleaned.split("### Answer:", 1)[-1].strip()
    stop_sequences = ["###", "\n\n", "```"]
    for stop in stop_sequences:
        if stop in cleaned:
            cleaned = cleaned.split(stop, 1)[0].strip()
    return cleaned


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

# --- Caricamento Schema Fisso ---
try:
    with open(schema_file_path, 'r', encoding='utf-8') as f:
        fixed_schema_sql_server = f.read()
    print(f"Caricato schema fisso SQL Server da: {schema_file_path}")
except FileNotFoundError:
    print(f"ERRORE: File dello schema '{schema_file_path}' non trovato.")
    exit(1)
except Exception as e:
    print(f"Errore durante la lettura del file schema '{schema_file_path}': {e}")
    exit(1)

# --- Carico il dataset JSON e ricreo la partizione di test della Fase 2 ---
try:
    dataset_sqlserver_full = load_dataset("json", data_files=phase2_dataset_path, split="train")
    split_dataset1_p2 = dataset_sqlserver_full.train_test_split(test_size=0.1, seed=3407)
    temp_eval_test_p2 = split_dataset1_p2['test']
    split_dataset2_p2 = temp_eval_test_p2.train_test_split(test_size=0.5, seed=3407)
    final_test_set = split_dataset2_p2['test']

    if final_test_set is None or len(final_test_set) == 0:
        print("Errore critico: Impossibile ottenere un dataset di test.")
        exit(1)

    print(f"Test dataset (Phase 2) loaded: {len(final_test_set)} samples available")

except Exception as e:
    print(f"Errore durante il caricamento o la suddivisione del dataset JSON: {e}")
    exit(1)


print(f"Caricamento modello e tokenizer da: {model_output_dir_phase2}")
try:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_output_dir_phase2,
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )
    if hasattr(model, 'is_peft_model') and model.is_peft_model:
         print("Modello PEFT (con adapter) caricato con successo.")
    else:
         print("Attenzione: Modello caricato, ma potrebbe non aver caricato gli adapter PEFT.")

except Exception as e:
    print(f"Errore caricando il modello da '{model_output_dir_phase2}': {e}")
    exit(1)

# Imposta pad_token se mancante
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print("Tokenizer pad_token impostato a eos_token")

print("Modello e Tokenizer caricati.")


@torch.inference_mode()
def generate_sql(fixed_schema, question, max_new_tok=512): 
    prompt = format_prompt(fixed_schema, question)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_seq_length).to("cuda")

   
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=max_new_tok,
        use_cache=True,
        do_sample=False, 
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    start_index = inputs["input_ids"].shape[1]
    prediction = tokenizer.decode(outputs[0, start_index:], skip_special_tokens=True)
    sql_response = clean_response(prediction)
    return sql_response

# --- Funzioni di Calcolo Metriche ---

# Valutazione Exact Match (Accuracy)
def compute_exact_match(predictions, references):
    if len(predictions) != len(references):
        print("Attenzione: Numero di predizioni e riferimenti non corrisponde!")
        return 0
    if not predictions:
        return 0
    exact_matches = [
        int(pred.strip().rstrip(';').lower() == ref.strip().rstrip(';').lower())
        for pred, ref in zip(predictions, references)
    ]
    return np.mean(exact_matches) if exact_matches else 0.0

# Valutazione ROUGE-L
def compute_rouge_l(predictions, references):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = []
    for pred, ref in zip(predictions, references):
        pred_str = pred if pred else " "
        ref_str = ref if ref else " "
        score = scorer.score(ref_str, pred_str) 
        scores.append(score['rougeL'].fmeasure) 
    return np.mean(scores) if scores else 0.0

# Valutazione BLEU
def compute_bleu(predictions, references):
    chencherry = SmoothingFunction().method1
    scores = []
    for pred, ref in zip(predictions, references):
        pred_tokens = pred.split()
        ref_tokens = ref.split()
        ref_tokens_list = [ref_tokens]

        if not pred_tokens or not ref_tokens_list[0]:
            scores.append(0.0)
            continue


        try:
            score = sentence_bleu(ref_tokens_list, pred_tokens,
                                  smoothing_function=chencherry,
                                  weights=(0.25, 0.25, 0.25, 0.25)) 
        except ZeroDivisionError:
            score = 0.0
        scores.append(score)
    return np.mean(scores) if scores else 0.0

# --- Esecuzione Valutazione ---
print(f"\nAvvio valutazione su {len(final_test_set)} campioni...")
predictions = []
references = []
errors = 0

for i, example in enumerate(tqdm(final_test_set, ascii=True)):
    schema = fixed_schema_sql_server
    question = example["question"]
    reference = example["answer"]

    try:
        prediction = generate_sql(schema, question, max_new_tok=768) 
        predictions.append(prediction)
        references.append(reference)

        if (i + 1) % 20 == 0:
             print(f"\n--- Esempio {i+1} ---")
             print(f"Domanda: {question}")
             print(f"Predizione:\n{prediction}")
             print(f"Riferimento:\n{reference}")
             print(f"Match: {prediction.strip().rstrip(';').lower() == reference.strip().rstrip(';').lower()}")

    except Exception as e:
        print(f"\nErrore durante la generazione per il campione {i}: {e}")
        
        predictions.append("ERROR")
        references.append(reference)
        errors += 1

# --- Calcolo e Stampa Metriche Finali ---
em_score = 0.0
rouge_l_score = 0.0
bleu_score = 0.0
num_total = len(references)
num_errors = errors
num_valid = 0

if num_total > num_errors:
    valid_preds = [p for p, r in zip(predictions, references) if p != "ERROR"]
    valid_refs = [r for p, r in zip(predictions, references) if p != "ERROR"]
    num_valid = len(valid_preds)

    if num_valid > 0:
        print(f"\nCalculating metrics on {num_valid} valid samples...")
        # Exact Match (Accuracy)
        em_score = compute_exact_match(valid_preds, valid_refs)

        # ROUGE-L
        rouge_l_score = compute_rouge_l(valid_preds, valid_refs)

        # BLEU Score
        bleu_score = compute_bleu(valid_preds, valid_refs)

        print(f"\n--- Risultato Finale ---")
        print(f"Numero campioni totali nel subset: {num_total}")
        print(f"Numero errori durante generazione: {num_errors}")
        print(f"Numero campioni valutati (senza errori): {num_valid}")
        print(f"Exact Match (EM) / Accuracy: {em_score:.4f} ({em_score*100:.2f}%)")
        print(f"ROUGE-L (F1): {rouge_l_score:.4f}")
        print(f"BLEU Score: {bleu_score:.4f}")
    else:
        print("\nErrore: Nessun campione valido è stato generato per la valutazione (tutti errori?).")
else:
     print(f"\nErrore: Nessun campione nel dataset di test o tutti hanno generato errori ({num_errors}/{num_total}).")


# --- Salvataggio Risultati ---
output_results_file = "evaluation_results_phase2_multi_metric.txt"
with open(output_results_file, "w", encoding="utf-8") as f:
    f.write(f"Modello Valutato: {model_output_dir_phase2}\n")
    f.write(f"Dataset Valutato: {phase2_dataset_path} (subset di test Fase 2)\n")
    f.write(f"Schema Usato: {schema_file_path}\n")
    f.write(f"Numero campioni totali nel subset: {num_total}\n")
    f.write(f"Numero errori generazione: {num_errors}\n")
    f.write(f"Numero campioni valutati: {num_valid}\n\n")
    f.write("--- Metriche Medie ---\n")
    f.write(f"Exact Match (EM) / Accuracy: {em_score:.4f} ({em_score*100:.2f}%)\n")
    f.write(f"ROUGE-L (F1): {rouge_l_score:.4f}\n")
    f.write(f"BLEU Score: {bleu_score:.4f}\n\n")

    f.write("--- Dettaglio Campioni ---\n")
    for i in range(len(references)):
        current_example = final_test_set[i] 
        question_text = current_example['question']
        reference_text = references[i] 
        prediction_text = predictions[i] 

        match_status = "ERROR"
        if prediction_text != "ERROR":
            is_match = prediction_text.strip().rstrip(';').lower() == reference_text.strip().rstrip(';').lower()
            match_status = str(is_match)

        f.write(f"--- Esempio {i+1} ---\n")
        f.write(f"Domanda: {question_text}\n")
        f.write(f"Predizione:\n{prediction_text}\n")
        f.write(f"Riferimento:\n{reference_text}\n")
        f.write(f"Match (EM): {match_status}\n\n") 

print(f"Risultati dettagliati salvati in '{output_results_file}'")