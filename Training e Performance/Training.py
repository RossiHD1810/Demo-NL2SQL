from unsloth import FastLanguageModel, is_bfloat16_supported,unsloth_train
from unsloth.chat_templates import get_chat_template
from datasets import load_from_disk,load_dataset,DatasetDict,interleave_datasets
from trl import SFTTrainer
from transformers import TrainingArguments,TrainerState,TrainerControl,TrainerCallback
from unsloth import is_bfloat16_supported
import torch
import os
import numpy as np
os.environ["TORCHINDUCTOR_CACHE_DIR"] = "C:\\Users\\rossi\\torchinductor_cache"



#Funzione che assembla i prompts usati per l'addestramento del modello
def formatting_prompts_func_phase1(examples,schema_col, question_col, answer_col):
    database_schema = examples["context"]
    inputs       = examples["question"]
    outputs      = examples["answer"]
    texts = []
    for schema, input, output in zip(database_schema, inputs, outputs):
        text = alpaca_prompt.format(schema, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }

def formatting_prompts_func_phase2(examples, fixed_schema, question_col, answer_col):
    inputs          = examples[question_col]
    outputs         = examples[answer_col]
    texts = []
    for input, output in zip(inputs, outputs):
        text = alpaca_prompt.format(fixed_schema, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }

def get_average_token_length(dataset, tokenizer, text_column="text"):
    if not dataset or len(dataset) == 0:
        print(f"Warning: empty or null dataset.")
        return 0, 0, 0
    print(f"Calculating token lenght for {len(dataset)} sample...")
    
    tokenized_lengths = [len(tokenizer.encode(text, add_special_tokens=False)) for text in dataset[text_column]] 
    if not tokenized_lengths:
         print(f"Warning: No token lenght calculated.")
         return 0, 0, 0
    avg_length = np.mean(tokenized_lengths)
    min_length = np.min(tokenized_lengths)
    max_length = np.max(tokenized_lengths)
    print(f"Token Lenght - Min: {min_length}, Max: {max_length}, Avg: {avg_length:.2f}")
    return min_length, max_length, avg_length

alpaca_prompt="""
    You are a SQL Expert. Given the following database schema, answer the question in natural language and provide the SQL query to retrieve the answer.
    ### Database_schema:
    {}
    ### Question:
    {}
    ### Answer:
    {}
    """
   
if __name__ == '__main__':
    
    #Parametri Modello
    max_seq_length=8192
    dtype=None
    load_in_4bit=True
    model_name = "unsloth/Meta-Llama-3.1-8B-bnb-4bit"
    output_dir_phase1 = "./output_phase1"
    output_dir_phase2 = "./output_phase2"
    gguf_final_model_name = "ggufmodel_llama_4_phase2_sqlserver"

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit
    )

    model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias='none',
    use_gradient_checkpointing="unsloth",
    random_state=3047,
    use_rslora=False,
    loftq_config=None,
    )
   
    model=FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj","gate_proj", "up_proj", "down_proj",],
        lora_alpha=16,
        lora_dropout=0,
        bias='none',
        use_gradient_checkpointing="unsloth",
        random_state=3047,
        use_rslora=False,
        loftq_config=None,)
    EOS_TOKEN = tokenizer.eos_token
    
        # --- FASE 1: TRAINING SU DATASET SQLITE ---
    print("--- START PHASE 1: TRAINING SQLITE ---")

    dataset_sqlite_path = "merged_dataset_optimized"
    try:
        dataset_sqlite = load_from_disk(dataset_sqlite_path)
        print(f"Loaded Saved SQLite Dataset from {dataset_sqlite_path}")
    except Exception as e:
        print(f"Error loading saved SQLite dataset from {dataset_sqlite_path}: {e}")
        print("Attempting to download/create the SQLite dataset...")
        dataset_tts_train = load_dataset("Clinton/Text-to-sql-v1", split='train[:100%]')
        dataset_tts_train = dataset_tts_train.remove_columns(['source', 'text'])
        dataset_tts_train = dataset_tts_train.rename_columns({'instruction': 'question', 'input': 'context', 'response': 'answer'})
        dataset_sqlite = DatasetDict({'train': interleave_datasets([dataset_tts_train])})
        dataset_sqlite.save_to_disk(dataset_sqlite_path)
        print(f"Downloaded/created and saved SQLite dataset to {dataset_sqlite_path}")

    split_dataset1 = dataset_sqlite["train"].train_test_split(test_size=0.1, seed=3407)
    train_dataset_raw_p1 = split_dataset1['train']
    temp_eval_test_p1 = split_dataset1['test']
    split_dataset2_p1 = temp_eval_test_p1.train_test_split(test_size=0.5, seed=3407)
    eval_dataset_raw_p1 = split_dataset2_p1['train'].select(range(min(100, len(split_dataset2_p1['train']))))

    schema_col_p1 = "context"
    question_col_p1 = "question"
    answer_col_p1 = "answer"

    train_dataset_p1 = train_dataset_raw_p1.map(
        formatting_prompts_func_phase1,
        batched=True,
        num_proc=1,
        fn_kwargs={
            "schema_col": schema_col_p1,
            "question_col": question_col_p1,
            "answer_col": answer_col_p1
        }
    )
    eval_dataset_p1 = eval_dataset_raw_p1.map(
        formatting_prompts_func_phase1,
        batched=True,
        num_proc=1,
        fn_kwargs={
            "schema_col": schema_col_p1,
            "question_col": question_col_p1,
            "answer_col": answer_col_p1
        }
    )
    print("Example formatted prompt Phase 1:")
    print(train_dataset_p1[3]["text"])

    training_args_p1 = TrainingArguments(
        num_train_epochs=2,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        warmup_steps=300,
        max_steps=35200,
        learning_rate=3e-5,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=50,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        output_dir=output_dir_phase1,
        eval_strategy='steps',
        eval_steps=500, 
        per_device_eval_batch_size=2,
        report_to="none",
    )

    trainer_p1 = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset_p1,
        eval_dataset=eval_dataset_p1,
        dataset_text_field="text",
        dataset_num_proc=1,
        max_seq_length=max_seq_length,
        packing=False,
        args=training_args_p1,
    )

    print("Star Phase 1 Training...")
    trainer_stats_p1 = trainer_p1.train(resume_from_checkpoint=True)
    print("Completed Phase 1 Training.")
    
    schema_file_path = ".schemi_e_report/Views_Owner_BI_simplified.txt"
    try:
        with open(schema_file_path, 'r', encoding='utf-8') as f:
            fixed_schema_sql_server = f.read()
        print(f"Loaded 2 Phase schema: {schema_file_path}")
        
    except FileNotFoundError:
        print(f"Error schema '{schema_file_path}' not found.")
        exit()
    except Exception as e:
        print(f"Error during schema reading '{schema_file_path}': {e}")
        exit()

    # Caricamento Dataset Fase 2 (JSON)
    second_dataset_path = "2_dataset.json"
    question_col_p2 = "question"           
    answer_col_p2 = "answer"             

    try:
        
        dataset_sqlserver = load_dataset("json", data_files=second_dataset_path, split="train")
        print(f"Loaded 2 training dataset from: {second_dataset_path}")
        required_cols_p2 = {question_col_p2, answer_col_p2}
        if not required_cols_p2.issubset(dataset_sqlserver.column_names):
            raise ValueError(f"Json dataset does not contain le requested columns: {required_cols_p2}. Columns in the dataset: {dataset_sqlserver.column_names}")

    except Exception as e:
        print(f"Error loading 2 training dataset from: {second_dataset_path}: {e}")
        exit()

    
    split_dataset1_p2 = dataset_sqlserver.train_test_split(test_size=0.1, seed=3407) 
    train_dataset_raw_p2 = split_dataset1_p2['train']
    temp_eval_test_p2 = split_dataset1_p2['test']

    split_dataset2_p2 = temp_eval_test_p2.train_test_split(test_size=0.5, seed=3407) 
    eval_dataset_raw_p2 = split_dataset2_p2['train']
        
    #eval_dataset_raw_p2 = eval_dataset_raw_p2.select(range(min(50, len(eval_dataset_raw_p2)))) 
    print(f"Dataset Fase 2 - Train: {len(train_dataset_raw_p2)}, Eval: {len(eval_dataset_raw_p2)}")


    # Applica la formattazione specifica della Fase 2 usando lo schema fisso
    train_dataset_p2 = train_dataset_raw_p2.map(
        formatting_prompts_func_phase2, 
        batched=True,
        num_proc=1,
        fn_kwargs={ 
            "fixed_schema": fixed_schema_sql_server,
            "question_col": question_col_p2,
            "answer_col": answer_col_p2
        }
    )
    eval_dataset_p2 = eval_dataset_raw_p2.map(
        formatting_prompts_func_phase2, 
        batched=True,
        num_proc=1,
        fn_kwargs={
            "fixed_schema": fixed_schema_sql_server,
            "question_col": question_col_p2,
            "answer_col": answer_col_p2
        }
    )
    print("Example for 2 Phase prompt:")
    print(train_dataset_p2[0]["text"])

    print("\n--- Sequence lenght for 2 Phase  ---")
    avg_len_train_p2 = get_average_token_length(train_dataset_p2, tokenizer)

    # Argomenti Training Fase 2 (Fine-tuning)
    training_args_p2 = TrainingArguments(
        num_train_epochs=1,
        per_device_train_batch_size=1, 
        gradient_accumulation_steps=2,
        warmup_steps=20, 
        learning_rate=1e-5,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10, 
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear", 
        seed=3407,
        save_steps=50, 
        save_total_limit=2,
        output_dir=output_dir_phase2,
        eval_strategy='steps', 
        eval_steps=150, 
        per_device_eval_batch_size=2,
        report_to="none",
    )

    # Inizializzazione Trainer Fase 2
    trainer_p2 = SFTTrainer(
        model=trainer_p1.model,
        tokenizer=tokenizer,
        train_dataset=train_dataset_p2,
        eval_dataset=eval_dataset_p2,
        dataset_text_field="text",
        dataset_num_proc=1,
        max_seq_length=max_seq_length,
        packing=False,
        args=training_args_p2, 
    )

    
    print("Starting 2 Phase Training ...")
    trainer_stats_p2 = trainer_p2.train() 
    print("2 Phase training completed.")

    # --- SALVATAGGIO FINALE (DOPO FASE 2) ---
    print(f"Saving Final GGUF Model to: {gguf_final_model_name}")
    final_model = trainer_p2.model
    final_model.save_pretrained_gguf(gguf_final_model_name, tokenizer, quantization_method="q4_k_m")
