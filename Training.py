from unsloth import FastLanguageModel, is_bfloat16_supported,unsloth_train
from unsloth.chat_templates import get_chat_template
from datasets import load_from_disk,load_dataset,DatasetDict,interleave_datasets
from trl import SFTTrainer
from transformers import TrainingArguments,TrainerState,TrainerControl,TrainerCallback
from unsloth import is_bfloat16_supported
import torch
import os
import random
os.environ["TORCHINDUCTOR_CACHE_DIR"] = "C:\\Users\\rossi\\torchinductor_cache"



#Funzione che assembla i prompts usati per l'addestramento del modello
def formatting_prompts_func(examples):
    database_schema = examples["context"]
    inputs       = examples["question"]
    outputs      = examples["answer"]
    texts = []
    for schema, input, output in zip(database_schema, inputs, outputs):
        text = alpaca_prompt.format(schema, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }
pass
   
if __name__ == '__main__':
    
    #Parametri Modello
    max_seq_length=2048
    dtype=None
    load_in_4bit=True

    #Caricamento Dataset
    try:
        #dataset = load_from_disk("t-sql_dataset_corrected")
        dataset=load_dataset("json",data_files={"train":"t-sql_dataset_corrected/train.json"})
        print("Loaded Saved Dataset")
    except:
        '''
        dataset_tts_train = load_dataset("Clinton/Text-to-sql-v1", split='train[:100%]')
        dataset_tts_train = dataset_tts_train.remove_columns(['source', 'text'])
        dataset_tts_train = dataset_tts_train.rename_columns({'instruction': 'question', 'input': 'context', 'response': 'answer'})
        
        dataset = DatasetDict({ 'train': interleave_datasets([dataset_tts_train]) })
        dataset.save_to_disk("merged_dataset_optimized")'
        '''
        print("Error loading saved dataset, downloaded loaded new one")
        
    #Caricamento Modello e Tokenizer
    model,tokenizer=FastLanguageModel.from_pretrained(
        model_name="unsloth/codellama-7b-bnb-4bit",
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit
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
    
    #Template prompt su cui si assembla i prompt finali
    alpaca_prompt="""
    Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
    ### Database_schema:
    {}
    ### Input:
    {}
    ### Response:
    {}
    """
    EOS_TOKEN=tokenizer.eos_token

    #Creo lo split test a partire dal dataset completo per usarlo poi in inferenza
    split_dataset=dataset["train"].train_test_split(test_size=0.1,seed=3407)
    train_dataset=split_dataset["train"].map(formatting_prompts_func,batched=True,num_proc=1)
    test_dataset=split_dataset["test"].map(formatting_prompts_func,batched=True,num_proc=1)
    print(train_dataset[0])
    
    #Inizializzazione Trainier con parametri custum
    trainer=SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        dataset_num_proc=4,
        max_seq_length=max_seq_length,
        packing=False,
        args=TrainingArguments(
            num_train_epochs=4,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=300,
            #max_steps=37655,
            learning_rate=5e-5,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=50,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=3407,
            save_strategy="steps",      
            save_steps=50,                
            save_total_limit=3,
            output_dir="./output",
            report_to="none",
            
        ),
        
        )

    #Avvio Training con checkpoint e poi salvataggio modello alla fine
    trainer_stats=trainer.train()
    model.save_pretrained_gguf("ggufmodel_llama_3.1_v3",tokenizer,quantization_method="q4_k_m")