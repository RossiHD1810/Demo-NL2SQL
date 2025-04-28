import json
from collections import Counter
import os

# --- Modifica questi percorsi ---
file_path = '2_dataset.json'
output_file_path_default = '2_dataset_deduplicato.json'
# -----------------------------


if not os.path.exists(file_path):
    print(f"Errore: Il file '{file_path}' non è stato trovato.")
else:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            
            original_data = json.load(f)

        # Verifica se i dati caricati sono una lista
        if not isinstance(original_data, list):
            print(f"Errore: Il contenuto del file JSON '{file_path}' non è una lista come previsto.")
        else:
            print(f"--- Analisi del file originale: '{file_path}' ---")
            # 1. Conteggio totale dei campioni originali
            total_samples_original = len(original_data)
            print(f"Conteggio totale originale: {total_samples_original} campioni.")

            # 2. Conteggio unici e duplicati nel file originale
            stringified_samples_original = []
            try:
                stringified_samples_original = [json.dumps(sample, sort_keys=True) for sample in original_data]
                sample_frequency = Counter(stringified_samples_original)
                unique_samples_count_original = len(sample_frequency)
                total_duplicates_original = total_samples_original - unique_samples_count_original

                print(f"Conteggio unici originali: {unique_samples_count_original} campioni unici.")
                print(f"Numero di duplicati originali: {total_duplicates_original} occorrenze duplicate.")
            except TypeError as e:
                print(f"Errore durante l'analisi dei duplicati: {e}")
                original_data = None 

            if original_data: 
                print("\n--- Creazione della lista deduplicata ---")
                
                unique_samples_list = []
                seen_samples_hashes = set() 
                for sample in original_data:
                    try:
                        
                        sample_hash = json.dumps(sample, sort_keys=True)

                        
                        if sample_hash not in seen_samples_hashes:
                            seen_samples_hashes.add(sample_hash)
                            unique_samples_list.append(sample)
                    except TypeError as e:
                         print(f"Attenzione: Saltato un elemento non processabile durante la deduplicazione: {sample} - Errore: {e}")
                         continue 

                # 4. Stampa del risultato della deduplicazione
                print(f"Lista deduplicata creata con successo.")
                print(f"Numero di campioni nella lista deduplicata: {len(unique_samples_list)}")
                if len(unique_samples_list) == unique_samples_count_original:
                    print("(Questo numero corrisponde correttamente al conteggio dei campioni unici originali).")
                else:
                     print("(ATTENZIONE: Il numero di elementi nella lista deduplicata non corrisponde al conteggio iniziale degli unici. Potrebbero esserci stati errori nel processo.)")


                # 5. Opzione per salvare la nuova lista
                print("\n--- Salvataggio (Opzionale) ---")
                save_choice = input(f"Vuoi salvare la lista deduplicata in un nuovo file JSON? (s/N): ").strip().lower()

                if save_choice == 's':
                    output_filename = input(f"Inserisci il nome del file di output (invio per '{output_file_path_default}'): ").strip()
                    if not output_filename:
                        output_filename = output_file_path_default

                    try:
                        with open(output_filename, 'w', encoding='utf-8') as f_out:
                            json.dump(unique_samples_list, f_out, ensure_ascii=False, indent=4)
                        print(f"Lista deduplicata salvata con successo in '{output_filename}'.")
                    except Exception as e:
                        print(f"Errore durante il salvataggio del file '{output_filename}': {e}")
                else:
                    print("La lista deduplicata non è stata salvata.")

    except json.JSONDecodeError:
        print(f"Errore: Il file '{file_path}' non contiene JSON valido o è corrotto.")
    except FileNotFoundError: 
         print(f"Errore: Il file '{file_path}' non è stato trovato o non può essere letto.")
    except Exception as e:
        print(f"Si è verificato un errore imprevisto durante l'elaborazione: {e}")