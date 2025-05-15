import torch
import sys

def verifica_installazione_pytorch():
    print("--- Verifica Installazione PyTorch ---")
    try:
        # 1. Stampa la versione di Python usata
        print(f"Versione Python: {sys.version}")

        # 2. Importa torch e stampa la versione di PyTorch
        print(f"Versione PyTorch: {torch.__version__}")

        # 3. Verifica se CUDA è disponibile
        cuda_disponibile = torch.cuda.is_available()
        print(f"CUDA disponibile per PyTorch: {cuda_disponibile}")

        if cuda_disponibile:
            print(f"  Numero di GPU disponibili: {torch.cuda.device_count()}")
            # Itera su tutte le GPU disponibili (di solito solo una in ambienti desktop/WSL)
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            
            # Stampa la versione di CUDA con cui PyTorch è stato compilato.
            # Questa è la versione del CUDA toolkit che PyTorch usa, non la versione del tuo driver.
            print(f"  PyTorch compilato con CUDA versione: {torch.version.cuda}")
            
            # Stampa la versione di cuDNN (se disponibile e PyTorch la espone)
            if torch.backends.cudnn.is_available():
                print(f"  Versione cuDNN: {torch.backends.cudnn.version()}")
            else:
                print("  cuDNN non disponibile o non rilevato da PyTorch.")

            # Test rapido: crea un tensore sulla GPU
            print("\n  Test: Creazione di un tensore sulla GPU...")
            try:
                tensor_esempio = torch.tensor([1.0, 2.0, 3.0], device='cuda')
                print(f"  Tensore creato sulla GPU: {tensor_esempio}")
                print(f"  Dispositivo del tensore: {tensor_esempio.device}")
                print("  Test GPU superato!")
            except Exception as e:
                print(f"  Errore durante il test della GPU: {e}")
        else:
            print("\nCUDA non è disponibile per PyTorch.")
            print("Se hai una GPU NVIDIA e ti aspettavi il supporto CUDA, considera di verificare:")
            print("  1. L'installazione del driver NVIDIA sul tuo sistema host (Windows) sia corretta e aggiornata.")
            print("  2. Che PyTorch sia stato installato con il pacchetto CUDA corretto")
            print("     (es. `pytorch-cuda=X.X` tramite Conda o la build corretta con `pip`).")
            print("  3. La configurazione di WSL2 per l'accesso alla GPU sia attiva.")

    except ImportError:
        print("\nERRORE: PyTorch non sembra essere installato correttamente (impossibile importare 'torch').")
        print("Assicurati di aver attivato l'ambiente Conda corretto.")
    except Exception as e:
        print(f"\nSi è verificato un errore imprevisto: {e}")
    finally:
        print("--- Fine Verifica ---")

if __name__ == "__main__":
    verifica_installazione_pytorch()