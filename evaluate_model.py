import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt # Per visualizzare la confusion matrix
import pandas as pd
import data_utils # Il nostro modulo
import logging
import numpy as np
from tqdm import tqdm

# Impostazioni di logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configurazioni
MODEL_PATH = "./cami_model_finetuned"  # Path al modello salvato
DATASET_CSV_PATH = 'data/CAMI_dataset_v2.csv' # Path al dataset originale per ottenere il test set
MAX_LENGTH = 128 # Deve essere la stessa usata durante il training
BATCH_SIZE_EVAL = 16 # Batch size per la valutazione

def preprocess_function_eval(examples, tokenizer):
    """Tokenizza i testi per la valutazione."""
    return tokenizer(examples['testo'], truncation=True, padding='max_length', max_length=MAX_LENGTH, return_tensors="pt")

def evaluate():
    """Funzione principale per la valutazione del modello."""
    logging.info("Avvio del processo di valutazione...")

    # 0. Controlla disponibilità GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Utilizzo del dispositivo: {device}")

    # 1. Caricamento Tokenizer e Modello Fine-tunato
    logging.info(f"Caricamento tokenizer e modello da {MODEL_PATH}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        model.to(device) # Sposta il modello sulla GPU/CPU
        model.eval()     # Imposta il modello in modalità valutazione
    except Exception as e:
        logging.error(f"Errore nel caricamento del modello o tokenizer: {e}")
        return

    # 2. Caricamento e Preparazione del Test Set
    logging.info("Caricamento e preparazione del test set...")
    # Carichiamo l'intero dataset e usiamo la stessa suddivisione del training
    # per ottenere il test_df corretto.
    # NOTA: Se hai già salvato train_df e test_df separatamente, caricali direttamente.
    # Qui ri-eseguiamo la suddivisione per coerenza con lo script di training.
    _, test_df = data_utils.load_and_preprocess_data(DATASET_CSV_PATH, test_size=0.2) # Stessa random_state di train_model.py

    if test_df is None or test_df.empty:
        logging.error("Errore nel caricamento del test set o test set vuoto. Valutazione interrotta.")
        return
    
    test_df = test_df.rename(columns={'etichetta': 'label'})
    test_df = test_df[['testo', 'label']] # Assicurati che ci sia la colonna 'label'
    
    test_texts = test_df['testo'].tolist()
    true_labels = test_df['label'].tolist()
    
    logging.info(f"Test set caricato: {len(test_texts)} campioni.")

    # 3. Predizioni sul Test Set
    logging.info("Effettuando predizioni sul test set...")
    all_predictions = []
    
    with torch.no_grad(): # Disabilita il calcolo dei gradienti
        for i in tqdm(range(0, len(test_texts), BATCH_SIZE_EVAL), desc="Evaluating"):
            batch_texts = test_texts[i:i+BATCH_SIZE_EVAL]
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH)
            inputs = {k: v.to(device) for k, v in inputs.items()} # Sposta i dati sul device
            
            outputs = model(**inputs)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            all_predictions.extend(predictions.cpu().numpy()) # Riporta le predizioni sulla CPU e converte in numpy

    # 4. Calcolo e Stampa delle Metriche
    logging.info("\n--- Risultati della Valutazione ---")
    
    # Classification Report
    report = classification_report(true_labels, all_predictions, target_names=['Non-Metafora (0)', 'Metafora (1)'])
    print("\nClassification Report:\n", report)

    # Confusion Matrix
    cm = confusion_matrix(true_labels, all_predictions)
    print("\nConfusion Matrix:\n", cm)

    # Visualizzazione della Confusion Matrix (opzionale, ma utile)
    try:
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-Metafora', 'Metafora'])
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.savefig("confusion_matrix.png") # Salva l'immagine
        logging.info("Confusion matrix salvata come confusion_matrix.png")
        # In Colab, plt.show() la mostrerebbe direttamente
        # plt.show() 
    except Exception as e:
        logging.warning(f"Impossibile generare l'immagine della confusion matrix: {e}")
        logging.warning("Assicurati che matplotlib sia installato e funzioni correttamente.")

if __name__ == '__main__':
    evaluate()