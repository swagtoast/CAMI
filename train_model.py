import os
import torch
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import data_utils # Il nostro modulo
import logging

# Impostazioni di logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Commento per Colab: installare le librerie se necessario
# !pip install transformers datasets scikit-learn torch pandas

# Configurazioni
MODEL_NAME = "Musixmatch/umberto-wikipedia-uncased-v1"
DATASET_CSV_PATH = 'data/CAMI_dataset_v2.csv' # Assicurati che il path sia corretto
OUTPUT_DIR = "./cami_model_finetuned" # Directory per salvare il modello e i checkpoint
LOGGING_DIR = "./cami_logs"

# Parametri di training
NUM_EPOCHS = 3
BATCH_SIZE = 8 # Riduci se hai problemi di memoria (es. OOM error)
LEARNING_RATE = 2e-5
MAX_LENGTH = 128 # Lunghezza massima delle sequenze tokenizzate

def preprocess_function(examples, tokenizer):
    """Tokenizza i testi."""
    return tokenizer(examples['testo'], truncation=True, padding='max_length', max_length=MAX_LENGTH)

def compute_metrics(pred):
    """Calcola metriche per la valutazione."""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def train():
    """Funzione principale per il training del modello."""
    logging.info("Avvio del processo di training...")

    # 0. Controlla disponibilità GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Utilizzo del dispositivo: {device}")

    # 1. Caricamento e Preprocessing Dati
    logging.info("Caricamento e preprocessing dei dati...")
    train_df, val_df = data_utils.load_and_preprocess_data(DATASET_CSV_PATH, test_size=0.2)

    if train_df is None or val_df is None:
        logging.error("Errore nel caricamento dei dati. Training interrotto.")
        return

    # Rinomina 'etichetta' in 'label' come atteso da HuggingFace Trainer
    train_df = train_df.rename(columns={'etichetta': 'label'})
    val_df = val_df.rename(columns={'etichetta': 'label'})
    
    # Seleziona solo le colonne necessarie
    train_df = train_df[['testo', 'label']]
    val_df = val_df[['testo', 'label']]

    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    
    logging.info(f"Dataset di training: {len(train_dataset)} campioni.")
    logging.info(f"Dataset di validazione: {len(val_dataset)} campioni.")

    # 2. Caricamento Tokenizer e Modello
    logging.info(f"Caricamento tokenizer e modello pre-addestrato: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2) # 2 labels: metafora (1), non-metafora (0)
    model.to(device) # Sposta il modello sulla GPU se disponibile

    # 3. Tokenizzazione dei Dati
    logging.info("Tokenizzazione dei dataset...")
    train_tokenized_dataset = train_dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)
    val_tokenized_dataset = val_dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)

    # Rimuovi colonne non necessarie per il training e formatta per PyTorch
    train_tokenized_dataset = train_tokenized_dataset.remove_columns(["testo", "__index_level_0__"] if "__index_level_0__" in train_tokenized_dataset.column_names else ["testo"])
    val_tokenized_dataset = val_tokenized_dataset.remove_columns(["testo", "__index_level_0__"] if "__index_level_0__" in val_tokenized_dataset.column_names else ["testo"])
    
    train_tokenized_dataset.set_format("torch")
    val_tokenized_dataset.set_format("torch")

    # 4. Impostazione Argomenti di Training
    logging.info("Configurazione degli argomenti di training...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        evaluation_strategy="epoch",      # Valuta alla fine di ogni epoca
        save_strategy="epoch",            # Salva un checkpoint alla fine di ogni epoca
        load_best_model_at_end=True,      # Carica il miglior modello alla fine del training
        logging_dir=LOGGING_DIR,
        logging_steps=10,                 # Logga ogni 10 step
        report_to="tensorboard",          # Opzionale: per visualizzare log con TensorBoard
        fp16=torch.cuda.is_available(),   # Usa precisione mista se GPU disponibile e supportata
        # no_cuda= (device.type == 'cpu') # Decommenta se vuoi forzare CPU
    )

    # 5. Creazione del Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized_dataset,
        eval_dataset=val_tokenized_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # 6. Training
    logging.info("Avvio del training...")
    try:
        trainer.train()
        logging.info("Training completato.")
    except Exception as e:
        logging.error(f"Errore durante il training: {e}")
        # Potresti voler salvare lo stato attuale qui se possibile
        raise

    # 7. Salvataggio del Modello Fine-tunato
    logging.info(f"Salvataggio del modello fine-tunato in {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR) # Salva anche il tokenizer per coerenza
    logging.info("Modello e tokenizer salvati.")

if __name__ == '__main__':
    # Crea le directory di output se non esistono
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOGGING_DIR, exist_ok=True)
    
    # Per Colab: Assicurati che la GPU sia attiva!
    # Runtime -> Change runtime type -> GPU
    
    train()