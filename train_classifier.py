import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
import numpy as np
import evaluate # <-- IMPORTA LA LIBRERIA

# --- 1. CONFIGURAZIONE (invariata) ---
MODEL_NAME = "Musixmatch/umberto-commoncrawl-cased-v1"
DATASET_PATH = "data/metafore_dataset.csv" 
OUTPUT_MODEL_DIR = "models/cami_classifier_v1"
TEXT_COLUMN = "testo"
LABEL_COLUMN = "etichetta"
TEST_SIZE = 0.2
RANDOM_STATE = 42

def load_and_prepare_dataset(path: str) -> DatasetDict:
    # Questa funzione rimane identica
    print(f"Caricamento del dataset da: {path}")
    df = pd.read_csv(path, sep=';')
    df.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN], inplace=True)
    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(int)
    print(f"Dataset caricato. Numero di esempi totali: {len(df)}")
    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df[LABEL_COLUMN]
    )
    train_dataset = Dataset.from_pandas(train_df, preserve_index=False)
    test_dataset = Dataset.from_pandas(test_df, preserve_index=False)
    return DatasetDict({"train": train_dataset, "test": test_dataset})

# --- NUOVA SEZIONE: FUNZIONE PER CALCOLARE LE METRICHE ---
# Carichiamo in anticipo le metriche che ci interessano
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    """
    Questa funzione viene chiamata dal Trainer alla fine di ogni epoca di valutazione.
    Prende le previsioni del modello e le etichette reali e calcola le metriche.
    """
    logits, labels = eval_pred
    # Le previsioni del modello (logits) sono dei numeri grezzi.
    # Usiamo np.argmax per ottenere la classe predetta (0 o 1).
    predictions = np.argmax(logits, axis=-1)
    
    # Calcoliamo le metriche
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels)
    
    # Restituiamo un dizionario con i risultati
    return {
        "accuracy": accuracy["accuracy"],
        "f1": f1["f1"],
    }
# --------------------------------------------------------

def main():
    raw_datasets = load_and_prepare_dataset(DATASET_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_function(examples):
        return tokenizer(examples[TEXT_COLUMN], padding="max_length", truncation=True, max_length=512)

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.rename_column(LABEL_COLUMN, "labels")
    tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        model.to(device)
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_MODEL_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        logging_dir='./logs',
        load_best_model_at_end=True,
        # --- MODIFICA CHIAVE ---
        # Adesso possiamo dire al modello di scegliere il migliore basandosi sull'F1-score!
        metric_for_best_model="f1",
        use_mps_device=torch.backends.mps.is_available()
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        # --- MODIFICA CHIAVE ---
        # Passiamo la nostra nuova funzione al Trainer
        compute_metrics=compute_metrics,
    )

    print("Inizio del processo di fine-tuning (con calcolo delle metriche)...")
    trainer.train()
    
    print("\n--- Valutazione Finale sul Test Set ---")
    eval_results = trainer.evaluate()
    
    # Stampa i risultati in modo più leggibile
    for key, value in eval_results.items():
        print(f"{key}: {value:.4f}")
    print("---------------------------------------")

if __name__ == "__main__":
    main()