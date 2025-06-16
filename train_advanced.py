# Filename: train_advanced.py

import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
import numpy as np
import evaluate

# ==============================================================================
# --- 1. IPERPARAMETRI E CONFIGURAZIONE ---
# ==============================================================================
HP = {
    "model_name": "Musixmatch/umberto-commoncrawl-cased-v1",
    "output_dir": "models/cami_classifier_tuned",
    "learning_rate": 2e-5,
    "num_epochs": 4,
    "batch_size": 16,
    "weight_decay": 0.01
}

# --- Costanti del Progetto ---
DATASET_PATH = "data/metafore_dataset.csv" 
TEXT_COLUMN = "testo"
LABEL_COLUMN = "etichetta"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# --- MODIFICA 1: Definizione esplicita delle etichette ---
ID2LABEL = {0: "Letterale", 1: "Metafora"}
LABEL2ID = {"Letterale": 0, "Metafora": 1}

def load_and_prepare_dataset(path: str) -> DatasetDict:
    print(f"Caricamento del dataset da: {path}")
    df = pd.read_csv(path, sep=';')
    df.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN], inplace=True)
    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(int)
    
    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df[LABEL_COLUMN]
    )
    train_dataset = Dataset.from_pandas(train_df, preserve_index=False)
    test_dataset = Dataset.from_pandas(test_df, preserve_index=False)
    return DatasetDict({"train": train_dataset, "test": test_dataset})

accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels, pos_label=1) # Specifichiamo la classe positiva per F1
    return {"accuracy": accuracy["accuracy"], "f1": f1["f1"]}

def main():
    raw_datasets = load_and_prepare_dataset(DATASET_PATH)
    tokenizer = AutoTokenizer.from_pretrained(HP["model_name"])

    def tokenize_function(examples):
        return tokenizer(examples[TEXT_COLUMN], padding="max_length", truncation=True, max_length=512)

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.rename_column(LABEL_COLUMN, "labels")
    tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    
    # --- MODIFICA 2: Passiamo le etichette al modello durante l'inizializzazione ---
    model = AutoModelForSequenceClassification.from_pretrained(
        HP["model_name"], 
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )
    
    device = "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    model.to(device)
    
    print("\n--- Iperparametri di Training ---")
    for key, value in HP.items():
        print(f"{key}: {value}")
    print("----------------------------------\n")

    training_args = TrainingArguments(
        output_dir=HP["output_dir"],
        learning_rate=HP["learning_rate"],
        num_train_epochs=HP["num_epochs"],
        per_device_train_batch_size=HP["batch_size"],
        per_device_eval_batch_size=HP["batch_size"],
        weight_decay=HP["weight_decay"],
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir='./logs_tuned',
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        use_mps_device=torch.backends.mps.is_available()
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        compute_metrics=compute_metrics,
        tokenizer=tokenizer
    )
    
    print("--- Inizio Training del Classificatore (Versione Corretta) ---")
    trainer.train()
    
    print(f"\nTraining completato. Salvataggio del modello migliore in: {HP['output_dir']}")
    trainer.save_model(HP['output_dir'])
    
    print("\n--- Valutazione Finale sul Modello Migliore ---")
    eval_results = trainer.evaluate()
    
    for key, value in eval_results.items():
        print(f"{key.replace('eval_', '').capitalize()}: {value:.4f}")
    print("---------------------------------------------")

if __name__ == "__main__":
    main()