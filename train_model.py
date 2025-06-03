"""
train_model.py

Script per il fine-tuning del modello "Musixmatch/umberto-wikipedia-uncased-v1" su CAMI_dataset_v2.

Funzionalità:
- Carica i CSV train.csv e test.csv prodotti da data_utils.py
- Tokenizza in batch, con padding e truncation
- Imposta TrainingArguments per 3 epoche (batch_size=8, lr=2e-5, salvataggi per epoca)
- Usa Trainer di HuggingFace per l'addestramento su GPU (se disponibile)
- Calcola metriche (accuracy, precision, recall, F1) per epoca
- Salva checkpoint ogni epoca e modello finale in cami_model/
- Include funzioni riusabili per l'import da altri moduli (es. predict.py)
"""

import os
import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import data_utils

def compute_metrics(pred):
    """
    Calcola le metriche di valutazione per la classificazione binaria.
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def tokenize_function(examples, tokenizer):
    """
    Tokenizzazione dei testi con padding e truncation a max_length=128.
    """
    return tokenizer(
        examples['testo'],
        padding='max_length',
        truncation=True,
        max_length=128
    )

def train_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_name: str = "Musixmatch/umberto-wikipedia-uncased-v1",
    output_dir: str = "cami_model",
    num_epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 2e-5
):
    """
    Funzione principale per il fine-tuning del modello.
    
    Args:
        train_df (pd.DataFrame): DataFrame di training con colonne ['testo', 'etichetta', ...].
        test_df (pd.DataFrame): DataFrame di test con colonne ['testo', 'etichetta', ...].
        model_name (str): Identificatore del modello pre-addestrato su HuggingFace.
        output_dir (str): Cartella in cui salvare i checkpoint e il modello finale.
        num_epochs (int): Numero di epoche di training.
        batch_size (int): Dimensione del batch per train/validation.
        learning_rate (float): Learning rate per l'ottimizzazione.
    """
    # Controlla se GPU è disponibile
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device usato per l'addestramento: {device}")
    
    # Carica tokenizer e modello
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)
    
    # Crea HuggingFace Dataset da DataFrame
    train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
    test_ds = Dataset.from_pandas(test_df.reset_index(drop=True))
    
    # Mantieni solo le colonne necessarie
    train_ds = train_ds.map(lambda examples: {'etichetta': examples['etichetta']}, remove_columns=[c for c in train_ds.column_names if c not in ['testo', 'etichetta']])
    test_ds = test_ds.map(lambda examples: {'etichetta': examples['etichetta']}, remove_columns=[c for c in test_ds.column_names if c not in ['testo', 'etichetta']])
    
    # Tokenizzazione in batch
    tokenized_train = train_ds.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    tokenized_test = test_ds.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    
    # Specifica le colonne di input per Trainer
    tokenized_train = tokenized_train.rename_column("etichetta", "labels")
    tokenized_test = tokenized_test.rename_column("etichetta", "labels")
    tokenized_train.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    tokenized_test.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    
    # Definisci gli argomenti di training
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=3
    )
    
    # Instanzia il Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        compute_metrics=compute_metrics
    )
    
    # Avvia il training
    trainer.train()
    
    # Salva il modello fine-tunato
    os.makedirs(output_dir, exist_ok=True)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Modello fine-tunato salvato in '{output_dir}'.")

def main():
    """
    Punto di ingresso se eseguito come script.
    Esegue: load_and_prepare_data -> train_model
    """
    # Ora load_and_prepare_data legge da data/CAMI_dataset_v2.csv
    train_df, test_df = data_utils.load_and_prepare_data(csv_path='data/CAMI_dataset_v2.csv')
    train_model(train_df, test_df)

if __name__ == "__main__":
    # Eventuali installazioni necessarie:
    # !pip install transformers datasets scikit-learn
    main()