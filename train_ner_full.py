# Filename: train_ner_full.py (Versione Finale con Correzione Sub-token)

import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
import numpy as np
import evaluate
import spacy

# --- CONFIGURAZIONE ---
MODEL_NAME = "dbmdz/bert-base-italian-xxl-cased"
DATASET_PATH = "data/metafore_dataset.csv"
# Nuovo nome per il modello finale
OUTPUT_MODEL_DIR = "models/cami_ner_v5_final" 
RANDOM_STATE = 42

print("Caricamento del modello linguistico spaCy 'it_core_news_lg'...")
nlp = spacy.load("it_core_news_lg")
print("Modello spaCy caricato.")

label_list = ["O", "B-ARG", "I-ARG", "B-VEI", "I-VEI"]
label_to_id = {label: i for i, label in enumerate(label_list)}

def load_and_prepare_data(path):
    print(f"Caricamento dataset da: {path}")
    df = pd.read_csv(path, sep=';')
    df_filtered = df[df['argomento'].notna() & df['veicolo'].notna()].copy()
    print(f"Trovate {len(df_filtered)} frasi totali con argomento e veicolo annotati.")
    df_filtered['argomento'] = df_filtered['argomento'].astype(str)
    df_filtered['veicolo'] = df_filtered['veicolo'].astype(str)
    return Dataset.from_pandas(df_filtered)

# --- FUNZIONE DI ETICHETTATURA CORRETTA ---
def tokenize_and_align_labels(example, tokenizer):
    tokenized_inputs = tokenizer(example["testo"], truncation=True, is_split_into_words=False)
    
    # 1. Trovare le posizioni corrette usando spaCy
    text = example["testo"]
    arg_lemma = example["argomento"].lower().strip()
    veh_lemma = example["veicolo"].lower().strip()
    
    doc = nlp(text)
    arg_spans = [(t.idx, t.idx + len(t.text)) for t in doc if t.lemma_.lower() == arg_lemma]
    veh_spans = [(t.idx, t.idx + len(t.text)) for t in doc if t.lemma_.lower() == veh_lemma]

    # 2. Assegnare le etichette con la logica corretta per i sub-token
    word_ids = tokenized_inputs.word_ids()
    previous_word_idx = None
    label_ids = []
    
    for word_idx in word_ids:
        if word_idx is None:
            label_ids.append(-100)
            continue

        token_span = tokenized_inputs.word_to_chars(word_idx)
        token_start, token_end = token_span.start, token_span.end
        
        # Determina se il token corrente è dentro un argomento o un veicolo
        is_in_arg = any(token_start >= start and token_end <= end for start, end in arg_spans)
        is_in_veh = any(token_start >= start and token_end <= end for start, end in veh_spans)
        
        current_label = label_ids[-1] if label_ids else label_to_id["O"]
        
        if word_idx != previous_word_idx: # Inizio di una nuova parola
            if is_in_arg:
                label_ids.append(label_to_id["B-ARG"])
            elif is_in_veh:
                label_ids.append(label_to_id["B-VEI"])
            else:
                label_ids.append(label_to_id["O"])
        else: # Sub-token successivo della stessa parola
            if is_in_arg:
                # Se il token precedente era B-ARG o I-ARG, questo è I-ARG
                if current_label in [label_to_id["B-ARG"], label_to_id["I-ARG"]]:
                    label_ids.append(label_to_id["I-ARG"])
                else: # Altrimenti, è l'inizio di una nuova entità
                    label_ids.append(label_to_id["B-ARG"])
            elif is_in_veh:
                if current_label in [label_to_id["B-VEI"], label_to_id["I-VEI"]]:
                    label_ids.append(label_to_id["I-VEI"])
                else:
                    label_ids.append(label_to_id["B-VEI"])
            else:
                # Se è un sub-token di una parola non-entità, va ignorato
                label_ids.append(-100)

        previous_word_idx = word_idx
        
    tokenized_inputs["labels"] = label_ids
    return tokenized_inputs

def main():
    raw_dataset = load_and_prepare_data(DATASET_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenized_dataset = raw_dataset.map(tokenize_and_align_labels, fn_kwargs={"tokenizer": tokenizer})
    tokenized_dataset = tokenized_dataset.remove_columns(raw_dataset.column_names)
    dataset = DatasetDict(tokenized_dataset.train_test_split(test_size=0.2, seed=RANDOM_STATE))
    
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=len(label_list),
        id2label={i: l for i, l in enumerate(label_list)},
        label2id=label_to_id
    )
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_MODEL_DIR,
        learning_rate=2e-5, # Leggera riduzione per un training più stabile
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1"
    )
    
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    seqeval = evaluate.load("seqeval")
    
    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)
        true_predictions = [[label_list[p] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]
        true_labels = [[label_list[l] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]
        results = seqeval.compute(predictions=true_predictions, references=true_labels)
        return {"precision": results["overall_precision"], "recall": results["overall_recall"], "f1": results["overall_f1"], "accuracy": results["overall_accuracy"]}

    trainer = Trainer(model=model, args=training_args, train_dataset=dataset["train"], eval_dataset=dataset["test"], tokenizer=tokenizer, data_collator=data_collator, compute_metrics=compute_metrics)
    
    print("\n--- Inizio Training del modello NER Finale (con correzione sub-token) ---")
    trainer.train()
    trainer.save_model(OUTPUT_MODEL_DIR)
    
    print(f"\n--- Training completato. Modello salvato in: {OUTPUT_MODEL_DIR} ---")
    
    print("\n--- Valutazione Finale sul Test Set ---")
    eval_results = trainer.evaluate()
    for key, value in eval_results.items():
        print(f"{key.replace('eval_', '').capitalize()}: {value:.4f}")

if __name__ == "__main__":
    main()