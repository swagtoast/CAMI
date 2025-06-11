import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
import numpy as np
import evaluate

MODEL_NAME = "dbmdz/bert-base-italian-xxl-cased"
DATASET_PATH = "data/metafore_dataset.csv"
OUTPUT_MODEL_DIR = "models/cami_ner_v2"
RANDOM_STATE = 42

label_list = ["O", "B-ARG", "I-ARG", "B-VEI", "I-VEI"]
label_to_id = {label: i for i, label in enumerate(label_list)}

def load_and_prepare_data(path):
    print(f"Caricamento dataset da: {path}")
    df = pd.read_csv(path, sep=';')
    df = df[(df['etichetta'] == 1) & df['argomento'].notna() & df['veicolo'].notna()].copy()
    print(f"Trovate {len(df)} frasi metaforiche con argomento e veicolo annotati.")
    df['argomento'] = df['argomento'].astype(str)
    df['veicolo'] = df['veicolo'].astype(str)
    return Dataset.from_pandas(df)

def tokenize_and_align_labels(example, tokenizer):
    tokenized_inputs = tokenizer(example["testo"], truncation=True, is_split_into_words=False)
    word_ids = tokenized_inputs.word_ids()
    previous_word_idx = None
    label_ids = []
    text = example["testo"]
    arg_text = example["argomento"]
    veh_text = example["veicolo"]
    arg_start = text.find(arg_text)
    arg_end = arg_start + len(arg_text) if arg_start != -1 else -1
    veh_start = text.find(veh_text)
    veh_end = veh_start + len(veh_text) if veh_start != -1 else -1
    for word_idx in word_ids:
        if word_idx is None:
            label_ids.append(-100)
        elif word_idx != previous_word_idx:
            char_span = tokenized_inputs.word_to_chars(word_idx)
            token_start, token_end = char_span.start, char_span.end
            label = "O"
            if arg_start != -1 and token_start >= arg_start and token_end <= arg_end:
                label = "B-ARG" if token_start == arg_start else "I-ARG"
            elif veh_start != -1 and token_start >= veh_start and token_end <= veh_end:
                label = "B-VEI" if token_start == veh_start else "I-VEI"
            label_ids.append(label_to_id[label])
        else:
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
        learning_rate=3e-5,
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
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        results = seqeval.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    
    trainer.train()
    trainer.save_model(OUTPUT_MODEL_DIR)
    
    print("\n--- Valutazione Finale sul Test Set ---")
    eval_results = trainer.evaluate()
    for key, value in eval_results.items():
        print(f"{key.replace('eval_', '').capitalize()}: {value:.4f}")

if __name__ == "__main__":
    main()