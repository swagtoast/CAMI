"""
evaluate_model.py

Script per la valutazione del modello fine-tunato su CAMI_dataset_v2.

Funzionalità:
- Carica il modello salvato in cami_model/
- Carica test.csv prodotto da data_utils.py
- Tokenizza e ottiene predizioni
- Calcola e stampa accuracy, precision, recall, F1 con sklearn.classification_report
- Opzionale: stampa e visualizza confusion matrix con matplotlib
"""

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns  # Se si preferisce una heatmap più carina; altrimenti si può usare solo matplotlib

def evaluate(model_dir='cami_model', test_csv='test.csv'):
    """
    Esegue la valutazione del modello salvato su test set.
    """
    # Controlla dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device per la valutazione: {device}")
    
    # Carica test set
    df_test = pd.read_csv(test_csv, encoding='utf-8')
    df_test = df_test[df_test['testo'].notna() & df_test['testo'].str.strip().astype(bool)]
    
    # Carica tokenizer e modello
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    
    # Tokenizza il test set
    encodings = tokenizer(
        df_test['testo'].tolist(),
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )
    input_ids = encodings['input_ids'].to(device)
    attention_mask = encodings['attention_mask'].to(device)
    
    # Disabilita gradienti
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1).cpu().numpy()
    
    # Etichette reali
    true_labels = df_test['etichetta'].values
    
    # Stampa report
    report = classification_report(true_labels, preds, target_names=['Non Metafora', 'Metafora'])
    print("=== Classification Report ===")
    print(report)
    
    # Matrice di confusione
    cm = confusion_matrix(true_labels, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Pred: Non', 'Pred: Meta'],
                yticklabels=['True: Non', 'True: Meta'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Eventuali installazioni necessarie:
    # !pip install transformers scikit-learn seaborn matplotlib
    evaluate()