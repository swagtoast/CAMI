"""
predict.py

Script per l'inferenza su nuovi testi in italiano:
- Legge tutti i file .txt da una cartella (es. data/nuovi_testi/)
- Divide i testi in frasi (nltk.sent_tokenize o spaCy)
- Per ogni frase:
    - Tokenizza e predice la probabilità di metafora con il modello fine-tunato
    - Se prob > 0.65: etichetta 'metafora'
      Se prob < 0.35: etichetta 'non metafora'
      Se 0.35 <= prob <= 0.65: richiede annotazione manuale (input())
    - Salva annotazioni manuali in manual_annotations.csv
    - Quando si raggiungono 75 annotazioni manuali, si autopettra addestramento aggiungendo i nuovi esempi
- Per le frasi etichettate come metafora:
    - Estrae argomento e veicolo come la coppia di sostantivi con similarità semantica minima
    - Calcola distanza semantica (usando semantics.distanza_semantica)
    - Estrae embedding di argomento e veicolo (modello embedding separato) e li riduce a 2D (PCA)
    - Visualizza scatterplot dei vettori 2D
    - Visualizza heatmap dell'attenzione (ultima layer, media over heads) sui token
- Calcola indice di figuralità per ciascun file:
    indice = (num_metafore / 1000) * (20 / avg_num_parole_frase)
- Salva risultati finali in risultati_inferenza.csv con colonne:
    file_name, indice_figuralita, conteggio_frasi, conteggio_metafore
"""

import os
import csv
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
import nltk
from semantics import distanza_semantica, stima_concretezza
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import warnings

# In Colab, assicurarsi di eseguire:
# !pip install transformers sklearn nltk spacy
# !python -m spacy download it_core_news_sm

# Import spaCy per estrazione dei sostantivi
import spacy
nlp = spacy.load("it_core_news_sm")

# Download risorse NLTK per sentence tokenizer
nltk.download('punkt')

def load_model(model_dir='cami_model'):
    """
    Carica il tokenizer, il classificatore e il modello base per embedding.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    classifier = AutoModelForSequenceClassification.from_pretrained(model_dir, output_attentions=True)
    classifier.to(device)
    classifier.eval()
    # Modello separate per embedding (solo base, senza testa di classificazione)
    embed_model = AutoModel.from_pretrained("Musixmatch/umberto-wikipedia-uncased-v1")
    embed_model.to(device)
    embed_model.eval()
    return tokenizer, classifier, embed_model, device

def get_sentence_embedding(embed_model, tokenizer, sentence, device):
    """
    Restituisce embedding medio (mean pooling) di un'intera frase.
    """
    inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = embed_model(**inputs)
        last_hidden = outputs.last_hidden_state  # [1, seq_len, hidden_size]
        mask = inputs['attention_mask'].unsqueeze(-1)  # [1, seq_len, 1]
        masked_hidden = last_hidden * mask  # maschera i padding
        sum_hidden = masked_hidden.sum(dim=1)  # [1, hidden_size]
        lengths = mask.sum(dim=1)  # [1, 1]
        return (sum_hidden / lengths).cpu().numpy().flatten()  # vettore 1D di dimensione hidden_size

def get_word_embedding(embed_model, tokenizer, word, device):
    """
    Restituisce embedding medio (mean pooling) di una singola parola (fuori contesto).
    """
    inputs = tokenizer(word, return_tensors='pt', padding=True, truncation=True, max_length=32).to(device)
    with torch.no_grad():
        outputs = embed_model(**inputs)
        last_hidden = outputs.last_hidden_state  # [1, seq_len, hidden_size]
        mask = inputs['attention_mask'].unsqueeze(-1)
        masked_hidden = last_hidden * mask
        sum_hidden = masked_hidden.sum(dim=1)
        lengths = mask.sum(dim=1)
        return (sum_hidden / lengths).cpu().numpy().flatten()

def extract_noun_pairs(sentence):
    """
    Estrae dalla frase tutti i possibili sostantivi (lemma) usando spaCy.
    Restituisce la lista dei lemma dei sostantivi.
    """
    doc = nlp(sentence)
    nouns = [token.lemma_ for token in doc if token.pos_ == 'NOUN']
    return nouns

def select_arg_veh(nouns):
    """
    Dato l'elenco di sostantivi, calcola per ogni coppia la similarità semantica
    e restituisce la coppia (argomento, veicolo) con similarità minima.
    Se non ci sono almeno 2 sostantivi, restituisce (None, None).
    """
    if len(nouns) < 2:
        return None, None
    best_pair = (None, None)
    min_sim = 1.0  # più bassa è la similarità, più probabile che sia metafora
    for i in range(len(nouns)):
        for j in range(i + 1, len(nouns)):
            w1, w2 = nouns[i], nouns[j]
            sim = distanza_semantica(w1, w2)
            if sim is None:
                sim = 0.0
            if sim < min_sim:
                min_sim = sim
                best_pair = (w1, w2)
    return best_pair

def plot_attention_heatmap(tokens, attention_matrix):
    """
    Visualizza una heatmap dell'attenzione (seq_len x seq_len) per una data sequenza di token.
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(attention_matrix, interpolation='nearest', cmap='viridis')
    plt.xticks(range(len(tokens)), tokens, rotation=90, fontsize=6)
    plt.yticks(range(len(tokens)), tokens, fontsize=6)
    plt.colorbar()
    plt.title("Heatmap dell'attenzione (ultimo layer, media over heads)")
    plt.tight_layout()
    plt.show()

def infer_on_folder(
    input_folder='data/nuovi_testi/',
    results_csv='risultati_inferenza.csv',
    manual_csv='manual_annotations.csv',
    retrain_threshold=75
):
    """
    Esegue l'inferenza su tutti i file .txt nella cartella input_folder.
    Salva i risultati aggregati in results_csv e le annotazioni manuali in manual_csv.
    Quando manual_csv raggiunge retrain_threshold righe, lancia il ri-addestramento.
    """
    # Carica modelli e tokenizer
    tokenizer, classifier, embed_model, device = load_model()
    
    # Se non esistono, crea i file CSV vuoti con header
    if not os.path.exists(results_csv):
        with open(results_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['file_name', 'indice_figuralita', 'conteggio_frasi', 'conteggio_metafore'])
    if not os.path.exists(manual_csv):
        with open(manual_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['testo', 'etichetta', 'probabilita'])
    
    # Itera sui file .txt
    for filename in os.listdir(input_folder):
        if not filename.lower().endswith('.txt'):
            continue
        filepath = os.path.join(input_folder, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        
        if not text:
            # Se il testo è vuoto, indice di figuralità = 0
            with open(results_csv, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([filename, 0.0, 0, 0])
            continue
        
        # Suddividi in frasi (NLTK)
        sentences = nltk.sent_tokenize(text, language='italian')
        num_sentences = len(sentences)
        num_metaphors = 0
        total_word_count = 0
        
        for sentence in sentences:
            total_word_count += len(sentence.split())
            # Tokenizza e predici
            inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
            with torch.no_grad():
                outputs = classifier(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                prob_meta = probs[1]  # probabilità di "metafora"
            
            if prob_meta > 0.65:
                label = 1  # Metafora
            elif prob_meta < 0.35:
                label = 0  # Non metafora
            else:
                # Caso di incertezza: richiede annotazione manuale
                print(f"\nFrase incerta (probabilità={prob_meta:.2f}):")
                print(f"\"{sentence}\"")
                ans = input("La frase è metaforica? (y/n): ").strip().lower()
                if ans == 'y':
                    label = 1
                else:
                    label = 0
                # Salva annotazione manuale
                with open(manual_csv, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([sentence, label, f"{prob_meta:.4f}"])
                
                # Controlla se raggiunto threshold per retraining
                current_manual = pd.read_csv(manual_csv, encoding='utf-8')
                if len(current_manual) >= retrain_threshold:
                    print("\n=== Soglia di annotazioni manuali raggiunta: avvio ri-addestramento ===")
                    retrain_model(manual_csv)
            
            if label == 1:
                num_metaphors += 1
                # Estrai argomento e veicolo
                nouns = extract_noun_pairs(sentence)
                arg, veh = select_arg_veh(nouns)
                if arg and veh:
                    sim = distanza_semantica(arg, veh)
                    concretezza_arg = stima_concretezza(arg)
                    concretezza_veh = stima_concretezza(veh)
                    print(f"--> Metafora rilevata: \"{sentence}\"")
                    print(f"    Argomento: {arg}, Veicolo: {veh}, Similarità sem.: {sim}")
                    print(f"    Concretezza: {arg}:{concretezza_arg}, {veh}:{concretezza_veh}")
                    
                    # Calcola embedding dei due termini (fuori contesto)
                    vec_arg = get_word_embedding(embed_model, tokenizer, arg, device)
                    vec_veh = get_word_embedding(embed_model, tokenizer, veh, device)
                    # Riduci dimensione a 2D con PCA
                    pca = PCA(n_components=2)
                    pts_2d = pca.fit_transform([vec_arg, vec_veh])
                    # Scatter plot
                    plt.figure(figsize=(4, 4))
                    plt.scatter(pts_2d[0, 0], pts_2d[0, 1], label=f"Arg: {arg}")
                    plt.scatter(pts_2d[1, 0], pts_2d[1, 1], label=f"Veh: {veh}")
                    plt.legend()
                    plt.title("Rappresentazione 2D Argomento/Veicolo")
                    plt.tight_layout()
                    plt.show()
                    
                    # Matrice di attenzione (ultimo layer, media over heads)
                    inputs_att = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
                    with torch.no_grad():
                        out_att = classifier(**inputs_att, output_attentions=True)
                        attentions = out_att.attentions  # tuple di length=num_layers
                        last_layer = attentions[-1][0]  # [num_heads, seq_len, seq_len]
                        avg_att = last_layer.mean(dim=0).cpu().numpy()  # [seq_len, seq_len]
                    tokens = tokenizer.convert_ids_to_tokens(inputs_att['input_ids'][0])
                    plot_attention_heatmap(tokens, avg_att)
                else:
                    print(f"--> Metafora rilevata ma non sono stati estratti argomento/veicolo (frase: \"{sentence}\")")
        
        # Calcola indice di figuralità: (num_metaphors / 1000) * (20 / avg_num_parole_frase)
        avg_words = total_word_count / num_sentences if num_sentences > 0 else 0
        if avg_words > 0:
            indice_fig = (num_metaphors / 1000) * (20 / avg_words)
        else:
            indice_fig = 0.0
        
        # Salva i risultati per il file
        with open(results_csv, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([filename, f"{indice_fig:.4f}", num_sentences, num_metaphors])
        
        print(f"\nFile: {filename}")
        print(f"  Totale frasi: {num_sentences}, Metafore: {num_metaphors}, Indice: {indice_fig:.4f}\n")
    print("=== Inferenza completata ===")

def retrain_model(manual_csv, model_dir='cami_model', epochs=1):
    """
    Esegue il ri-addestramento automatico aggiungendo le annotazioni manuali
    al training set originale e salva il nuovo modello in cami_model/.
    """
    import data_utils
    import train_model
    # Leggi train.csv esistente
    train_original = pd.read_csv('train.csv', encoding='utf-8')
    # Leggi test.csv per mantenere invariato il test set
    test_df = pd.read_csv('test.csv', encoding='utf-8')
    # Leggi annotazioni manuali
    manual_df = pd.read_csv(manual_csv, encoding='utf-8')
    # Prepara DataFrame manual con colonne minime: testo, etichetta
    manual_df = manual_df[['testo', 'etichetta']].copy()
    manual_df['argomento'] = None
    manual_df['veicolo'] = None
    # Unisci con il training originale
    train_combined = pd.concat([train_original, manual_df], ignore_index=True)
    # Chiamata a train_model con numero di epoche ridotto
    print("Avvio ri-addestramento con dataset esteso...")
    train_model.train_model(
        train_df=train_combined,
        test_df=test_df,
        output_dir=model_dir,
        num_epochs=epochs
    )
    print("Ri-addestramento completato. Reset manual_annotations.csv a zero.")
    # Svuota file annotazioni manuali
    open(manual_csv, 'w', encoding='utf-8').close()

if __name__ == "__main__":
    # Esempio di utilizzo: infer_on_folder('data/nuovi_testi/')
    infer_on_folder()