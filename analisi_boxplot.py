import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import numpy as np
import spacy
from scipy.spatial.distance import cosine
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- CONFIGURAZIONE ---
NER_MODEL_DIR = "models/cami_ner_v5_final"
DATASET_PATH = "data/boxplot_test_dataset.csv" 
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# --- CARICAMENTO MODELLI ---

print("Caricamento modelli...")
# Carica il modello linguistico spaCy per la lemmatizzazione
try:
    nlp = spacy.load("it_core_news_lg")
except OSError:
    print("Modello spaCy 'it_core_news_lg' non trovato. Esegui 'python -m spacy download it_core_news_lg'")
    exit()

# Carica il tokenizer e il modello NER addestrato
tokenizer = AutoTokenizer.from_pretrained(NER_MODEL_DIR)
model = AutoModelForTokenClassification.from_pretrained(
    NER_MODEL_DIR,
    output_hidden_states=True
).to(DEVICE)
model.eval()
print(f"Modelli caricati e pronti. Operazioni su dispositivo: {DEVICE}")


# --- FUNZIONI DI SUPPORTO ---

def get_word_vector(hidden_states, tokenized_sentence, sentence_text, target_word_lemma):
    """
    Estrae il vettore di una parola target dalla frase, usando il lemma per la ricerca.
    Questa funzione è una versione adattata di quella presente in `demo_completa.py`.
    """
    doc = nlp(sentence_text)
    
    # Trova gli span di caratteri del target usando il lemma
    target_spans = [(t.idx, t.idx + len(t.text)) for t in doc if t.lemma_.lower() == target_word_lemma.lower()]
    
    if not target_spans:
        return None

    # Prendi solo il primo match se ce ne sono multipli
    target_char_start, target_char_end = target_spans[0]

    # Trova gli indici dei token che corrispondono alla parola
    word_indices = [
        i for i, _ in enumerate(tokenized_sentence.input_ids[0])
        if tokenized_sentence.token_to_chars(i) 
        and tokenized_sentence.token_to_chars(i).start >= target_char_start
        and tokenized_sentence.token_to_chars(i).end <= target_char_end
    ]

    if not word_indices:
        return None

    # Estrai e media i vettori dei token
    return hidden_states[word_indices, :].mean(dim=0).cpu().numpy()


# --- LOGICA PRINCIPALE ---

def main():
    """
    Funzione principale che esegue l'analisi e genera il boxplot.
    """
    # 1. Carica e prepara il dataset
    print(f"Caricamento dataset da: {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH, sep=';')
    # Filtra per avere solo righe complete di argomento e veicolo
    df_filtered = df[df['argomento'].notna() & df['veicolo'].notna()].copy()
    print(f"Trovate {len(df_filtered)} frasi valide per l'analisi.")

    metaphor_distances = []
    literal_distances = []

    # 2. Itera sul dataset per calcolare le distanze
    print("Calcolo delle distanze semantiche...")
    # tqdm fornisce una barra di avanzamento
    for _, row in tqdm(df_filtered.iterrows(), total=df_filtered.shape[0]):
        sentence = row['testo']
        arg_lemma = str(row['argomento']).strip()
        veh_lemma = str(row['veicolo']).strip()
        is_metaphor = row['etichetta'] == 1

        with torch.no_grad():
            # Tokenizza la frase
            inputs = tokenizer(sentence, return_tensors="pt", truncation=True).to(DEVICE)
            # Ottieni gli output del modello
            outputs = model(**inputs)
            # Estrai l'ultimo stato nascosto
            last_hidden_state = outputs.hidden_states[-1].squeeze(0)

            # Ottieni i vettori per argomento e veicolo
            arg_vector = get_word_vector(last_hidden_state, inputs, sentence, arg_lemma)
            veh_vector = get_word_vector(last_hidden_state, inputs, sentence, veh_lemma)

            # Se entrambi i vettori sono stati trovati, calcola la distanza
            if arg_vector is not None and veh_vector is not None:
                distance = cosine(arg_vector, veh_vector)
                if is_metaphor:
                    metaphor_distances.append(distance)
                else:
                    literal_distances.append(distance)

    # 3. Analisi statistica dei risultati
    print("\n--- Statistiche delle Distanze Semantiche ---")
    if metaphor_distances:
        print(f"Metafore ({len(metaphor_distances)} campioni):")
        print(f"  - Distanza Media: {np.mean(metaphor_distances):.4f}")
        print(f"  - Deviazione Standard: {np.std(metaphor_distances):.4f}")
    else:
        print("Nessuna distanza calcolata per le metafore.")

    if literal_distances:
        print(f"Frasi Letterali ({len(literal_distances)} campioni):")
        print(f"  - Distanza Media: {np.mean(literal_distances):.4f}")
        print(f"  - Deviazione Standard: {np.std(literal_distances):.4f}")
    else:
        print("Nessuna distanza calcolata per le frasi letterali.")
        
    # 4. Creazione del Boxplot
    print("\nGenerazione del boxplot...")
    
    # Prepara i dati per il formato richiesto da Seaborn
    plot_data = []
    for dist in metaphor_distances:
        plot_data.append({"Distanza": dist, "Tipo": "Metafora"})
    for dist in literal_distances:
        plot_data.append({"Distanza": dist, "Tipo": "Letterale"})
    
    plot_df = pd.DataFrame(plot_data)

    # Crea e mostra il grafico
    plt.figure(figsize=(10, 7))
    sns.boxplot(x="Tipo", y="Distanza", data=plot_df, palette="viridis")
    plt.title("Confronto Distanza Semantica Coseno tra Frasi Metaforiche e Letterali", fontsize=16)
    plt.xlabel("Tipo di Frase", fontsize=12)
    plt.ylabel("Distanza Coseno", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


if __name__ == "__main__":
    main()