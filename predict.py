import os
import glob
import nltk # Per tokenizzazione in frasi
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from tqdm import tqdm
import logging
# import semantics # Opzionale: per analisi semantica delle metafore trovate, se argomento/veicolo sono disponibili

# Impostazioni di logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configurazioni
MODEL_PATH = "./cami_model_finetuned"  # Path al modello salvato
NEW_TEXTS_DIR = "data/nuovi_testi/"    # Directory con i file .txt da analizzare
OUTPUT_CSV_PATH = "figurality_results.csv" # File CSV per salvare i risultati
MAX_LENGTH = 128 # Stessa usata nel training
BATCH_SIZE_INFERENCE = 16 # Batch size per l'inferenza

# Download risorse NLTK (da eseguire una volta se non già fatto)
# In Colab:
# import nltk
# nltk.download('punkt')
# nltk.download('wordnet') # Se si usa semantics.py
# nltk.download('omw-1.4') # Se si usa semantics.py

def split_into_sentences(text_content: str) -> list[str]:
    """Suddivide il testo in frasi usando NLTK."""
    try:
        # Assicurati che nltk.download('punkt') sia stato eseguito
        sentences = nltk.sent_tokenize(text_content, language='italian')
        return [s.strip() for s in sentences if s.strip()] # Rimuovi frasi vuote
    except Exception as e:
        logging.error(f"Errore nella tokenizzazione delle frasi (hai scaricato 'punkt'?): {e}")
        return []

def predict_metaphors_in_batch(sentences: list[str], tokenizer, model, device) -> list[int]:
    """Predice etichette (0 o 1) per una lista di frasi in batch."""
    if not sentences:
        return []
    
    all_predictions = []
    model.eval() # Modalità valutazione
    with torch.no_grad():
        for i in range(0, len(sentences), BATCH_SIZE_INFERENCE):
            batch_sentences = sentences[i:i+BATCH_SIZE_INFERENCE]
            inputs = tokenizer(
                batch_sentences, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=MAX_LENGTH
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = model(**inputs)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            all_predictions.extend(predictions.cpu().numpy().tolist())
    return all_predictions

def calculate_figurality_index(num_metaphors: int, num_sentences: int, avg_words_per_sentence: float) -> float:
    """
    Calcola l'indice di figuralità.
    Formula: (metafore / 1000 frasi) * (20 / numero_medio_di_parole_in_una_frase_del_testo_in_esame)
    """
    if num_sentences == 0 or avg_words_per_sentence == 0:
        return 0.0
    
    metaphors_per_1000_sentences = (num_metaphors / num_sentences) * 1000
    
    # Il fattore 20 è un termine di normalizzazione, come specificato nella richiesta.
    # Potrebbe essere basato su una lunghezza media di frase di riferimento (es. 20 parole).
    length_factor = 20 / avg_words_per_sentence 
    
    figurality_idx = (metaphors_per_1000_sentences / 1000) * length_factor # Diviso 1000 per portarlo a %
    # La formula originale è (metafore/1000 frasi) * (20 / media_parole)
    # Se vogliamo un indice che può essere > 1, usiamo:
    # figurality_idx = metaphors_per_1000_sentences * length_factor
    # Se vogliamo un indice che sia più simile a una "percentuale di metafore per 1000 frasi, normalizzata per lunghezza":
    # figurality_idx = (num_metaphors / num_sentences) * length_factor 
    # L'interpretazione della richiesta "metafore/1000 frasi" può essere (num_metaphors / num_sentences) * 1000
    # oppure (num_metaphors / 1000) se num_sentences è vicino a 1000.
    # Assumendo la prima interpretazione:
    
    # ((num_metaphors / num_sentences) * 1000) * (20 / avg_words_per_sentence)
    # Questa formula produce numeri che possono essere grandi.
    # Se "metafore/1000 frasi" significa "densità di metafore per 1000 frasi",
    # allora (num_metaphors / num_sentences) è la densità per frase.
    # Per 1000 frasi, è (num_metaphors / num_sentences) * 1000.
    # figurality_idx = ((num_metaphors / num_sentences) * 1000) * (20 / avg_words_per_sentence)
    # Questo è un valore, non una percentuale. Se si vuole come "percentuale di metafore per 1000 frasi normalizzato"
    # si potrebbe dividere ulteriormente.
    # Manteniamo la formula come interpretata:
    fig_index = (num_metaphors / num_sentences * 1000) * (20 / avg_words_per_sentence) if num_sentences > 0 and avg_words_per_sentence > 0 else 0.0
    return fig_index


def main_predict():
    """Funzione principale per l'inferenza su nuovi testi."""
    logging.info("Avvio del processo di inferenza su nuovi testi...")

    # 0. Download NLTK 'punkt' se non già fatto (per Colab/primo avvio)
    try:
        nltk.data.find('tokenizers/punkt')
    except nltk.downloader.DownloadError:
        logging.info("Download del tokenizer 'punkt' di NLTK...")
        nltk.download('punkt')
    
    # 1. Controlla disponibilità GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Utilizzo del dispositivo: {device}")

    # 2. Caricamento Tokenizer e Modello Fine-tunato
    logging.info(f"Caricamento tokenizer e modello da {MODEL_PATH}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        model.to(device)
    except Exception as e:
        logging.error(f"Errore nel caricamento del modello o tokenizer: {e}")
        return

    # 3. Trova tutti i file .txt nella directory specificata
    text_files = glob.glob(os.path.join(NEW_TEXTS_DIR, "*.txt"))
    if not text_files:
        logging.warning(f"Nessun file .txt trovato in {NEW_TEXTS_DIR}. Processo terminato.")
        return
    
    logging.info(f"Trovati {len(text_files)} file .txt da analizzare.")

    results = []

    # 4. Processa ciascun file
    for txt_file_path in tqdm(text_files, desc="Processing text files"):
        filename = os.path.basename(txt_file_path)
        logging.info(f"\n--- Analisi del file: {filename} ---")
        
        try:
            with open(txt_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            logging.error(f"Errore nella lettura del file {filename}: {e}")
            results.append({
                'nome_testo': filename, 'indice_figuralita': 0,
                'conteggio_frasi': 0, 'conteggio_metafore': 0,
                'media_parole_frase': 0
            })
            continue

        if not content.strip():
            logging.warning(f"Il file {filename} è vuoto o contiene solo spazi bianchi.")
            results.append({
                'nome_testo': filename, 'indice_figuralita': 0,
                'conteggio_frasi': 0, 'conteggio_metafore': 0,
                'media_parole_frase': 0
            })
            continue

        # Suddividi in frasi
        sentences = split_into_sentences(content)
        num_sentences = len(sentences)

        if num_sentences == 0:
            logging.warning(f"Nessuna frase trovata in {filename} dopo la tokenizzazione.")
            results.append({
                'nome_testo': filename, 'indice_figuralita': 0,
                'conteggio_frasi': 0, 'conteggio_metafore': 0,
                'media_parole_frase': 0
            })
            continue
            
        logging.info(f"Numero di frasi estratte: {num_sentences}")

        # Predici metafore
        predictions = predict_metaphors_in_batch(sentences, tokenizer, model, device)
        num_metaphors = sum(predictions) # predictions contiene 0 o 1
        logging.info(f"Numero di metafore identificate: {num_metaphors}")

        # Calcola numero medio di parole per frase
        total_words = sum(len(s.split()) for s in sentences)
        avg_words_per_sentence = total_words / num_sentences if num_sentences > 0 else 0
        logging.info(f"Numero medio di parole per frase: {avg_words_per_sentence:.2f}")

        # Calcola indice di figuralità
        fig_index = calculate_figurality_index(num_metaphors, num_sentences, avg_words_per_sentence)
        logging.info(f"Indice di Figuralità Calcolato: {fig_index:.4f}")

        results.append({
            'nome_testo': filename,
            'indice_figuralita': fig_index,
            'conteggio_frasi': num_sentences,
            'conteggio_metafore': num_metaphors,
            'media_parole_frase': round(avg_words_per_sentence, 2)
        })

        # Opzionale: Analisi semantica delle frasi metaforiche
        # Questa parte richiede che tu abbia un modo per estrarre argomento/veicolo
        # dalle frasi identificate come metaforiche. Questo è un compito complesso (NER + Relation Extraction)
        # che va oltre la semplice classificazione. Se il dataset CAMI avesse queste info,
        # si potrebbero usare per le frasi del dataset. Per testi nuovi, è più difficile.
        # Esempio concettuale se avessi arg/vec:
        # for i, (sentence, pred) in enumerate(zip(sentences, predictions)):
        #     if pred == 1: # Se è una metafora
        #         print(f"Metafora trovata: {sentence}")
        #         # arg, vec = estrai_arg_vec(sentence) # Funzione ipotetica
        #         # if arg and vec and 'semantics' in globals(): # Controlla se il modulo semantics è importato
        #         #     dist = semantics.distanza_semantica(arg, vec)
        #         #     conc_arg = semantics.stima_concretezza(arg)
        #         #     conc_vec = semantics.stima_concretezza(vec)
        #         #     print(f"  Arg: {arg} (Conc: {conc_arg}), Vec: {vec} (Conc: {conc_vec}), Dist: {dist}")

    # 5. Salva i risultati in un CSV
    if results:
        results_df = pd.DataFrame(results)
        try:
            results_df.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8')
            logging.info(f"Risultati dell'inferenza salvati in: {OUTPUT_CSV_PATH}")
        except Exception as e:
            logging.error(f"Errore nel salvataggio del CSV dei risultati: {e}")
    else:
        logging.info("Nessun risultato da salvare.")

if __name__ == '__main__':
    # Crea la directory dei testi di input se non esiste
    os.makedirs(NEW_TEXTS_DIR, exist_ok=True)
    # Aggiungi qui file .txt di esempio in data/nuovi_testi/ per testare
    # Esempio:
    # with open(os.path.join(NEW_TEXTS_DIR, "esempio_testo1.txt"), "w", encoding="utf-8") as f:
    #     f.write("La vita è un lungo viaggio pieno di sorprese. Il tempo vola quando ci si diverte. Quel politico è una vecchia volpe.")
    # with open(os.path.join(NEW_TEXTS_DIR, "esempio_testo2.txt"), "w", encoding="utf-8") as f:
    #     f.write("Il cielo oggi è sereno. Il gatto dorme sulla sedia. Mi piace leggere libri.")
    
    main_predict()