# Filename: cami_app.py (Versione Corretta e Semplificata)

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import nltk
import os

# --- SETUP DI NLTK (Metodo Semplificato e Robusto) ---
# Questa chiamata è sicura. Scarica 'punkt' solo se non è già presente.
# Altrimenti, verifica rapidamente che sia aggiornato e prosegue.
print("Verifica/Download delle risorse NLTK (punkt)...")
nltk.download('punkt', quiet=True) # 'quiet=True' evita di stampare messaggi se è già aggiornato
nltk.download('punkt_tab')
print("Risorse NLTK pronte.")
# ----------------------------------------------------

# --- 1. CONFIGURAZIONE ---
MODEL_DIR = "models/cami_classifier_tuned"
ID_TO_LABEL = {0: "Letterale", 1: "Metafora"}

class CAMI:
    # ... (Il resto della classe CAMI rimane IDENTICO a prima) ...
    def __init__(self, model_path: str):
        print(f"Caricamento del modello da: {model_path}")
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("Trovato dispositivo MPS (GPU Apple). L'inferenza sarà accelerata.")
        else:
            self.device = torch.device("cpu")
            print("Dispositivo MPS non trovato. L'inferenza verrà eseguita su CPU.")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        print("Modello caricato e pronto per l'inferenza.")

    def predict(self, text: str) -> dict:
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            inputs = {key: val.to(self.device) for key, val in inputs.items()}
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = F.softmax(logits, dim=-1).squeeze()
            predicted_class_id = torch.argmax(probabilities).item()
            confidence = torch.max(probabilities).item()
            predicted_label = ID_TO_LABEL[predicted_class_id]
            return {"label": predicted_label, "confidence": confidence}

    def analyze_corpus(self, corpus_path: str, confidence_threshold: float = 0.5) -> dict:
        print(f"\n--- Inizio Analisi Distant Reading per il file: {os.path.basename(corpus_path)} ---")
        if not os.path.exists(corpus_path):
            return {"error": f"File non trovato: {corpus_path}"}
        with open(corpus_path, 'r', encoding='utf-8') as f:
            full_text = f.read()
        sentences = nltk.sent_tokenize(full_text, language='italian')
        total_sentences = len(sentences)
        metaphorical_sentences = 0
        total_words = 0
        found_metaphors = []
        if total_sentences == 0:
            return {"error": "Il file è vuoto o non contiene frasi riconoscibili."}
        for i, sentence in enumerate(sentences):
            clean_sentence = sentence.strip().replace('\n', ' ')
            if not clean_sentence:
                continue
            total_words += len(clean_sentence.split())
            result = self.predict(clean_sentence)
            if result["label"] == "Metafora" and result["confidence"] > confidence_threshold:
                metaphorical_sentences += 1
                found_metaphors.append({"text": clean_sentence, "confidence": result["confidence"]})
            if (i + 1) % 100 == 0:
                print(f"  ... Analizzate {i + 1}/{total_sentences} frasi ...")
        print("Analisi completata.")
        average_sentence_length = total_words / total_sentences
        figurality_density = metaphorical_sentences / total_sentences
        figurality_index = figurality_density / average_sentence_length if average_sentence_length > 0 else 0
        report = {
            "file_name": os.path.basename(corpus_path), "total_sentences": total_sentences,
            "metaphorical_sentences": metaphorical_sentences, "total_words": total_words,
            "average_sentence_length": average_sentence_length, "figurality_density": figurality_density,
            "figurality_index": figurality_index, "found_metaphors": found_metaphors
        }
        return report

# Blocco principale per l'esecuzione dello script

if __name__ == "__main__":
    # Definiamo il percorso della cartella che contiene i testi da analizzare
    CORPUS_DIR = os.path.join("data", "nuovi_testi")

    print(f"Avvio dell'applicazione CAMI. Analisi dei file in: '{CORPUS_DIR}'")
    
    # Ci assicuriamo che la cartella esista, altrimenti la creiamo per l'utente
    os.makedirs(CORPUS_DIR, exist_ok=True)
    
    # Carichiamo il nostro classificatore
    classifier = CAMI(model_path=MODEL_DIR)
    
    # Troviamo tutti i file che terminano con .txt nella cartella specificata
    try:
        all_files = os.listdir(CORPUS_DIR)
        text_files = [f for f in all_files if f.endswith(".txt")]
    except FileNotFoundError:
        print(f"ERRORE: La cartella '{CORPUS_DIR}' non è stata trovata.")
        text_files = []

    if not text_files:
        print("\nNessun file .txt trovato nella cartella.")
        print(f"Per favore, aggiungi uno o più file di testo (romanzi, articoli, ecc.) in '{CORPUS_DIR}' e riesegui lo script.")
    else:
        print(f"\nTrovati {len(text_files)} file di testo. Inizio analisi...")
        
        # Iteriamo su ogni file di testo e lanciamo l'analisi
        for filename in text_files:
            full_path = os.path.join(CORPUS_DIR, filename)
            report = classifier.analyze_corpus(full_path)
            
            # Stampiamo un report dettagliato per ogni file analizzato
            if "error" not in report:
                print("\n==========================================================")
                print(f"RISULTATI PER: {report['file_name']}")
                print("==========================================================")
                print(f"Frasi totali: {report['total_sentences']}")
                print(f"Frasi metaforiche trovate: {report['metaphorical_sentences']}")
                print(f"Parole totali: {report['total_words']}")
                print(f"Lunghezza media delle frasi: {report['average_sentence_length']:.2f} parole")
                print(f"Densità di figuralità (metafore/totale): {report['figurality_density']:.2%}")
                print(f"INDICE DI FIGURALITÀ: {report['figurality_index']:.4f}")
                
                # Stampiamo le prime 10 metafore trovate, se ce ne sono
                if report['found_metaphors']:
                    print("\nPrime 10 metafore individuate (ordinate per grado di sicurezza):")
                    # Ordiniamo le metafore per grado di sicurezza decrescente prima di stamparle
                    sorted_metaphors = sorted(report['found_metaphors'], key=lambda x: x['confidence'], reverse=True)
                    for metaphor in sorted_metaphors[:10]:
                        print(f"  - '{metaphor['text']}' (Grado di sicurezza: {metaphor['confidence']:.1%})")
                print("==========================================================\n")
            else:
                print(f"\nERRORE durante l'analisi del file {filename}: {report['error']}")
    
    print("Processo terminato.")