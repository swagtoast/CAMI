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

# --- BLOCCO DI ESECUZIONE PRINCIPALE (invariato) ---
if __name__ == "__main__":
    print("Avvio dell'applicazione CAMI...")
    classifier = CAMI(model_path=MODEL_DIR)
    
    print("\n--- Test 1: Previsione su singola frase ---")
    test_sentence = "Il tempo è un ladro che ruba i ricordi."
    result = classifier.predict(test_sentence)
    print(f"Frase: '{test_sentence}'")
    print(f"  -> Predizione: {result['label']} (Confidenza: {result['confidence']:.2%})")
    
    test_corpus_path = "corpus_di_prova.txt"
    test_corpus_content = """
    Il sole tramontava all'orizzonte, dipingendo il cielo di rosso. Era uno spettacolo magnifico.
    Maria sentiva che il suo cuore era una prigione di ghiaccio, incapace di provare calore. 
    Prese il telefono e chiamò sua madre. Parlarono per quasi un'ora. 
    In quel mondo di squali, lui era solo un piccolo pesce. Doveva imparare a nuotare velocemente.
    Il computer era un modello di ultima generazione, con un processore molto potente.
    La vita, a volte, è un'arancia amara. Devi sbucciarla con pazienza per trovare la dolcezza.
    Il treno arrivò in stazione puntuale, alle 18:30.
    """
    with open(test_corpus_path, 'w', encoding='utf-8') as f:
        f.write(test_corpus_content)
    
    report = classifier.analyze_corpus(test_corpus_path)
    
    if "error" not in report:
        print("\n--- Risultati dell'analisi del corpus ---")
        print(f"File analizzato: {report['file_name']}")
        print(f"Frasi totali: {report['total_sentences']}")
        print(f"Frasi metaforiche trovate: {report['metaphorical_sentences']}")
        print(f"Lunghezza media delle frasi: {report['average_sentence_length']:.2f} parole")
        print(f"Densità di figuralità (metafore/totale): {report['figurality_density']:.2%}")
        print(f"INDICE DI FIGURALITÀ: {report['figurality_index']:.4f}")
        print("\nMetafore individuate:")
        for metaphor in report['found_metaphors']:
            print(f"  - '{metaphor['text']}' (Confidenza: {metaphor['confidence']:.1%})")
        print("------------------------------------------")
    else:
        print(f"Errore durante l'analisi: {report['error']}")
        
    os.remove(test_corpus_path)