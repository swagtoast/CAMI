# Filename: demo_completa.py (Punta al modello NER finale)

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification, pipeline
import torch.nn.functional as F
import numpy as np
import pandas as pd
import spacy

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# --- CONFIGURAZIONE ---
CLASSIFIER_DIR = "models/cami_classifier_tuned"
# MODIFICA CHIAVE: Puntiamo al nuovo modello NER finale
NER_DIR = "models/cami_ner_v5_final" 
DATASET_PATH = "data/metafore_dataset.csv"
ID_TO_LABEL_CLASSIFIER = {0: "Letterale", 1: "Metafora"}

# (Il resto del codice delle classi CAMIClassifier, CAMIExtractor, CAMIVisualizer
# e della funzione setup_global_pca rimane IDENTICO alla versione precedente.
# Non lo ripeto qui per brevità, ma puoi usare quello dell'ultima risposta.)

# ... (Incollare qui il resto del codice delle classi e della funzione di setup)
class CAMIClassifier:
    def __init__(self, model_path):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        print(f"Classifier caricato su dispositivo: {self.device}")
    def predict(self, text):
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)
            logits = self.model(**inputs).logits
            probabilities = F.softmax(logits, dim=-1).squeeze()
            confidence = torch.max(probabilities).item()
            predicted_class_id = torch.argmax(probabilities).item()
            return {"label": ID_TO_LABEL_CLASSIFIER[predicted_class_id], "confidence": confidence}
class CAMIExtractor:
    def __init__(self, model_path):
        self.ner_pipeline = pipeline("token-classification",model=model_path,tokenizer=model_path,aggregation_strategy="simple",device=0 if torch.backends.mps.is_available() else -1)
        print(f"Extractor (NER) caricato e pronto.")
    def extract(self, text: str) -> dict:
        try:
            ner_results = self.ner_pipeline(text)
            argomento = None
            veicolo = None
            for entity in ner_results:
                if entity['entity_group'] == 'ARG':
                    argomento = entity['word']
                elif entity['entity_group'] == 'VEI':
                    veicolo = entity['word']
            return {"argomento": argomento, "veicolo": veicolo}
        except Exception as e:
            print(f"  -> Errore durante l'estrazione NER: {e}")
            return {"argomento": None, "veicolo": None}
class CAMIVisualizer:
    def __init__(self, model_path: str, pca_model: PCA):
        print("Caricamento del modello per la visualizzazione...")
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path, output_hidden_states=True, output_attentions=True)
        self.model.to(self.device)
        self.model.eval()
        self.pca_model = pca_model
    def _get_word_vector(self, hidden_states, token_ids, target_word):
        clean_target = target_word.strip().lower().replace(" ", "")
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
        word_indices = [i for i, token in enumerate(tokens) if clean_target in token.lower()]
        if not word_indices: return None
        word_vectors = hidden_states[word_indices, :].mean(dim=0)
        return word_vectors.cpu().numpy()
    def plot_vector_space(self, sentence: str, arg: str, veh: str, label: str):
        print(f"  -> 3. Generazione plot dello spazio vettoriale 2D...")
        with torch.no_grad():
            inputs = self.tokenizer(sentence, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs, output_hidden_states=True)
            last_hidden_state = outputs.hidden_states[-1].squeeze(0)
            token_ids = inputs["input_ids"].squeeze(0)
            arg_vector = self._get_word_vector(last_hidden_state, token_ids, arg)
            veh_vector = self._get_word_vector(last_hidden_state, token_ids, veh)
            if arg_vector is None or veh_vector is None:
                print(f"    AVVISO: Impossibile generare il plot. Vettori non trovati per '{arg}' o '{veh}'.")
                return
            transformed_vectors = self.pca_model.transform([arg_vector, veh_vector])
            plt.figure(figsize=(10, 8))
            plt.scatter(transformed_vectors[0, 0], transformed_vectors[0, 1], c='red', s=150, label=f'Argomento: "{arg}"', alpha=0.8, zorder=5)
            plt.text(transformed_vectors[0, 0] + 0.05, transformed_vectors[0, 1], f' "{arg}"', fontsize=14)
            plt.scatter(transformed_vectors[1, 0], transformed_vectors[1, 1], c='blue', s=150, label=f'Veicolo: "{veh}"', alpha=0.8, zorder=5)
            plt.text(transformed_vectors[1, 0] + 0.05, transformed_vectors[1, 1], f' "{veh}"', fontsize=14)
            distance = np.linalg.norm(transformed_vectors[0] - transformed_vectors[1])
            plt.plot([transformed_vectors[0, 0], transformed_vectors[1, 0]], [transformed_vectors[0, 1], transformed_vectors[1, 1]],'g--', label=f'Distanza Semantica: {distance:.2f}')
            plt.title(f'Spazio Vettoriale ({label})\n"{sentence}"', fontsize=16)
            plt.xlabel("Componente Principale 1", fontsize=12)
            plt.ylabel("Componente Principale 2", fontsize=12)
            plt.axhline(0, color='grey', linewidth=0.5)
            plt.axvline(0, color='grey', linewidth=0.5)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.legend()
            plt.show()
    def plot_attention_heatmap(self, sentence: str, label: str):
        print(f"  -> 4. Generazione heatmap della matrice di attenzione...")
        with torch.no_grad():
            inputs = self.tokenizer(sentence, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs, output_attentions=True)
            attentions = outputs.attentions[-1]
            tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            attention_matrix = attentions.squeeze(0).mean(dim=0).cpu().numpy()
            plt.figure(figsize=(12, 10))
            sns.heatmap(attention_matrix, xticklabels=tokens, yticklabels=tokens, cmap="viridis", annot=False)
            plt.title(f'Heatmap di Attenzione Media ({label})\n"{sentence}"', fontsize=16)
            plt.xticks(rotation=45, ha="right")
            plt.yticks(rotation=0)
            plt.show()
def setup_global_pca(model, tokenizer, device, dataset_path):
    print("\n--- Setup del Modello PCA Globale (per grafici confrontabili) ---")
    df = pd.read_csv(dataset_path, sep=';')
    df.dropna(subset=['argomento', 'veicolo'], inplace=True)
    arg_words = set(df['argomento'].astype(str).str.lower())
    veh_words = set(df['veicolo'].astype(str).str.lower())
    vocab = list(arg_words.union(veh_words))
    word_vectors = []
    with torch.no_grad():
        for word in vocab:
            if not word or pd.isna(word): continue
            inputs = tokenizer(word, return_tensors="pt").to(device)
            vector = model(**inputs, output_hidden_states=True).hidden_states[-1].squeeze(0).mean(dim=0).cpu().numpy()
            if np.isfinite(vector).all():
                word_vectors.append(vector)
            else:
                print(f"  -> Avviso: il vettore per la parola '{word}' non sarà usato.")
    print(f"Addestramento del modello PCA su {len(word_vectors)} vettori validi...")
    pca = PCA(n_components=2, random_state=42)
    pca.fit(word_vectors)
    print("Modello PCA globale pronto.\n")
    return pca

# --- BLOCCO DI ESECUZIONE PRINCIPALE ---
if __name__ == "__main__":
    print("Avvio Demo Completa e Autonoma di CAMI...")
    
    classifier = CAMIClassifier(CLASSIFIER_DIR)
    extractor = CAMIExtractor(NER_DIR)
    
    vis_model = AutoModelForTokenClassification.from_pretrained(NER_DIR).to(classifier.device)
    vis_tokenizer = AutoTokenizer.from_pretrained(NER_DIR)

    global_pca_model = setup_global_pca(vis_model, vis_tokenizer, classifier.device, DATASET_PATH)

    visualizer = CAMIVisualizer(NER_DIR, global_pca_model)
    print("--- Tutti i modelli sono pronti per l'analisi. ---\n")

    test_sentences = ["Quell'avvocato è uno squalo.", "Quell'animale è uno squalo.", "Quel fanciullo è una volpe.", "Quel fanciullo è uno scolaro."]

    for sentence in test_sentences:
        print(f"--- Analisi Frase: '{sentence}' ---")
        classification_result = classifier.predict(sentence)
        label = classification_result['label']
        conf = classification_result['confidence']
        print(f"  -> 1. Classificazione: {label} (Confidenza: {conf:.1%})")
        extraction_result = extractor.extract(sentence)
        arg = extraction_result.get("argomento")
        veh = extraction_result.get("veicolo")
        print(f"  -> 2. Estrazione NER: Argomento='{arg}', Veicolo='{veh}'")
        if arg and veh:
            visualizer.plot_vector_space(sentence, arg, veh, label)
            visualizer.plot_attention_heatmap(sentence, label)
        else:
            print("  -> Visualizzazioni saltate.")
    
    print("\n--- Fine Demo Completa ---")