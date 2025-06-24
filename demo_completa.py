import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification, pipeline
import torch.nn.functional as F
import numpy as np
import pandas as pd
import spacy

# Import per la visualizzazione e il calcolo della distanza
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from scipy.spatial.distance import cosine

# --- CONFIGURAZIONE GLOBALE ---
CLASSIFIER_DIR = "models/cami_classifier_tuned"
NER_DIR = "models/cami_ner_v5_final" 
ID_TO_LABEL_CLASSIFIER = {0: "Letterale", 1: "Metafora"}

# Caricamento del modello linguistico spaCy
print("Caricamento del modello linguistico spaCy 'it_core_news_lg'...")
nlp = spacy.load("it_core_news_lg")
print("Modello spaCy caricato.")

# --- CLASSE 1: Classificatore di Frasi ---
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
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(self.device)
            logits = self.model(**inputs).logits
            probabilities = F.softmax(logits, dim=-1).squeeze()
            confidence = torch.max(probabilities).item()
            predicted_class_id = torch.argmax(probabilities).item()
            return {"label": ID_TO_LABEL_CLASSIFIER[predicted_class_id], "confidence": confidence}

# --- CLASSE 2: Estrattore di Argomento e Veicolo (NER) ---
class CAMIExtractor:
    def __init__(self, model_path):
        device_num = 0 if torch.backends.mps.is_available() else -1
        self.ner_pipeline = pipeline(
            "token-classification", 
            model=model_path, 
            tokenizer=model_path, 
            device=device_num
        )
        print(f"Extractor (NER) caricato in modalità di aggregazione manuale (più robusta).")

    def extract(self, text: str) -> dict:
        try:
            ner_results = self.ner_pipeline(text)
            entities = {}
            current_entity_words = []
            current_entity_type = None
            for token_data in ner_results:
                entity_label = token_data['entity']
                word = token_data['word']
                if word.startswith("##"):
                    if current_entity_words:
                        current_entity_words[-1] = current_entity_words[-1] + word.replace("##", "")
                    continue
                if entity_label.startswith('B-'):
                    if current_entity_type:
                        entities[current_entity_type] = " ".join(current_entity_words)
                    current_entity_type = entity_label.split('-')[1]
                    current_entity_words = [word]
                elif entity_label.startswith('I-') and current_entity_type == entity_label.split('-')[1]:
                    current_entity_words.append(word)
                else:
                    if current_entity_type:
                        entities[current_entity_type] = " ".join(current_entity_words)
                    current_entity_type = None
                    current_entity_words = []
            if current_entity_type:
                entities[current_entity_type] = " ".join(current_entity_words)
            return {"argomento": entities.get('ARG'), "veicolo": entities.get('VEI')}
        except Exception as e:
            print(f"  -> Errore durante l'estrazione NER: {e}")
            return {"argomento": None, "veicolo": None}

# --- CLASSE 3: Visualizzatore dello Spazio Semantico ---
class CAMIVisualizer:
    def __init__(self, model_path: str, reducer_model: TSNE):
        print("Caricamento del modello per la visualizzazione...")
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_path, 
            output_hidden_states=True, 
            output_attentions=True
        )
        self.model.to(self.device)
        self.model.eval()
        self.reducer = reducer_model

    def _get_word_vector(self, hidden_states, tokenized_sentence, sentence_text, target_word):
        doc = nlp(sentence_text)
        target_char_start, target_char_end = -1, -1
        for token in doc:
            if token.text.lower() == target_word.lower():
                target_char_start = token.idx
                target_char_end = token.idx + len(token.text)
                break
        if target_char_start == -1: return None
        word_indices = [i for i, _ in enumerate(tokenized_sentence.input_ids[0]) if tokenized_sentence.token_to_chars(i) and tokenized_sentence.token_to_chars(i).start >= target_char_start and tokenized_sentence.token_to_chars(i).end <= target_char_end]
        if not word_indices: return None
        return hidden_states[word_indices, :].mean(dim=0).cpu().numpy()

    def get_vectors_and_attentions(self, sentence: str, arg: str, veh: str):
        with torch.no_grad():
            inputs = self.tokenizer(sentence, return_tensors="pt", truncation=True, max_length=512).to(self.device)
            outputs = self.model(**inputs)
            last_hidden_state = outputs.hidden_states[-1].squeeze(0)
            arg_vector = self._get_word_vector(last_hidden_state, inputs, sentence, arg)
            veh_vector = self._get_word_vector(last_hidden_state, inputs, sentence, veh)
            attentions = outputs.attentions[-1].squeeze(0).mean(dim=0).cpu().numpy()
            tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            return arg_vector, veh_vector, attentions, tokens

    def plot_semantic_comparison(self, info1, info2):
        fig, ax = plt.subplots(figsize=(15, 8))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])

        fig.text(0.1, 0.97, f'Frase 1: "{info1["sentence"]}"', ha='left', va='top', fontsize=12, color='darkred', weight='bold')
        fig.text(0.1, 0.93, f'Frase 2: "{info2["sentence"]}"', ha='left', va='top', fontsize=12, color='darkblue', weight='bold')
        fig.text(0.1, 0.89, f'-> Classificazione Frase 1: {info1["label"]} (Sicurezza: {info1["confidence"]:.1%}) / Distanza: {info1["distance"]:.4f}', ha='left', va='top', fontsize=12, color='red')
        fig.text(0.1, 0.85, f'-> Classificazione Frase 2: {info2["label"]} (Sicurezza: {info2["confidence"]:.1%}) / Distanza: {info2["distance"]:.4f}', ha='left', va='top', fontsize=12, color='blue')

        ax_inset = fig.add_axes([0.1, 0.1, 0.8, 0.7])
        ax_inset.scatter(info1["arg_2d"][0], info1["arg_2d"][1], c='darkred', s=150, zorder=5, label=f'Argomento 1: "{info1["arg_str"]}"')
        ax_inset.scatter(info1["veh_2d"][0], info1["veh_2d"][1], c='lightcoral', s=150, zorder=5, label=f'Veicolo 1: "{info1["veh_str"]}"')
        ax_inset.plot([info1["arg_2d"][0], info1["veh_2d"][0]], [info1["arg_2d"][1], info1["veh_2d"][1]], '--', color='red', alpha=0.8)
        ax_inset.scatter(info2["arg_2d"][0], info2["arg_2d"][1], c='darkblue', s=150, zorder=5, label=f'Argomento 2: "{info2["arg_str"]}"')
        ax_inset.scatter(info2["veh_2d"][0], info2["veh_2d"][1], c='lightblue', s=150, zorder=5, label=f'Veicolo 2: "{info2["veh_str"]}"')
        ax_inset.plot([info2["arg_2d"][0], info2["veh_2d"][0]], [info2["arg_2d"][1], info2["veh_2d"][1]], ':', color='blue', alpha=0.8)

        for info, color in [(info1, 'red'), (info2, 'blue')]:
            ax_inset.text(info["arg_2d"][0], info["arg_2d"][1] + 0.1, info["arg_str"], color=color, ha='center', fontsize=11)
            ax_inset.text(info["veh_2d"][0], info["veh_2d"][1] + 0.1, info["veh_str"], color=color, ha='center', fontsize=11)

        ax_inset.set_xlabel("Componente t-SNE 1", fontsize=12)
        ax_inset.set_ylabel("Componente t-SNE 2", fontsize=12)
        ax_inset.grid(True, linestyle='--', alpha=0.6)
        ax_inset.legend(loc='best', fontsize=10)
        plt.show()

    def plot_attention_heatmap(self, sentence, label, attentions, tokens, cmap):
        print(f"  -> Generazione heatmap di attenzione per la frase ({label})...")
        plt.figure(figsize=(12, 8))
        sns.heatmap(attentions, xticklabels=tokens, yticklabels=tokens, cmap=cmap, annot=False)
        plt.title(f'Heatmap di Attenzione ({label})\n"{sentence}"', fontsize=16)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()

# --- Funzione di Setup per t-SNE ---
def setup_tsne():
    """Crea e restituisce un'istanza del modello t-SNE."""
    print("\n--- Setup del Riduttore Dimensionale (t-SNE) ---")
    tsne_reducer = TSNE(n_components=2, random_state=42, max_iter=1200, perplexity=3, init='random', learning_rate='auto')
    print("Modello t-SNE inizializzato e pronto.\n")
    return tsne_reducer

# --- Blocco di Esecuzione Principale ---
if __name__ == "__main__":
    print("Avvio Demo Automatica CAMI...")
    
    classifier = CAMIClassifier(CLASSIFIER_DIR)
    extractor = CAMIExtractor(NER_DIR)
    
    tsne_reducer = setup_tsne()
    visualizer = CAMIVisualizer(NER_DIR, tsne_reducer)
    
    print("--- Modelli pronti. Inizio analisi delle coppie di frasi. ---\n")

    sentence_pairs = [
        ("Questo cuoco è un cane.", "Questo cuoco è bravo."),
        ("Il mio sedere è una mongolfiera.", "Ho un sedere grande."),
        ("Lo studente è una volpe.", "Lo studente è furbo."),
    ]

    for i, (sent1, sent2) in enumerate(sentence_pairs):
        print(f"--- ANALISI COPPIA #{i+1} ---")
        print(f"  Frase 1: \"{sent1}\"")
        print(f"  Frase 2: \"{sent2}\"")
        
        class1 = classifier.predict(sent1)
        class2 = classifier.predict(sent2)
        
        print(f"    -> Classificazione Frase 1: {class1['label']} (Grado di sicurezza: {class1['confidence']:.1%})")
        print(f"    -> Classificazione Frase 2: {class2['label']} (Grado di sicurezza: {class2['confidence']:.1%})")

        extract1 = extractor.extract(sent1)
        arg1, veh1 = extract1.get("argomento"), extract1.get("veicolo")
        
        extract2 = extractor.extract(sent2)
        arg2, veh2 = extract2.get("argomento"), extract2.get("veicolo")

        if not all([arg1, veh1, arg2, veh2]):
            print("    -> ERRORE: Estrazione NER fallita per una o più parti. Visualizzazione saltata.")
            print("-" * 50 + "\n")
            continue
        
        print(f"    -> Entità Estratte: ('{arg1}', '{veh1}') e ('{arg2}', '{veh2}')")

        arg1_vec, veh1_vec, attentions1, tokens1 = visualizer.get_vectors_and_attentions(sent1, arg1, veh1)
        arg2_vec, veh2_vec, attentions2, tokens2 = visualizer.get_vectors_and_attentions(sent2, arg2, veh2)
        
        if any(v is None for v in [arg1_vec, veh1_vec, arg2_vec, veh2_vec]):
            print("    -> ERRORE: Estrazione vettori fallita. Visualizzazione saltata.")
            print("-" * 50 + "\n")
            continue
        
        dist1 = cosine(arg1_vec, veh1_vec)
        dist2 = cosine(arg2_vec, veh2_vec)

        all_vectors = [arg1_vec, veh1_vec, arg2_vec, veh2_vec]
        transformed_vectors = tsne_reducer.fit_transform(np.array(all_vectors))

        info1 = {"sentence": sent1, "label": class1["label"], "confidence": class1["confidence"], "arg_str": arg1, "veh_str": veh1, "arg_2d": transformed_vectors[0], "veh_2d": transformed_vectors[1], "distance": dist1}
        info2 = {"sentence": sent2, "label": class2["label"], "confidence": class2["confidence"], "arg_str": arg2, "veh_str": veh2, "arg_2d": transformed_vectors[2], "veh_2d": transformed_vectors[3], "distance": dist2}

        visualizer.plot_semantic_comparison(info1, info2)
        visualizer.plot_attention_heatmap(sent1, class1["label"], attentions1, tokens1, cmap="Reds")
        visualizer.plot_attention_heatmap(sent2, class2["label"], attentions2, tokens2, cmap="Blues")
        
        print("-" * 50 + "\n")

    print("\n--- Fine Demo Completa ---")