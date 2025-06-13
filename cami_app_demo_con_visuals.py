import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification, pipeline
import torch.nn.functional as F
import numpy as np

# --- Import per le visualizzazioni ---
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# --- SETUP ---
CLASSIFIER_DIR = "models/cami_classifier_tuned"
NER_DIR = "models/cami_ner_v2"
ID_TO_LABEL_CLASSIFIER = {0: "Letterale", 1: "Metafora"}

# --- MODELLO 1: CLASSIFICATORE (invariato) ---
class CAMIClassifier:
    def __init__(self, model_path):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text):
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True).to(self.device)
            logits = self.model(**inputs).logits
            probabilities = F.softmax(logits, dim=-1).squeeze()
            confidence = torch.max(probabilities).item()
            predicted_class_id = torch.argmax(probabilities).item()
            return {"label": ID_TO_LABEL_CLASSIFIER[predicted_class_id], "confidence": confidence}

# --- MODELLO 2: ESTRATTORE NER (invariato) ---
class CAMIExtractor:
    def __init__(self, model_path):
        device_id = 0 if torch.backends.mps.is_available() else -1
        self.ner_pipeline = pipeline("token-classification", model=model_path, tokenizer=model_path, aggregation_strategy="simple", device=device_id)

    def extract(self, text: str) -> dict:
        ner_results = self.ner_pipeline(text)
        argomento = None
        veicolo = None
        for entity in ner_results:
            if entity['entity_group'] == 'ARG':
                argomento = entity['word']
            elif entity['entity_group'] == 'VEI':
                veicolo = entity['word']
        return {"argomento": argomento, "veicolo": veicolo}

# --- NUOVA CLASSE PER LE VISUALIZZAZIONI ---
class CAMIVisualizer:
    def __init__(self, model_path: str):
        """
        Carica un modello specificamente per l'analisi interna, 
        abilitando l'output degli stati nascosti e delle matrici di attenzione.
        """
        print("Caricamento del modello per la visualizzazione...")
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        # Carichiamo il modello chiedendogli di restituire anche questi output speciali
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_path, 
            output_hidden_states=True, 
            output_attentions=True
        )
        self.model.to(self.device)
        self.model.eval()

    def _get_word_vector(self, hidden_states, token_ids, target_word):
        """Estrae e media i vettori dei sub-token per una parola specifica."""
        word_indices = [i for i, token in enumerate(self.tokenizer.convert_ids_to_tokens(token_ids)) if target_word in token]
        if not word_indices:
            return None
        # Media i vettori di tutti i sub-token che compongono la parola
        word_vectors = hidden_states[word_indices, :].mean(dim=0)
        return word_vectors.cpu().numpy()

    def plot_vector_space(self, sentence: str, arg: str, veh: str):
        """Crea un plot 2D che mostra la divergenza tra i vettori di argomento e veicolo."""
        print("  -> Generazione plot dello spazio vettoriale 2D...")
        
        with torch.no_grad():
            inputs = self.tokenizer(sentence, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            
            # L'ultimo stato nascosto contiene i vettori finali per ogni token
            last_hidden_state = outputs.hidden_states[-1].squeeze(0)
            token_ids = inputs["input_ids"].squeeze(0)

            # Estraiamo i vettori per l'argomento e il veicolo
            arg_vector = self._get_word_vector(last_hidden_state, token_ids, arg)
            veh_vector = self._get_word_vector(last_hidden_state, token_ids, veh)

            if arg_vector is None or veh_vector is None:
                print("    AVVISO: Impossibile trovare i vettori per argomento o veicolo nel testo.")
                return

            # Usiamo PCA per ridurre la dimensionalità da 768 a 2
            pca = PCA(n_components=2)
            transformed_vectors = pca.fit_transform([arg_vector, veh_vector])
            
            plt.figure(figsize=(8, 6))
            # Plot dell'argomento (rosso)
            plt.scatter(transformed_vectors[0, 0], transformed_vectors[0, 1], c='red', s=100, label=f'Argomento: "{arg}"')
            plt.text(transformed_vectors[0, 0], transformed_vectors[0, 1], f' "{arg}"', fontsize=12)
            # Plot del veicolo (blu)
            plt.scatter(transformed_vectors[1, 0], transformed_vectors[1, 1], c='blue', s=100, label=f'Veicolo: "{veh}"')
            plt.text(transformed_vectors[1, 0], transformed_vectors[1, 1], f' "{veh}"', fontsize=12)

            plt.title(f'Spazio Vettoriale 2D per:\n"{sentence}"')
            plt.xlabel("Componente Principale 1")
            plt.ylabel("Componente Principale 2")
            plt.grid(True)
            plt.legend()
            plt.show() # Mostra il grafico. Devi chiuderlo per continuare.

    def plot_attention_heatmap(self, sentence: str):
        """Crea una heatmap della matrice di attenzione dell'ultimo layer."""
        print("  -> Generazione heatmap della matrice di attenzione...")

        with torch.no_grad():
            inputs = self.tokenizer(sentence, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)

            # Estraiamo le matrici di attenzione e i token
            attentions = outputs.attentions[-1] # Attenzione dell'ultimo layer
            tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            
            # Scegliamo una "testa" di attenzione (es. la prima) e la mediamo
            attention_head_matrix = attentions.squeeze(0)[0].cpu().numpy()

            plt.figure(figsize=(10, 8))
            sns.heatmap(attention_head_matrix, xticklabels=tokens, yticklabels=tokens, cmap="viridis")
            plt.title(f'Heatmap di Attenzione (Ultimo Layer, Testa 0)\n"{sentence}"')
            plt.show() # Mostra il grafico. Devi chiuderlo per continuare.


# --- BLOCCO DI ESECUZIONE DEMO CON VISUALIZZAZIONI ---
if __name__ == "__main__":
    print("Caricamento dei modelli CAMI...")
    classifier = CAMIClassifier(CLASSIFIER_DIR)
    extractor = CAMIExtractor(NER_DIR)
    visualizer = CAMIVisualizer(NER_DIR) # Il visualizer usa il modello NER
    print("Modelli pronti.\n")

    test_sentences = [
        "C'è un caldo che mi scioglie.",
        "C'è un caldo torrido.",
        "La luna è un occhio.",
        "La luna è un satellite.",
    ]

    for sentence in test_sentences:
        print(f"\n--- Analisi Frase: '{sentence}' ---")
        
        classification_result = classifier.predict(sentence)
        label = classification_result['label']
        conf = classification_result['confidence']
        print(f"  -> Classificazione: {label} (Grado di sicurezza: {conf:.1%})")
        
        if label == "Metafora":
            extraction_result = extractor.extract(sentence)
            arg = extraction_result.get("argomento")
            veh = extraction_result.get("veicolo")
            print(f"  -> Close Reading: Argomento='{arg}', Veicolo='{veh}'")
            
            # Solo se abbiamo trovato sia argomento che veicolo, procediamo con le visualizzazioni
            if arg and veh:
                visualizer.plot_vector_space(sentence, arg, veh)
                visualizer.plot_attention_heatmap(sentence)
            else:
                print("  -> Visualizzazioni saltate (argomento o veicolo non trovati).")
    
    print("\n--- Fine Demo ---")