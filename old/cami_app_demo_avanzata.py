import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification
import torch.nn.functional as F
import numpy as np
import pandas as pd

# --- Import per le visualizzazioni ---
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# --- SETUP ---
CLASSIFIER_DIR = "models/cami_classifier_tuned"
NER_DIR = "models/cami_ner_v2"
DATASET_PATH = "data/metafore_dataset.csv"
ID_TO_LABEL_CLASSIFIER = {0: "Letterale", 1: "Metafora"}

# --- MODELLO 1: CLASSIFICATORE (invariato) ---
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
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True).to(self.device)
            logits = self.model(**inputs).logits
            probabilities = F.softmax(logits, dim=-1).squeeze()
            confidence = torch.max(probabilities).item()
            predicted_class_id = torch.argmax(probabilities).item()
            return {"label": ID_TO_LABEL_CLASSIFIER[predicted_class_id], "confidence": confidence}

# --- CLASSE PER LE VISUALIZZAZIONI (MODIFICATA) ---
class CAMIVisualizer:
    def __init__(self, model_path: str, pca_model: PCA):
        """
        Carica un modello e riceve un modello PCA GIA' ADDESTRATO.
        """
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
        # Salva il modello PCA globale già addestrato
        self.pca_model = pca_model

    def _get_word_vector(self, hidden_states, token_ids, target_word):
        """Estrae e media i vettori dei sub-token per una parola specifica."""
        # Pulisce la parola target da eventuali spazi o punteggiatura per un matching migliore
        clean_target = target_word.strip().lower()
        tokens = [self.tokenizer.decode(t_id).lower() for t_id in token_ids]
        
        # Trova gli indici dei token che compongono la parola target
        word_indices = [i for i, token in enumerate(tokens) if clean_target in token]

        if not word_indices:
            # Se non trova la parola esatta, prova a cercare parti di essa (può aiutare con la tokenizzazione)
            word_indices = [i for i, token in enumerate(tokens) if any(sub_token in token for sub_token in clean_target.split())]
            if not word_indices:
                return None
        
        word_vectors = hidden_states[word_indices, :].mean(dim=0)
        return word_vectors.cpu().numpy()

    def plot_vector_space(self, sentence: str, arg: str, veh: str, label: str):
        """
        Crea un plot 2D usando il modello PCA globale pre-addestrato.
        """
        print(f"  -> Generazione plot dello spazio vettoriale 2D per la frase '{label}'...")
        
        with torch.no_grad():
            inputs = self.tokenizer(sentence, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            
            last_hidden_state = outputs.hidden_states[-1].squeeze(0)
            token_ids = inputs["input_ids"].squeeze(0)

            arg_vector = self._get_word_vector(last_hidden_state, token_ids, arg)
            veh_vector = self._get_word_vector(last_hidden_state, token_ids, veh)

            if arg_vector is None or veh_vector is None:
                print(f"    AVVISO: Impossibile trovare i vettori per '{arg}' o '{veh}' nel testo.")
                return

            # USA IL MODELLO PCA GLOBALE per trasformare, non per addestrare!
            transformed_vectors = self.pca_model.transform([arg_vector, veh_vector])
            
            plt.figure(figsize=(10, 8))
            plt.scatter(transformed_vectors[0, 0], transformed_vectors[0, 1], c='red', s=150, label=f'Argomento: "{arg}"', alpha=0.8)
            plt.text(transformed_vectors[0, 0], transformed_vectors[0, 1], f' "{arg}"', fontsize=14)
            
            plt.scatter(transformed_vectors[1, 0], transformed_vectors[1, 1], c='blue', s=150, label=f'Veicolo: "{veh}"', alpha=0.8)
            plt.text(transformed_vectors[1, 0], transformed_vectors[1, 1], f' "{veh}"', fontsize=14)

            # Calcola e mostra la distanza Euclidea
            distance = np.linalg.norm(transformed_vectors[0] - transformed_vectors[1])
            plt.plot([transformed_vectors[0, 0], transformed_vectors[1, 0]], 
                     [transformed_vectors[0, 1], transformed_vectors[1, 1]], 
                     'g--', 
                     label=f'Distanza: {distance:.2f}')

            plt.title(f'Spazio Vettoriale 2D ({label})\n"{sentence}"', fontsize=16)
            plt.xlabel("Componente Principale 1", fontsize=12)
            plt.ylabel("Componente Principale 2", fontsize=12)
            # Fissa i limiti degli assi per rendere i grafici confrontabili
            plt.xlim(-3.5, 3.5)
            plt.ylim(-3.5, 3.5)
            plt.grid(True)
            plt.legend()
            plt.show()

    def plot_attention_heatmap(self, sentence: str, label: str):
        """Crea una heatmap della matrice di attenzione (invariato)."""
        print(f"  -> Generazione heatmap di attenzione per la frase '{label}'...")

        with torch.no_grad():
            inputs = self.tokenizer(sentence, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            
            attentions = outputs.attentions[-1]
            tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            
            attention_head_matrix = attentions.squeeze(0)[0].cpu().numpy()

            plt.figure(figsize=(12, 10))
            sns.heatmap(attention_head_matrix, xticklabels=tokens, yticklabels=tokens, cmap="viridis")
            plt.title(f'Heatmap di Attenzione ({label})\n"{sentence}"', fontsize=16)
            plt.xticks(rotation=45, ha="right")
            plt.show()

# --- NUOVA FUNZIONE DI SETUP PER IL PCA GLOBALE ---
def setup_global_pca(model, tokenizer, device, dataset_path):
    """
    Addestra un modello PCA su un vocabolario ampio estratto dal dataset
    per creare uno spazio di riferimento coerente.
    """
    print("\n--- Setup del Modello PCA Globale ---")
    print("Caricamento del vocabolario dal dataset per creare uno spazio vettoriale di riferimento...")
    
    df = pd.read_csv(dataset_path, sep=';')
    df.dropna(subset=['argomento', 'veicolo'], inplace=True)
    
    # Estrai tutte le parole uniche per argomento e veicolo
    arg_words = set(df['argomento'].astype(str).str.lower())
    veh_words = set(df['veicolo'].astype(str).str.lower())
    vocab = list(arg_words.union(veh_words))
    
    print(f"Vocabolario di riferimento creato con {len(vocab)} parole uniche.")
    
    # Ottieni i vettori per ogni parola nel vocabolario
    word_vectors = []
    with torch.no_grad():
        for word in vocab:
            inputs = tokenizer(word, return_tensors="pt").to(device)
            outputs = model(**inputs)
            # Usiamo la media dei vettori dello stato nascosto come rappresentazione della parola
            vector = outputs.hidden_states[-1].squeeze(0).mean(dim=0).cpu().numpy()
            word_vectors.append(vector)

    # Addestra il modello PCA su tutti questi vettori
    print("Addestramento del modello PCA per la riduzione dimensionale...")
    pca = PCA(n_components=2, random_state=42)
    pca.fit(word_vectors)
    print("Modello PCA globale pronto.\n")
    return pca


# --- BLOCCO DI ESECUZIONE DEMO CON VISUALIZZAZIONI ---
if __name__ == "__main__":
    print("Avvio Demo Avanzata CAMI...")
    # Caricamento del classificatore (usato solo per mostrare l'output)
    classifier = CAMIClassifier(CLASSIFIER_DIR)
    
    # Caricamento del modello e tokenizer per le visualizzazioni
    vis_tokenizer = AutoTokenizer.from_pretrained(NER_DIR)
    vis_model = AutoModelForTokenClassification.from_pretrained(
        NER_DIR, output_hidden_states=True, output_attentions=True
    ).to(classifier.device)

    # 1. SETUP DEL PCA GLOBALE
    global_pca_model = setup_global_pca(vis_model, vis_tokenizer, classifier.device, DATASET_PATH)

    # 2. INIZIALIZZAZIONE DEL VISUALIZER CON IL PCA GLOBALE
    visualizer = CAMIVisualizer(NER_DIR, global_pca_model)
    print("Modelli pronti per l'analisi.\n")

    # Lista di frasi di test, incluse frasi letterali
    # Definiamo manualmente Argomento e Veicolo per la coerenza della demo
    test_data = [
        {"sentence": "La sua mente è un computer.", "arg": "mente", "veh": "computer"},
        {"sentence": "Il computer è sul tavolo.", "arg": "computer", "veh": "tavolo"},
        {"sentence": "Quell'avvocato è uno squalo.", "arg": "avvocato", "veh": "squalo"},
        {"sentence": "Quell'animale è uno squalo.", "arg": "animale", "veh": "squalo"},
        {"sentence": "La memoria è un registratore.", "arg": "memoria", "veh": "registratore"},
        {"sentence": "L'apparecchio è un registratore.", "arg": "apparecchio", "veh": "registratore"},
    ]

    for item in test_data:
        sentence = item['sentence']
        arg = item['arg']
        veh = item['veh']
        
        print(f"\n--- Analisi Frase: '{sentence}' ---")
        
        # Usiamo il classificatore per vedere cosa predice il modello
        classification_result = classifier.predict(sentence)
        label = classification_result['label']
        conf = classification_result['confidence']
        print(f"  -> Classificazione Predetta: {label} (Confidenza: {conf:.1%})")
        
        # Generiamo le visualizzazioni per TUTTE le frasi, usando arg e veh predefiniti
        if arg and veh:
            visualizer.plot_vector_space(sentence, arg, veh, label)
            visualizer.plot_attention_heatmap(sentence, label)
        else:
            print("  -> Visualizzazioni saltate (argomento o veicolo non specificati).")
    
    print("\n--- Fine Demo Avanzata ---")