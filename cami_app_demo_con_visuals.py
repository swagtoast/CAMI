import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification, pipeline
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from scipy.spatial.distance import cosine
import os


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
        # IMPORTANTE: Rimuoviamo aggregation_strategy per avere il controllo completo
        self.ner_pipeline = pipeline(
            "token-classification", 
            model=model_path, 
            tokenizer=model_path, 
            device=device_id
        )

    def extract(self, text: str) -> dict:
        ner_results = self.ner_pipeline(text)
        
        entities = {}
        current_entity_type = None
        current_entity_word = ""

        for token_data in ner_results:
            entity_label = token_data['entity']
            word = token_data['word']

            # Se il token è l'inizio di una nuova entità (B-...)
            if entity_label.startswith('B-'):
                # Prima salva l'entità precedente, se ce n'era una
                if current_entity_type:
                    entities[current_entity_type] = current_entity_word.replace("##", "")
                
                # Inizia la nuova entità
                current_entity_type = entity_label.split('-')[1] # Es. da 'B-ARG' prendi 'ARG'
                current_entity_word = word
            
            # Se il token è la continuazione di un'entità (I-...)
            elif entity_label.startswith('I-') and current_entity_type == entity_label.split('-')[1]:
                current_entity_word += word
            
            # Se il token è 'O' o un'altra entità
            else:
                # Salva l'entità che si è appena conclusa
                if current_entity_type:
                    entities[current_entity_type] = current_entity_word.replace("##", "")
                
                # Resetta tutto
                current_entity_type = None
                current_entity_word = ""

        # Salva l'ultima entità rimasta dopo il ciclo
        if current_entity_type:
            entities[current_entity_type] = current_entity_word.replace("##", "")

        # Estrai ARG e VEI dal dizionario
        argomento = entities.get('ARG')
        veicolo = entities.get('VEI')

        return {"argomento": argomento, "veicolo": veicolo}

# --- NUOVA CLASSE PER LE VISUALIZZAZIONI ---
class CAMIVisualizer:
    def __init__(self, model_path: str):
        print("Caricamento del modello per la visualizzazione...")
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_path,
            output_hidden_states=True,
            output_attentions=True,
            attn_implementation="eager"
        )
        self.model.to(self.device)
        self.model.eval()
        self.pca = None
        self.plot_limits = None

    def _find_flexible_start_char(self, sentence, target_word):
        """
        Cerca la parola target in modo flessibile, gestendo plurali semplici.
        Restituisce la parola trovata e la sua posizione iniziale.
        """
        clean_target = target_word.strip()
        sentence_lower = sentence.lower()

        # Strategia 1: Corrispondenza esatta
        start_char = sentence_lower.find(clean_target)
        if start_char != -1:
            return clean_target, start_char

        # Strategia 2: Gestione plurali comuni (o -> i, a -> e, e -> i).
        plural_forms_to_try = []
        if clean_target.endswith('o'):
            plural_forms_to_try.append(clean_target[:-1] + 'i')
        elif clean_target.endswith('a'):
            plural_forms_to_try.append(clean_target[:-1] + 'e')
        elif clean_target.endswith('e'):
             plural_forms_to_try.append(clean_target[:-1] + 'i')

        for form in plural_forms_to_try:
            start_char = sentence_lower.find(form)
            if start_char != -1:
                return form, start_char # Restituisce la forma trovata (es. 'astronauti')

        return None, -1

    def _get_word_vector(self, sentence: str, target_word: str, tokenized_inputs, hidden_states, num_layers_to_avg=4):
        """Metodo robusto che usa la ricerca flessibile."""
        
        found_word, target_start_char = self._find_flexible_start_char(sentence, target_word)

        if target_start_char == -1:
            print(f"    AVVISO: La parola '{target_word}' (o sue varianti) non è stata trovata nella frase.")
            return None
            
        target_end_char = target_start_char + len(found_word)
        word_ids = tokenized_inputs.word_ids()
        
        token_indices = [i for i, word_id in enumerate(word_ids) if word_id is not None and tokenized_inputs.word_to_chars(word_id).start >= target_start_char and tokenized_inputs.word_to_chars(word_id).end <= target_end_char]

        if not token_indices:
            return None

        stacked_layers = torch.stack(hidden_states[-num_layers_to_avg:])
        avg_hidden_states = stacked_layers.mean(dim=0).squeeze(0)
        word_vector = avg_hidden_states[token_indices, :].mean(dim=0)
        return word_vector.cpu().numpy()

    def calibrate_pca_and_limits(self, sentences, args, vehs):
        print(f"\n--- Calibrazione della PCA su {len(sentences)} esempi ---")
        all_vectors = []
        with torch.no_grad():
            for i, sentence in enumerate(sentences):
                tokenized_inputs = self.tokenizer(sentence, return_tensors="pt", truncation=True).to(self.device)
                outputs = self.model(**tokenized_inputs)
                hidden_states = outputs.hidden_states
                
                arg_vec = self._get_word_vector(sentence, args[i], tokenized_inputs, hidden_states)
                veh_vec = self._get_word_vector(sentence, vehs[i], tokenized_inputs, hidden_states)
                
                if arg_vec is not None: all_vectors.append(arg_vec)
                if veh_vec is not None: all_vectors.append(veh_vec)
        
        # CONTROLLO DI SICUREZZA: Evita errori se non vengono estratti abbastanza vettori
        if len(all_vectors) < 2:
            print("\nERRORE CRITICO: Non sono stati estratti abbastanza vettori per la calibrazione. La PCA non può essere addestrata.")
            print("Controlla il dataset e la logica di estrazione delle parole.")
            return

        self.pca = PCA(n_components=2)
        transformed_vectors = self.pca.fit_transform(all_vectors)
        
        min_x, max_x = transformed_vectors[:, 0].min(), transformed_vectors[:, 0].max()
        min_y, max_y = transformed_vectors[:, 1].min(), transformed_vectors[:, 1].max()
        padding_x = (max_x - min_x) * 0.1
        padding_y = (max_y - min_y) * 0.1
        self.plot_limits = ((min_x - padding_x, max_x + padding_x), (min_y - padding_y, max_y + padding_y))
        
        print("Calibrazione PCA completata e limiti del grafico impostati.")

    def plot_vector_space(self, sentence: str, arg: str, veh: str, label_type: str):
        print(f"  -> Generazione plot vettoriale per la frase ({label_type})...")
        if self.pca is None:
            print("    ERRORE: La PCA non è stata calibrata. Impossibile generare il grafico.")
            return

        with torch.no_grad():
            tokenized_inputs = self.tokenizer(sentence, return_tensors="pt", truncation=True).to(self.device)
            outputs = self.model(**tokenized_inputs)
            hidden_states = outputs.hidden_states

            arg_vector = self._get_word_vector(sentence, arg, tokenized_inputs, hidden_states)
            veh_vector = self._get_word_vector(sentence, veh, tokenized_inputs, hidden_states)

            if arg_vector is None or veh_vector is None: return

            cosine_dist = cosine(arg_vector, veh_vector)
            transformed_vectors = self.pca.transform([arg_vector, veh_vector])
            
            plt.figure(figsize=(9, 7))
            plt.scatter(transformed_vectors[0, 0], transformed_vectors[0, 1], c='red', s=100, label=f'Parola 1: "{arg}"')
            plt.text(transformed_vectors[0, 0], transformed_vectors[0, 1], f' "{arg}"', fontsize=12)
            plt.scatter(transformed_vectors[1, 0], transformed_vectors[1, 1], c='blue', s=100, label=f'Parola 2: "{veh}"')
            plt.text(transformed_vectors[1, 0], transformed_vectors[1, 1], f' "{veh}"', fontsize=12)
            plt.plot([transformed_vectors[0, 0], transformed_vectors[1, 0]], [transformed_vectors[0, 1], transformed_vectors[1, 1]], 'k--', alpha=0.5)

            plt.title(f'Spazio Vettoriale 2D ({label_type}) - Distanza Coseno: {cosine_dist:.4f}\n"{sentence}"', fontsize=12)
            plt.xlabel("Componente Principale 1 (Scala Fissa)")
            plt.ylabel("Componente Principale 2 (Scala Fissa)")
            
            if self.plot_limits:
                plt.xlim(self.plot_limits[0])
                plt.ylim(self.plot_limits[1])
                
            plt.grid(True)
            plt.legend()
            plt.show()

    def plot_attention_heatmap(self, sentence: str):
        print("  -> Generazione heatmap della matrice di attenzione...")
        with torch.no_grad():
            inputs = self.tokenizer(sentence, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            attentions = outputs.attentions[-1]
            tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            attention_head_matrix = attentions.squeeze(0)[0].cpu().numpy()
            plt.figure(figsize=(10, 8))
            sns.heatmap(attention_head_matrix, xticklabels=tokens, yticklabels=tokens, cmap="viridis")
            plt.title(f'Heatmap di Attenzione (Ultimo Layer, Testa 0)\n"{sentence}"')
            plt.show()


# --- BLOCCO DI ESECUZIONE DEMO CON CALIBRAZIONE E VISUALIZZAZIONI ---
if __name__ == "__main__":
    DATASET_PATH = "data/metafore_dataset.csv" # Assicurati che il percorso sia corretto

    print("Avvio demo con calibrazione...")
    classifier = CAMIClassifier(CLASSIFIER_DIR) 
    extractor = CAMIExtractor(NER_DIR)
    
    # Inizializza il visualizer SENZA un modello PCA
    visualizer = CAMIVisualizer(NER_DIR)

    # --- FASE DI CALIBRAZIONE ---
    # Carichiamo una parte del dataset per calibrare la PCA
    try:
        df_full = pd.read_csv(DATASET_PATH, sep=';', on_bad_lines='skip')
        df_sample = df_full.dropna(subset=['testo', 'argomento', 'veicolo']).sample(n=200, random_state=42)
        
        cal_sentences = df_sample['testo'].tolist()
        cal_args = df_sample['argomento'].tolist()
        cal_vehs = df_sample['veicolo'].tolist()
        
        # Addestra la PCA e imposta i limiti degli assi
        visualizer.calibrate_pca_and_limits(cal_sentences, cal_args, cal_vehs)

    except FileNotFoundError:
        print(f"AVVISO: File del dataset '{DATASET_PATH}' non trovato. Impossibile calibrare la PCA.")
    except Exception as e:
        print(f"Errore durante la calibrazione: {e}")


    # --- FASE DI ANALISI ---
    test_sentences = [
        "Milano è un ospedale.",
        "Milano è una città.",
    ]

    for sentence in test_sentences:
        print(f"\n--- Analisi Frase: '{sentence}' ---")
        
        classification_result = classifier.predict(sentence)
        label = classification_result['label']
        conf = classification_result['confidence']
        print(f"  -> Classificazione: {label} (Grado di sicurezza: {conf:.1%})")
        
        extraction_result = extractor.extract(sentence)
        arg = extraction_result.get("argomento")
        veh = extraction_result.get("veicolo")
        
        if arg and veh:
            print(f"  -> Estrazione: Argomento='{arg}', Veicolo='{veh}'")
            visualizer.plot_vector_space(sentence, arg, veh, label)
            # visualizer.plot_attention_heatmap(sentence) # Puoi de-commentare se serve
        else:
            print("  -> Visualizzazioni saltate (argomento o veicolo non estratti).")
    
    print("\n--- Fine Demo ---")