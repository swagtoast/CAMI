# Filename: cami_demo.py

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F  # Useremo questa funzione per le probabilità

# --- 1. CONFIGURAZIONE ---
# Percorso della cartella che contiene il nostro modello migliore.
MODEL_DIR = "models/cami_classifier_tuned"

# Definiamo le etichette in modo che siano leggibili dall'uomo.
# L'ordine è importante: l'etichetta 0 corrisponde a "Letterale", l'etichetta 1 a "Metafora".
ID_TO_LABEL = {
    0: "Letterale",
    1: "Metafora"
}

class CAMI:
    """
    La classe principale per il Classificatore Automatico di Metafore per l'Italiano.
    Questa classe carica il modello e fornisce i metodi per interagire con esso.
    """
    def __init__(self, model_path: str):
        """
        Il costruttore della classe. Viene eseguito quando creiamo un nuovo oggetto CAMI.
        Il suo compito è caricare il modello e il tokenizer dalla memoria e prepararli.
        """
        print(f"Caricamento del modello da: {model_path}")
        
        # Determina su quale dispositivo eseguire il modello (GPU se disponibile, altrimenti CPU)
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("Trovato dispositivo MPS (GPU Apple). L'inferenza sarà accelerata.")
        else:
            self.device = torch.device("cpu")
            print("Dispositivo MPS non trovato. L'inferenza verrà eseguita su CPU.")

        # Carica il tokenizer e il modello dai file salvati
        # Questo avviene una sola volta, all'avvio dell'applicazione.
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        # Sposta il modello sul dispositivo scelto (GPU o CPU)
        self.model.to(self.device)
        
        # Mette il modello in "modalità valutazione". Questo disattiva alcuni
        # comportamenti specifici del training (come il dropout) e lo rende più veloce.
        self.model.eval()
        
        print("Modello caricato e pronto per l'inferenza.")

    def predict(self, text: str) -> dict:
        """
        Esegue una previsione su una singola stringa di testo.
        
        Args:
            text (str): La frase da classificare.
            
        Returns:
            dict: Un dizionario contenente l'etichetta predetta e il punteggio di confidenza.
        """
        # torch.no_grad() è un gestore di contesto che dice a PyTorch:
        # "Stiamo solo facendo una previsione, non siamo in training".
        # Questo disattiva il calcolo dei gradienti e rende l'inferenza molto più veloce
        # e meno dispendiosa in termini di memoria. È una pratica fondamentale.
        with torch.no_grad():
            # 1. Tokenizzazione: Convertiamo la frase di testo in numeri (tokens)
            # che il modello può capire. `return_tensors="pt"` crea un tensore PyTorch.
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            
            # 2. Spostiamo i dati di input sullo stesso dispositivo del modello (GPU/CPU)
            inputs = {key: val.to(self.device) for key, val in inputs.items()}
            
            # 3. Inferenza: Passiamo i dati tokenizzati al modello per ottenere l'output.
            outputs = self.model(**inputs)
            
            # 4. Post-processing: L'output del modello sono "logits", ovvero punteggi grezzi.
            # Dobbiamo convertirli in probabilità che siano comprensibili.
            # La funzione SoftMax fa esattamente questo: trasforma i punteggi in una distribuzione
            # di probabilità che somma a 1.
            logits = outputs.logits
            probabilities = F.softmax(logits, dim=-1).squeeze()
            
            # 5. Estrazione dei risultati:
            # Troviamo l'indice della probabilità più alta (0 o 1)
            predicted_class_id = torch.argmax(probabilities).item()
            # Troviamo il valore della probabilità più alta (il nostro punteggio di confidenza)
            confidence = torch.max(probabilities).item()
            # Troviamo l'etichetta leggibile corrispondente all'indice
            predicted_label = ID_TO_LABEL[predicted_class_id]
            
            # Restituiamo i risultati in un comodo dizionario
            return {
                "label": predicted_label,
                "confidence": confidence
            }

# --- 3. BLOCCO DI ESECUZIONE PRINCIPALE ---
# Questo codice viene eseguito solo quando lanciamo `python cami_app.py` dal terminale.
# È il nostro modo per testare la classe CAMI.
if __name__ == "__main__":
    print("Avvio dell'applicazione CAMI...")
    
    # Creiamo un'istanza della nostra classe, caricando il modello.
    classifier = CAMI(model_path=MODEL_DIR)
    
    print("\n--- Esecuzione dei test di previsione ---")
    
    # Lista di frasi di esempio per testare il nostro classificatore
    test_sentences = [
        "Quell'avvocato è uno squalo.",          # Metafora ovvia
        "Il mio capo è un vulcano di idee.",       # Metafora comune
        "La notizia lo ha fulminato.",             # Metafora verbale
        "Il cielo è coperto di nuvole.",           # Frase letterale
        "Ho comprato il pane e il latte.",         # Frase letterale
        "La sua mente è un computer.",             # Metafora
        "Il computer è sul tavolo.",               # Letterale
    ]
    
    # Iteriamo su ogni frase e stampiamo la previsione del modello
    for sentence in test_sentences:
        result = classifier.predict(sentence)
        print(f"\nFrase: '{sentence}'")
        print(f"  -> Predizione: {result['label']} (Grado di sicurezza: {result['confidence']:.2%})")

    print("\n--- Test completati. ---")