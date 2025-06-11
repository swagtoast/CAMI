import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification, pipeline
import torch.nn.functional as F
import nltk

# --- SETUP ---
nltk.download('punkt', quiet=True)
CLASSIFIER_DIR = "models/cami_classifier_tuned"
NER_DIR = "models/cami_ner_v1"
ID_TO_LABEL_CLASSIFIER = {0: "Letterale", 1: "Metafora"}

# --- MODELLO 1: CLASSIFICATORE ---
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

# --- MODELLO 2: ESTRATTORE NER (CLOSE READING) ---
class CAMIExtractor:
    def __init__(self, model_path):
        # Per NER, la classe 'pipeline' di transformers è incredibilmente comoda.
        # Gestisce tutta la tokenizzazione e il post-processing per noi.
        self.ner_pipeline = pipeline(
            "token-classification",
            model=model_path,
            tokenizer=model_path,
            aggregation_strategy="simple", # Raggruppa i sub-token (es. 'av-vo-ca-to' diventa 'avvocato')
            device=0 if torch.backends.mps.is_available() else -1 # 0 per MPS/CUDA, -1 per CPU
        )

    def extract(self, text: str) -> dict:
        """Estrae Argomento e Veicolo da un testo."""
        ner_results = self.ner_pipeline(text)
        
        argomento = None
        veicolo = None
        
        for entity in ner_results:
            if entity['entity_group'] == 'ARG':
                argomento = entity['word']
            elif entity['entity_group'] == 'VEI':
                veicolo = entity['word']
        
        return {"argomento": argomento, "veicolo": veicolo}

# --- BLOCCO DI ESECUZIONE DEMO ---
if __name__ == "__main__":
    print("Caricamento dei modelli CAMI...")
    classifier = CAMIClassifier(CLASSIFIER_DIR)
    extractor = CAMIExtractor(NER_DIR)
    print("Modelli pronti.\n")

    test_sentences = [
        "Quell'avvocato è uno squalo.",
        "Il mio capo è un vulcano di idee.",
        "La notizia lo ha fulminato.",
        "Il cielo è coperto di nuvole.",
        "Ho comprato il pane e il latte.",
        "La sua mente è un computer.",
        "Il computer è sul tavolo.",
    ]

    for sentence in test_sentences:
        print(f"--- Analisi Frase: '{sentence}' ---")
        
        # 1. Usiamo il primo modello per classificare la frase
        classification_result = classifier.predict(sentence)
        label = classification_result['label']
        conf = classification_result['confidence']
        print(f"  -> Classificazione: {label} (Confidenza: {conf:.1%})")
        
        # 2. SE è una metafora, usiamo il secondo modello per estrarre le parti
        if label == "Metafora":
            extraction_result = extractor.extract(sentence)
            arg = extraction_result.get("argomento", "N/A")
            veh = extraction_result.get("veicolo", "N/A")
            print(f"  -> Close Reading: Argomento='{arg}', Veicolo='{veh}'")
    
    print("\n--- Fine Demo ---")