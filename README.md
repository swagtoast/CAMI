# CAMI: Classificatore Automatico di Metafore per l’Italiano

Questo progetto fornisce un sistema completo per:
1. **Addestrare** un modello di classificazione (metafora vs. non metafora) su un dataset annotato.
2. **Valutare** le performance del modello.
3. **Effettuare inferenza** su nuovi testi in italiano, con:
   - Rilevazione automatica di metafore (“metafora”/“non metafora”)
   - Fase *human-in-the-loop* per frasi incerte (0.35 ≤ prob ≤ 0.65)
   - Estrazione di “argomento” e “veicolo” per frasi metaforiche
   - Calcolo dell’indice di figuralità per ciascun testo
   - Rappresentazione vettoriale di “argomento”/“veicolo” in 2D (PCA)
   - Visualizzazione della matrice di attenzione (heatmap)
   - Meccanismo di ri-addestramento automatico ogni 75 annotazioni manuali

## Struttura del progetto

```text
.
├── CAMI_demo.ipynb          # Jupyter Notebook che contiene una demo del software (solo classificazione metafora/letterale)
├── data/                    # Cartella che contiene CAMI_dataset_v2.csv e input di inferenza
│   ├── CAMI_dataset_v2.csv  # Dataset di training
│   └── nuovi_testi/         # Contiene i file .txt per l’inferenza 
├── data_utils.py            # Funzioni per caricamento e preparazione dati
├── train_model.py           # Script di fine-tuning del modello
├── evaluate_model.py        # Script di valutazione su test set
├── semantics.py             # Modulo per misure semantiche (WordNet/OMW)
├── predict.py               # Script di inferenza su nuovi testi (con human-in-the-loop)
├── README.md                # Questo file
├── train.csv                # Generato automaticamente (train/test split)
├── test.csv                 # Generato automaticamente (train/test split)
├── manual_annotations.csv   # Annotazioni raccolte durante l’inferenza
├── risultati_inferenza.csv  # Risultati aggregati dell’inferenza
└── CAMI_dataset_v2.csv      # (Non incluso) Dataset originale: colonne [testo, etichetta, argomento, veicolo]
```


## Requisiti

- Python 3.7+
    
- Librerie Python (installare in Colab o ambiente locale):
    

    `!pip install transformers datasets scikit-learn pandas matplotlib nltk spacy seaborn !python -m spacy download it_core_news_sm`
    
- **Dataset**: posizionare `CAMI_dataset_v2.csv` nella stessa cartella del progetto.
    

## Passaggi per eseguire in Google Colab

1. **Carica il dataset**  
    Carica il file `CAMI_dataset_v2.csv` nella root del Colab (es. tramite l’interfaccia di upload).
    
2. **Eseguire il training**
    
    `!python train_model.py`
    
    - Questo comando:
        
        - Richiama `data_utils.load_and_prepare_data` per creare `train.csv` e `test.csv`.
            
        - Fine-tuna il modello `Musixmatch/umberto-wikipedia-uncased-v1` per 3 epoche.
            
        - Salva il modello in `cami_model/`.
            
3. **Valutare il modello** (consigliato)
    
    `!python evaluate_model.py`
    
    - Carica `cami_model/` e `test.csv`, stampa il classification report e la confusion matrix.
        
4. **Effettuare inferenza su nuovi testi**
    
    - Crea una cartella `data/nuovi_testi/` e carica lì i file `.txt` da analizzare.
        
    - Esegui:
        
        `!python predict.py`

    - Il processo:
        
        - Suddivide ogni file in frasi (NLTK).
            
        - Classifica ogni frase con il modello fine-tunato.
            
        - Per frasi incerte (0.35 ≤ prob ≤ 0.65), viene richiesta annotazione manuale via `input()`.
            
        - Salva le annotazioni in `manual_annotations.csv`.
            
        - Per ogni metafora:
            
            - Estrae argomento/veicolo (sostantivi con similarità minima).
                
            - Calcola e stampa distanza semantica e concretezza.
                
            - Mostra scatter plot 2D dei vettori (PCA).
                
            - Mostra heatmap dell’attenzione (ultimo layer, media over heads).
                
        - Calcola l’indice di figuralità per ciascun file e salva in `risultati_inferenza.csv`.
            
5. **Ri-addestramento automatico**
    
    - Ogni volta che `manual_annotations.csv` raggiunge 75 righe, `predict.py` avvierà automaticamente il ri-addestramento:
        
        - Verranno uniti gli esempi annotati manualmente al training set originale (`train.csv`).
            
        - Il modello verrà ri-addestrato per 1 epoca aggiuntiva e salvato in `cami_model/`.
            
        - Il file `manual_annotations.csv` verrà svuotato.
